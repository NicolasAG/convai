import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import time
import cPickle as pkl


ACTIVATIONS = {
    'swish': lambda x: x * F.sigmoid(x),
    'relu': F.relu,
    'sigmoid': F.sigmoid
}


RNN_GATES = {
    'rnn': torch.nn.RNN,
    'gru': torch.nn.GRU,
    'lstm': torch.nn.LSTM
}


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class QNetwork(torch.nn.Module):
    # Small network:
    # - Get custom encoding of (article, context, candidate)
    # - Feed encodings to a MLP that predicts Q(s,a)
    def __init__(self, input_size, mlp_activation, mlp_dropout):
        """
        Build the q-network to predict q-values
        :param input_size: hidden size of the (article, context, candidate) encodding
        :param mlp_activation: activation function to use in each mlp layer
        :param mlp_dropout: if non-zero, will add a dropout layer in the mlp
        """
        super(QNetwork, self).__init__()

        self.input_size = input_size
        self.mlp_activation = mlp_activation

        self.fc_1 = torch.nn.Linear(self.input_size, self.input_size/2)
        self.fc_2 = torch.nn.Linear(self.input_size/2, self.input_size/2)
        self.fc_3 = torch.nn.Linear(self.input_size/2, self.input_size/4)
        self.fc_4 = torch.nn.Linear(self.input_size/4, 1)

        self.dropout = torch.nn.Dropout(p=mlp_dropout)

        self.init_weights()

    def init_weights(self):
        """
        initialize all weights for all MLPs
        """
        # fully connected parameters
        for fc in [self.fc_1, self.fc_2, self.fc_3, self.fc_4]:
            fc.weight.data.uniform_(-0.1, 0.1)
            fc.bias.fill_(0.0)

    def forward(self, x):
        """
        Perform a forward pass on a batch of (article, dialog, candidate)
        encoddings to predict Q-values.

        :param x: list of custom encoddings to predict Q-values of (article, dialog, candidate) triples
                  torch.Variable with Tensor ~ (batch, input_size)
        """
        out = ACTIVATIONS[self.mlp_activation](self.fc_1(x))
        out = ACTIVATIONS[self.mlp_activation](self.fc_2(out))
        out = ACTIVATIONS[self.mlp_activation](self.fc_3(out))
        out = self.dropout(out)  # dropout layer
        out = self.fc_4(out)  # last layer: no activation
        return out


class DeepQNetwork(torch.nn.Module):
    # Deepest network:
    # - Encodes article into a 2 layer RNN
    # - Encodes dialog history into a 2 layer RNN
    # - Encodes individual candidate response into a 1 layer RNN
    # - Feed encodings to a MLP that predicts Q(s,a)
    def __init__(self, mode, embeddings, fixed_embeddings,
            sentence_hs, sentence_rnn_bidir, sentence_rnn_dropout,
            article_hs, article_rnn_bidir, article_rnn_dropout,
            utterance_hs, utterance_rnn_bidir, utterance_rnn_dropout,
            context_hs, context_rnn_bidir, context_rnn_dropout,
            rnn_gate,
            custom_enc_hs, mlp_activation, mlp_dropout):
        """
        Build the deep q-network to predict q-values
        :param mode: one of 'rnn+mlp' or 'rnn+rnn+mlp' to decide if we have one
                        or two hierachical RNNs
        :param embeddings : |vocab| x |token_hs| numpy array
        :param fixed_embeddings : freeze word embeddings during training

        :param sentence_hs : hidden size of sentence vectors in the article
        :param sentence_rnn_bidir : bidirectional sentence rnn
        :param sentence_rnn_dropout : dropout rate of sentence rnn

        :param article_hs : hidden size of full article vector
        :param article_rnn_bidir : bidirectional article rnn
        :param article_rnn_dropout : dropout rate of article rnn

        :param utterance_hs: hidden size of one speaker utterance in the dialog
        :param utterance_rnn_bidir : bidirectional utterance rnn
        :param utterance_rnn_dropout : dropout rate of utterance rnn

        :param context_hs : hidden size of full dialog vector
        :param context_rnn_bidir : bidirectional context rnn
        :param context_rnn_dropout : dropout rate of context rnn

        :param rnn_gate: one of 'rnn', 'gru', or 'lstm'

        :param custom_enc_hs: hidden size of custom encodding of (article, dialog, candidate) triples
        :param mlp_activation: activation function to use in each mlp layer
        :param mlp_dropout: if non-zero, will add a dropout layer in the mlp
        """
        super(DeepQNetwork, self).__init__()

        self.sentence_hs = sentence_hs   # article sentence vector size
        self.article_hs = article_hs     # full article vector size
        self.utterance_hs = utterance_hs # dialog utterance vector size
        self.context_hs = context_hs     # full dialog vector size
        self.gate = rnn_gate

        self.mlp_activation = mlp_activation

        embeddings = torch.from_numpy(embeddings).float()  # convert nupy array to torch float tensor
        self.embed = torch.nn.Embedding(embeddings.size(0), embeddings.size(1)) # embedding layer ~(vocab x token_hs)
        self.embed.weight = torch.nn.Parameter(embeddings, requires_grad=(not fixed_embeddings))

        self.sentence_rnn = RNN_GATES[self.gate](input_size = self.embed.embedding_dim,
                                                 hidden_size = self.sentence_hs,
                                                 batch_first = True, # ~(batch, seq, hs)
                                                 dropout = sentence_rnn_dropout,
                                                 bidirectional = sentence_rnn_bidir)

        self.article_rnn = RNN_GATES[self.gate](input_size = self.sentence_hs,
                                                hidden_size = self.article_hs,
                                                batch_first = True, # ~(batch, seq, hs)
                                                dropout = article_rnn_dropout,
                                                bidirectional = article_rnn_bidir)
        if mode == 'rnn+mlp':
            self.utterance_rnn = self.sentence_rnn
        else:
            self.utterance_rnn = RNN_GATES[self.gate](input_size = self.embed.embedding_dim,
                                                      hidden_size = self.utterance_hs,
                                                      batch_first = True, # ~(batch, seq, hs)
                                                      dropout = utterance_rnn_dropout,
                                                      bidirectional = utterance_rnn_bidir)
        if mode == 'rnn+mlp':
            self.context_rnn = self.article_rnn
        else:
            self.context_rnn = RNN_GATES[self.gate](input_size = self.utterance_hs,
                                                    hidden_size = self.context_hs,
                                                    batch_first = True, # ~(batch, seq, hs)
                                                    dropout = context_rnn_dropout,
                                                    bidirectional = context_rnn_bidir)

        self.state_space = self.article_hs + self.context_hs  # state = (article, context)

        self.fc_1 = torch.nn.Linear(self.state_space, self.state_space/2)
        self.fc_2 = torch.nn.Linear(self.state_space/2, self.state_space/4)
        self.fc_3 = torch.nn.Linear(self.state_space/4, self.state_space/4)

        self.action_space = self.utterance_hs + custom_enc_hs  # action = (candidate, custom_features)
        self.advantage_space = self.state_space/4 + self.action_space # advantage = (state, action)

        # layers to predict value function V(s)
        self.fc_value_1 = torch.nn.Linear(self.state_space/4, self.state_space/8)
        self.fc_value_2 = torch.nn.Linear(self.state_space/8, 1)
        # layers to predict advantage function A(s, a)
        self.fc_adv_1 = torch.nn.Linear(self.advantage_space, self.advantage_space/2)
        self.fc_adv_2 = torch.nn.Linear(self.advantage_space/2, self.advantage_space/4)
        self.fc_adv_3 = torch.nn.Linear(self.advantage_space/4, 1)

        self.dropout = torch.nn.Dropout(p=mlp_dropout)

        self.init_weights()

    def init_weights(self):
        """
        initialize all weights for all RNNs and all MLPs
        """
        # rnn parameters
        for rnn in [self.sentence_rnn, self.article_rnn,
                    self.utterance_rnn, self.context_rnn]:
            # weights ~ N(0, 1) && bias = 0
            for name, param in rnn.named_parameters():
                if name.startswith('weight'):
                    param.data.normal_(0.0, 0.1)
                elif name.startswith('bias'):
                    param.data.fill_(0.0)
                else:
                    print "default initialization for parameter %s" % name

        # fully connected parameters
        for fc in [self.fc_1, self.fc_2, self.fc_3,
                   self.fc_value_1, self.fc_value_2,
                   self.fc_adv_1, self.fc_adv_2, self.fc_adv_3]:
            fc.weight.data.uniform_(-0.1, 0.1)
            fc.bias.fill_(0.0)

    def _encode_with(self, rnn, sequences, lengths):
        """
        Encode sequences with a given rnn
        :param rnn: RNN to use to encode
        :param sequences: sequence to encode ~(bs, max_length, hs)
        :param lengths: length of each sequence ~ (bs)
        """
        # pack sequences to avoid calculation on padded elements
        # see: https://discuss.pytorch.org/t/understanding-pack-padded-sequence-and-pad-packed-sequence/4099/5
        packed_seq = pack_padded_sequence(sequences, lengths, batch_first=True)  # convert embeddings to PackedSequence

        # output ~(batch, seq_length, hidden_size*num_directions) <-- output features (h_t) from the last layer of the RNN, for each t.
        # hidden ~(num_layers*num_directions, batch, hidden_size) <-- tensor containing the hidden state for t=seq_len
        output, hidden = rnn(packed_seq)  # encode sequences
        # lstm returns hidden state AND cell state
        if self.gate == 'lstm':
            hidden, cell = hidden

        # convert back output PackedSequence to tensor
        output, lengths_ = pad_packed_sequence(output, batch_first=True)
        assert lengths == lengths_

        # grab the encodding of the sentence, not the padded part!
        encodding = output[
            range(output.size(0)),  # take each sentence
            list(map(lambda l: l-1, lengths)),  # at their last index (ie: length-1)
            :  # take full encodding
        ] # ~ (bs, hs)

        return encodding


    def forward(self,
                sentences, article_lengths, sent_lengths,
                utterances, context_lengths, utt_lengths,
                candidates, cand_lengths, custom_enc):
        """
        Perform a forward pass on a batch of (article, dialog, candidate) to
        predict Q-values.

        :param sentences: list of sentences to encode to form a bunch of articles
                          torch.Variable with Tensor ~ (batch x #sent/article, max_sent_len)
        :param article_lengths: number of sentences for each article
                                torch Tensor ~ (batch)
        :param sent_lengths: number of tokens for each sentence
                             torch Tensor ~ (batch x #sent/article)

        :param utterances: list of utterances to encode to form a bunch of contexts
                           torch.Variable with Tensor ~ (batch x #utt/context, max_utt_len)
        :param context_lengths: number of utterances for each context
                                torch Tensor ~ (batch)
        :param utt_lengths: number of tokens for each utterance
                            torch Tensor ~ (batch x #utt/context)

        :param candidates: list of candidate responses
                           torch.Variable with Tensor ~ (batch, max_cand_length)
        :param cand_lengths: number of tokens for each candidate
                             torch Tensor ~ (batch)

        :param custom_enc: list of custom (article, dialog, candidate) triples encoddings
                           torch.Variable with tensor ~ (batch, custom_enc_hs)
        """
        # encode article sentences
        sentences_emb = self.embed(sentences)  # ~(bs x #sent, max_len, embed)
        sentences_enc = self._encode_with(self.sentence_rnn,
                                          sentences_emb,
                                          sent_lengths) # ~(bs x #sent, hs)
        # populate article embeddings
        max_article_len = max(article_lengths)
        article_emb = to_var(torch.zeros(len(article_lengths),
                                         max_article_len,
                                         sentences_enc.size(1)))
        start = 0
        for idx, length in enumerate(article_lengths):
            article_emb[idx, 0:length, :] = sentences_enc[start: start+length]
            start += length
        # encode articles
        article_enc = self._encode_with(self.article_rnn,
                                        article_emb,
                                        article_lengths) # ~(bs, article_hs)


        # encode context utterances
        utterances_emb = self.embed(utterances)
        utterances_enc = self._encode_with(self.utterance_rnn,
                                           utterances_emb,
                                           utt_lengths) # ~(bs x #utt, hs)
        # populate context embeddings
        max_context_len = max(context_lengths)
        context_emb = to_var(torch.zeros(len(context_lengths),
                                         max_context_len,
                                         utterances_enc.size(1)))
        start = 0
        for idx, length in enumerate(context_lengths):
            context_emb[idx, 0:length, :] = utterances_enc[start: start+length]
            start += length
        # encode contexts
        context_enc = self._encode_with(self.context_rnn,
                                        context_emb,
                                        context_lengths) # ~(bs, context_hs)


        # encode candidate responses
        candidate_emb = self.embed(candidates)  # ~(bs, max_len, embed)
        candidate_enc = self._encode_with(self.utterance_rnn,
                                          candidate_emb,
                                          cand_lengths) # ~(bs, hs)


        # Predict Q-values based on article_enc, context_enc, candidate_enc, and custom_enc
        state_enc = torch.cat((article_enc, context_enc), 1)  # ~ (bs, hs)
        state_enc = ACTIVATIONS[self.mlp_activation](self.fc_1(state_enc))
        state_enc = ACTIVATIONS[self.mlp_activation](self.fc_2(state_enc))
        state_enc = ACTIVATIONS[self.mlp_activation](self.fc_3(state_enc))

        # Dueling Q-network: value prediction
        value = ACTIVATIONS[self.mlp_activation](self.fc_value_1(state_enc))
        value = self.dropout(value)  # dropout layer
        value = self.fc_value_2(value) # last layer: no activation

        # Dueling Q-network: advantage prediction
        advantage = torch.cat((state_enc, candidate_enc, custom_enc), 1) # ~(bs, hs)
        advantage = ACTIVATIONS[self.mlp_activation](self.fc_adv_1(advantage))
        advantage = ACTIVATIONS[self.mlp_activation](self.fc_adv_2(advantage))
        advantage = self.dropout(advantage)  # dropout layer
        advantage = self.fc_adv_3(advantage) # last layer: no activation

        q_value = value + advantage  # ~(bs, 1)
        return q_value

