import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def to_tensor(x, type=torch.Tensor):
    if torch.cuda.is_available():
        return type(x).cuda()
    else:
        return type(x)


class QNetwork(torch.nn.Module):
    # Small network:
    # - Get custom encoding of (article, context, candidate)
    # - Feed encodings to a MLP that predicts Q(s,a)
    def __init__(self, input_size, mlp_activation, mlp_dropout, out_size):
        """
        Build the q-network to predict q-values
        :param input_size: hidden size of the (article, context, candidate) encoding
        :param mlp_activation: activation function to use in each mlp layer
        :param mlp_dropout: if non-zero, will add a dropout layer in the mlp
        :param out_size: 1 if predicting q-values, 2 if classifying rewards
        """
        super(QNetwork, self).__init__()

        self.input_size = input_size
        self.mlp_activation = mlp_activation
        assert out_size in [1, 2]

        self.fc_1 = torch.nn.Linear(self.input_size, self.input_size/2)
        self.fc_2 = torch.nn.Linear(self.input_size/2, self.input_size/2)
        self.fc_3 = torch.nn.Linear(self.input_size/2, self.input_size/4)
        self.fc_4 = torch.nn.Linear(self.input_size/4, out_size)

        self.dropout = torch.nn.Dropout(p=mlp_dropout)

        self.init_weights()

    def init_weights(self):
        """
        initialize all weights for all MLPs according to the method
        described in "Understanding the difficulty of training deep feedforward neural networks"
        - Glorot, X. & Bengio, Y. (2010), using a normal distribution.
        Also known as Glorot initialisation.
        """
        # fully connected parameters
        for fc in [self.fc_1, self.fc_2, self.fc_3, self.fc_4]:
            if self.mlp_activation in ['relu', 'swish']:
                tmp_activation = 'relu'  # consider swish as a ReLU
                mu = 0.1
            else:
                tmp_activation = self.mlp_activation
                mu = 0.0
            gain = torch.nn.init.calculate_gain(tmp_activation)

            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(fc.weight.data)
            std = gain * math.sqrt(2.0 / (fan_in + fan_out))
            fc.weight.data.normal_(mu, std)

            fc.bias.data.fill_(0.0)

    def forward(self, x):
        """
        Perform a forward pass on a batch of (article, dialog, candidate)
        encodings to predict Q-values.

        :param x: list of custom encodings to predict Q-values of (article, dialog, candidate) triples
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
                 custom_enc_hs, mlp_activation, mlp_dropout, out_size):
        """
        Build the deep q-network to predict q-values
        :param mode: one of 'rnn+mlp' or 'rnn+rnn+mlp' to decide if we have one
                        or two hierarchical RNNs
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

        :param custom_enc_hs: hidden size of custom encoding of (article, dialog, candidate) triples
        :param mlp_activation: activation function to use in each mlp layer
        :param mlp_dropout: if non-zero, will add a dropout layer in the mlp
        :param out_size: 1 if predicting q-values, 2 if classifying rewards
        """
        super(DeepQNetwork, self).__init__()

        self.sentence_hs = sentence_hs    # article sentence vector size
        self.article_hs = article_hs      # full article vector size
        self.utterance_hs = utterance_hs  # dialog utterance vector size
        self.context_hs = context_hs      # full dialog vector size
        self.gate = rnn_gate

        self.mlp_activation = mlp_activation
        assert out_size in [1, 2]

        embeddings = torch.from_numpy(embeddings).float()  # convert numpy array to torch float tensor
        self.embed = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))  # embedding layer ~(vocab x token_hs)
        self.embed.weight = torch.nn.Parameter(embeddings, requires_grad=(not fixed_embeddings))

        self.sentence_rnn = RNN_GATES[self.gate](input_size=self.embed.embedding_dim,
                                                 hidden_size=self.sentence_hs,
                                                 batch_first=True,  # ~(batch, seq, hs)
                                                 dropout=sentence_rnn_dropout,
                                                 bidirectional=sentence_rnn_bidir)

        self.article_rnn = RNN_GATES[self.gate](input_size=self.sentence_hs,
                                                hidden_size=self.article_hs,
                                                batch_first=True,  # ~(batch, seq, hs)
                                                dropout=article_rnn_dropout,
                                                bidirectional=article_rnn_bidir)
        if mode == 'rnn+mlp':
            self.utterance_rnn = self.sentence_rnn
        else:
            self.utterance_rnn = RNN_GATES[self.gate](input_size=self.embed.embedding_dim,
                                                      hidden_size=self.utterance_hs,
                                                      batch_first=True,  # ~(batch, seq, hs)
                                                      dropout=utterance_rnn_dropout,
                                                      bidirectional=utterance_rnn_bidir)
        if mode == 'rnn+mlp':
            self.context_rnn = self.article_rnn
        else:
            self.context_rnn = RNN_GATES[self.gate](input_size=self.utterance_hs,
                                                    hidden_size=self.context_hs,
                                                    batch_first=True,  # ~(batch, seq, hs)
                                                    dropout=context_rnn_dropout,
                                                    bidirectional=context_rnn_bidir)

        self.state_space = self.article_hs + self.context_hs  # state = (article, context)

        self.fc_1 = torch.nn.Linear(self.state_space, self.state_space/2)
        self.fc_2 = torch.nn.Linear(self.state_space/2, self.state_space/4)
        self.fc_3 = torch.nn.Linear(self.state_space/4, self.state_space/4)

        self.action_space = self.utterance_hs + custom_enc_hs  # action = (candidate, custom_features)
        self.advantage_space = self.state_space/4 + self.action_space  # advantage = (state, action)

        # layers to predict value function V(s)
        self.fc_value_1 = torch.nn.Linear(self.state_space/4, self.state_space/8)
        self.fc_value_2 = torch.nn.Linear(self.state_space/8, out_size)
        # layers to predict advantage function A(s, a)
        self.fc_adv_1 = torch.nn.Linear(self.advantage_space, self.advantage_space/2)
        self.fc_adv_2 = torch.nn.Linear(self.advantage_space/2, self.advantage_space/4)
        self.fc_adv_3 = torch.nn.Linear(self.advantage_space/4, out_size)

        self.dropout = torch.nn.Dropout(p=mlp_dropout)

        self.init_weights()

    def init_weights(self):
        """
        initialize all weights for all RNNs and all MLPs according to the method
        described in "Understanding the difficulty of training deep feedforward neural networks"
        - Glorot, X. & Bengio, Y. (2010), using a normal distribution.
        Also known as Glorot initialisation.
        """
        # rnn parameters
        for rnn in [self.sentence_rnn, self.article_rnn,
                    self.utterance_rnn, self.context_rnn]:
            if self.gate == 'rnn':
                self._init_rnn_params(rnn)
            elif self.gate == 'gru':
                self._init_gru_params(rnn)
            elif self.gate == 'lstm':
                self._init_lstm_params(rnn)
            else:
                print "ERROR: unknown recurrent network gate: %s" % self.gate

        # fully connected parameters
        for fc in [self.fc_1, self.fc_2, self.fc_3,
                   self.fc_value_1, self.fc_value_2,
                   self.fc_adv_1, self.fc_adv_2, self.fc_adv_3]:

            if self.mlp_activation in ['relu', 'swish']:
                tmp_activation = 'relu'  # consider swish as a ReLU
                mu = 0.1
            else:
                tmp_activation = self.mlp_activation
                mu = 0.0
            gain = torch.nn.init.calculate_gain(tmp_activation)

            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(fc.weight)
            std = gain * math.sqrt(2.0 / (fan_in + fan_out))
            fc.weight.data.normal_(mu, std)

            fc.bias.data.fill_(0.0)

    def _init_rnn_params(self, rnn):
        """
        Initialise weights of given RNN
        :param rnn: the rnn to initialise
        """
        for name, param in rnn.named_parameters():
            if name.startswith('weight'):
                gain = torch.nn.init.calculate_gain('tanh')  # RNN activation is tanh by default
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(param.data)
                std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                param.data.normal_(0.0, std)
            elif name.startswith('bias'):
                param.data.fill_(0.0)
            else:
                print "default initialization for parameter %s" % name

    def _init_gru_params(self, rnn):
        """
        Initialise weights of given GRU
        :param rnn: the gru to initialise
        """
        for name, param in rnn.named_parameters():
            '''
            weight_ih_l[k] - (W_ir|W_iz|W_in), of shape (3*hidden_size x input_size)
            weight_hh_l[k] - (W_hr|W_hz|W_hn), of shape (3*hidden_size x hidden_size)
            r & z are sigmoid (gain=1)
            n is tanh (gain=5/3)
            let's take average gain : (1+1+5/3) / 3
            '''
            if name.startswith('weight'):
                gain = (1. + 1. + 5./3.) / 3.
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(param.data)
                std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                param.data.normal_(0.0, std)
            elif name.startswith('bias'):
                param.data.fill_(0.0)
            else:
                print "default initialization for parameter %s" % name

    def _init_lstm_params(self, rnn):
        """
        Initialise weights of given LSTM
        :param rnn: the LSTM to initialise
        """
        for name, param in rnn.named_parameters():
            '''
            weight_ih_l[k] - (W_ii|W_if|W_ig|W_io), of shape (4*hidden_size x input_size)
            weight_hh_l[k] - (W_hi|W_hf|W_hg|W_ho), of shape (4*hidden_size x hidden_size)
            i & f & o are sigmoid (gain=1)
            g is tanh (gain=5/3)
            let's take average gain : (1+1+1+5/3) / 4
            '''
            if name.startswith('weight'):
                gain = (1. + 1. + 1. + 5./3.) / 4.
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(param.data)
                std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                param.data.normal_(0.0, std)
            elif name.startswith('bias'):
                param.data.fill_(0.0)
            else:
                print "default initialization for parameter %s" % name

    def _sort_by_length(self, sequence, lengths):
        """
        Sort a Tensor content by the length of each sequence
        :param sequence: LongTensor ~ (bs, ...)
        :param lengths: LongTensor ~ (bs)
        :return: LongTensor with sorted content & list of sorted indices
        """
        # Sort lengths
        _, sorted_idx = lengths.sort(descending=True)
        # Sort variable tensor by indexing at the sorted_idx
        sequence = sequence.index_select(0, sorted_idx)
        return sequence, sorted_idx

    def _unsort_tensor(self, sorted_tensor, sorted_idx):
        """
        Revert a Tensor to its original order. Undo the `_sort_by_length` function
        :param sorted_tensor: Tensor with content sorted by length ~ (bs, hs)
        :param sorted_idx: list of ordered indices ~ (bs)
        :return: Unsorted Tensor
        """
        # Sort the sorted idx to get the original positional idx
        _, pos_idx = sorted_idx.sort()
        # Unsort the tensor
        original = sorted_tensor.index_select(0, pos_idx)
        return original

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

        # grab the encoding of the sentence, not the padded part!
        encoding = output[
            range(output.size(0)),  # take each sentence
            list(map(lambda l: l-1, lengths)),  # at their last index (ie: length-1)
            :  # take full encoding
        ]  # ~ (bs, hs)

        return encoding

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

        :param custom_enc: list of custom (article, dialog, candidate) triples encodings
                           torch.Variable with tensor ~ (batch, custom_enc_hs)
        """
        ###
        # ARTICLE
        ###
        # sort sentences by length
        sorted_sentences, sorted_indices = self._sort_by_length(sentences, sent_lengths)
        # encode article sentences
        sentences_emb = self.embed(sorted_sentences)  # ~(bs x #sent, max_len, embed)
        sentences_enc = self._encode_with(self.sentence_rnn,
                                          sentences_emb,
                                          sorted(sent_lengths.data, reverse=True))  # ~(bs x #sent, hs)
        # recover original order of sentences
        sentences_enc = self._unsort_tensor(sentences_enc, sorted_indices)

        # populate article embeddings
        article_emb = to_var(torch.zeros(len(article_lengths),
                                         max(article_lengths.data),
                                         sentences_enc.size(1)))
        start = 0
        for idx, length in enumerate(article_lengths.data):
            article_emb[idx, 0:length, :] = sentences_enc[start: start+length]
            start += length
        # sort articles by length
        sorted_article_embs, sorted_indices = self._sort_by_length(article_emb, article_lengths)
        # encode articles
        article_enc = self._encode_with(self.article_rnn,
                                        sorted_article_embs,
                                        sorted(article_lengths.data, reverse=True))  # ~(bs, article_hs)
        # recover original article order
        article_enc = self._unsort_tensor(article_enc, sorted_indices)

        ###
        # CONTEXTS
        ###
        # sort utterances by length
        sorted_utterances, sorted_indices = self._sort_by_length(utterances, utt_lengths)
        # encode context utterances
        utterances_emb = self.embed(sorted_utterances)  # ~(bs, max_len, embed)
        utterances_enc = self._encode_with(self.utterance_rnn,
                                           utterances_emb,
                                           sorted(utt_lengths.data, reverse=True))  # ~(bs x #utt, hs)
        # recover original order of utterances
        utterances_enc = self._unsort_tensor(utterances_enc, sorted_indices)

        # populate context embeddings
        context_emb = to_var(torch.zeros(len(context_lengths),
                                         max(context_lengths.data),
                                         utterances_enc.size(1)))
        start = 0
        for idx, length in enumerate(context_lengths.data):
            context_emb[idx, 0:length, :] = utterances_enc[start: start+length]
            start += length
        # sort contexts by length
        sorted_context_embs, sorted_indices = self._sort_by_length(context_emb, context_lengths)
        # encode contexts
        context_enc = self._encode_with(self.context_rnn,
                                        sorted_context_embs,
                                        sorted(context_lengths.data, reverse=True))  # ~(bs, context_hs)
        # recover original order of contexts
        context_enc = self._unsort_tensor(context_enc, sorted_indices)

        ###
        # CANDIDATE RESPONSES
        ###
        # sort candidate responses by length
        sorted_candidates, sorted_indices = self._sort_by_length(candidates, cand_lengths)
        # encode candidate responses
        candidate_emb = self.embed(sorted_candidates)  # ~(bs, max_len, embed)
        candidate_enc = self._encode_with(self.utterance_rnn,
                                          candidate_emb,
                                          sorted(cand_lengths.data, reverse=True))  # ~(bs, hs)
        # recover original order of candidate responses
        candidate_enc = self._unsort_tensor(candidate_enc, sorted_indices)

        ###
        # Q-VALUES
        ###
        # Encode state-encoding : (article, context)
        state_enc = torch.cat((article_enc, context_enc), 1)  # ~ (bs, hs)
        state_enc = ACTIVATIONS[self.mlp_activation](self.fc_1(state_enc))
        state_enc = ACTIVATIONS[self.mlp_activation](self.fc_2(state_enc))
        state_enc = ACTIVATIONS[self.mlp_activation](self.fc_3(state_enc))

        # Dueling Q-network: value prediction
        value = ACTIVATIONS[self.mlp_activation](self.fc_value_1(state_enc))
        value = self.dropout(value)  # dropout layer
        value = self.fc_value_2(value)  # last layer: no activation

        # Dueling Q-network: advantage prediction
        # input to advantage prediction : (state, candidate, custom_enc)
        advantage = torch.cat((state_enc, candidate_enc, custom_enc), 1)  # ~(bs, hs)
        advantage = ACTIVATIONS[self.mlp_activation](self.fc_adv_1(advantage))
        advantage = ACTIVATIONS[self.mlp_activation](self.fc_adv_2(advantage))
        advantage = self.dropout(advantage)   # dropout layer
        advantage = self.fc_adv_3(advantage)  # last layer: no activation

        q_value = value + advantage  # ~(bs, out_size)
        return q_value
