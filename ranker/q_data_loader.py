import torch
import torch.utils.data as data
import nltk


class ConversationDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, json, vocab, mode):
        """
        Set the json file and vocabulary wrapper
        :param json: json data file
        :param vocab: Vocabulary wrapper
        :param mode: one of 'mlp', 'mlp+rnn', 'mlp+rnn+rnn' to indicate which QNetwork to use
        """
        self.json_data = json
        self.vocab = vocab
        self.mode = mode
        self.ids = []  # map from id to (article, position)
        for article, entries in self.json_data.iteritems():
            for i, _ in enumerate(entries):
                self.ids.append((article, i))

    def string_to_idx(self, phrase, start_tag, end_tag):
        """
        Convert a word phrase into a word_idx phrase
        :param phrase: the phrase to convert
        :param start_tag: string tag to represent the start of the phrase
        :param end_tag: string tag to represent the end of the phrase
        :return: list of word_tokens
        """
        indices = []
        tokens = nltk.tokenize.word_tokenize(phrase)
        indices.append(self.vocab(start_tag))
        indices.extend([self.vocab(tok) for tok in tokens])
        indices.append(self.vocab(end_tag))
        return torch.Tensor(indices)

    def convert_big_string_to_idx(self, phrases, start_tag, end_tag):
        """
        Convert list of string phrases to list of word_idx phrases.
        :param phrases: the big string to convert. Assumed to be a list of phrases.
        :param start_tag: string tag to represent the start of a phrase
        :param end_tag: string tag to represent the end of a phrase
        :return: 2D tensor of word indices, number of phrases, length of each phrase
        """
        indices = []
        lengths = []
        for phrase in phrases:
            phrase_idx = self.string_to_idx(phrase, start_tag, end_tag)
            indices.append(phrase_idx)
            lengths.append(len(phrase_idx))
        return indices, len(indices), lengths

    def _rescale_rewards(self, r, quality):
        if r > 0:
            # rescale based on quality measure
            if quality == 5:  # quality = 5 /5
                r *= 1.
            elif quality > 2:  # quality = 3|4 /5
                r *= 0.8
            else:  # quality = 1|2 /5
                r *= 0.2
        return r

    def __getitem__(self, item):
        """
        Returns one data tuple.
        if only using an MLP, return (custom_enc, reward, next_custom_enc)
        if using RNN and MLP, return (
            article_sentences, number of sentences, sentence lengths,
            context_utterances, number of turns, turn length,
            candidate_tokens, number of tokens,
            custom_enc, reward,
            next_context_utterances, number of turns, turn length,
            next_candidates, number of candidates, candidate length,
            next_custom_encs
        )
        """
        article, idx = self.ids[item]
        entry = self.json_data[article][idx]
        '''
        'chat_id': <string>,
        'state': <list of strings> ie: context,
        'action': {
            'candidate': <string>  ie: candidate,
            'custom_enc': <list of float>
        },
        'reward': <int {0,1}>,
        'next_state': <list of strings || None> ie: next_context,
        'next_actions': <list of actions || None> ie: next possible actions
        'quality': <int {1,2,3,4,5}>,
        '''
        context = entry['state']
        candidate = entry['action']['candidate']
        custom_enc = entry['action']['custom_enc']
        reward = self._rescale_rewards(entry['reward'], entry['quality'])

        next_state = entry['next_state']
        next_candidates = [a['candidate'] for a in entry['next_actions']]
        next_custom_encs = [a['custom_enc'] for a in entry['next_actions']]


        if self.mode == 'mlp':
            # in a simple MLP setting, only need the custom encoding, the reward, the next possible custom encodings
            return custom_encs, rewards, next_custom_encs

        else:
            # convert from string to idx
            article_sentences = nltk.tokenize.sent_tokenize(article)
            article, n_sent, l_sent = self.convert_big_string_to_idx(
                article_sentences, start_tag='<sos>', end_tag='<eos>'
            )
            context, n_turn, l_turn = self.convert_big_string_to_idx(
                context,
                start_tag='<sos>' if self.mode == 'rnn+mlp' else '<sot>',
                end_tag='<eos>' if self.mode == 'rnn+mlp' else '<eot>'
            )
            candidate = self.string_to_idx(
                candidate,
                start_tag='<sos>' if self.mode == 'rnn+mlp' else '<sot>',
                end_tag='<eos>' if self.mode == 'rnn+mlp' else '<eot>'
            )

            next_state, n_nextS_turn, l_nextS_turn = self.convert_big_string_to_idx(
                next_state,
                start_tag='<sos>' if self.mode == 'rnn+mlp' else '<sot>',
                end_tag='<eos>' if self.mode == 'rnn+mlp' else '<eot>'
            )
            next_candidates, n_next_candidate, l_next_candidate = self.convert_big_string_to_idx(
                next_candidates,
                start_tag='<sos>' if self.mode == 'rnn+mlp' else '<sot>',
                end_tag='<eos>' if self.mode == 'rnn+mlp' else '<eot>'
            )
            return article, n_sent, l_sent,\
                context, n_turn, l_turn,\
                candidate, custom_encs, rewards,\
                next_state, n_nextS_turn, l_nextS_turn,\
                next_candidates, n_next_candidate, l_next_candidate,\
                next_custom_encs


    def __len__(self):
        return len(self.ids)


Q_NETWORK_MODE = None


def collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples:
    :param data: list of tuples depending on the q-network mode
    :return: what the neural networks expects:
        this is what QNetwork expects:

            list of custom encodings of (article, dialog, candidate) triples
            torch.Variable with Tensor ~ (batch, custom_hs)

        this is what DeepQNetwork forward expects:

            list of sentences to encode to form a bunch of articles
            torch.Variable with Tensor ~ (batch x #sent/article, max_sent_len)

            number of sentences for each article
            torch Tensor ~ (batch)

            number of tokens for each sentence
            torch Tensor ~ (batch x #sent/article)
            ---
            list of utterances to encode to form a bunch of contexts
            torch.Variable with Tensor ~ (batch x #utt/context, max_utt_len)

            number of utterances for each context
            torch Tensor ~ (batch)

            number of tokens for each utterance
            torch Tensor ~ (batch x #utt/context)
            ---
            list of candidate responses
            torch.Variable with Tensor ~ (batch, max_cand_length)

            number of tokens for each candidate
            torch Tensor ~ (batch)
            ---
            list of custom (article, dialog, candidate) triples encodings
            torch.Variable with tensor ~ (batch, custom_enc_hs)
    """
    assert Q_NETWORK_MODE is not None
    if Q_NETWORK_MODE == 'mlp':
        custom_encs, rewards = zip(*data)

        # Merge custom encodings (from tuple of 1D tensor to 2D tensor)
        custom_encs = torch.stack(custom_encs, 0)  # ~ (batch, custom_hs)

        return custom_encs, torch.Tensor(rewards)

    else:
        articles, n_sents, l_sents, \
            contexts, n_turns, l_turns, \
            candidates, n_tokens, \
            custom_encs, rewards = zip(*data)

        # n_sents, n_turns, n_tokens, rewards can be return as is: ~(batch_size)

        # Flatten length of each sentence (from tuple of 1D list to 1D list)
        # we want the number of tokens for each sentence ~ (batch x #sent/article)
        l_sents = [length for sentences in l_sents for length in sentences]
        l_turns = [length for turns in l_turns for length in turns]

        # Merge sentences (from tuple of list of 1D Tensor to 2D tensor):
        # from tuple of list of sentences =to=> (batch x n_sentences, max_len)
        articles_tensor = torch.zeros(len(l_sents), max(l_sents)).long()
        i = 0
        for artcl in articles:
            for sent in artcl:
                end = l_sents[i]
                articles_tensor[i, :end] = sent[:end]
                i += 1
        # Merge utterances (from tuple of list of 1D Tensor to 2D tensor):
        # from tuple of list of turns =to=> (batch x n_turns, max_len)
        contexts_tensor = torch.zeros(len(l_turns), max(l_turns)).long()
        i = 0
        for cntxt in contexts:
            for turn in cntxt:
                end = l_turns[i]
                contexts_tensor[i, :end] = turn[:end]
                i += 1

        # Merge list of candidate responses ~ (batch, max_cand_length)
        candidates_tensor = torch.zeros(len(candidates), max(n_tokens)).long()
        for i, cand in enumerate(candidates):
            end = n_tokens[i]
            candidates_tensor[i, :end] = cand[:end]

        # Merge custom encodings (from tuple of 1D tensor to 2D tensor)
        custom_encs = torch.stack(custom_encs, 0)  # ~ (batch, custom_hs)

        return articles_tensor, n_sents, l_sents, \
            contexts_tensor, n_turns, l_turns, \
            candidates_tensor, n_tokens, \
            custom_encs, torch.Tensor(rewards)


def get_loader(json, vocab, q_net_mode, batch_size, shuffle, num_workers):
    global Q_NETWORK_MODE
    Q_NETWORK_MODE = q_net_mode

    conv = ConversationDataset(json, vocab, q_net_mode)
    data_loader = data.DataLoader(dataset=conv,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader
