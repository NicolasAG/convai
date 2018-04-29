import torch
import torch.utils.data as data
import nltk
import logging

logger = logging.getLogger(__name__)


class ConversationDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, json, vocab, mode, rescale_rewards):
        """
        Set the json file and vocabulary wrapper
        :param json: json data file
        :param vocab: Vocabulary wrapper
        :param mode: one of 'mlp', 'mlp+rnn', 'mlp+rnn+rnn' to indicate which QNetwork to use
        :param rescale_rewards: replace rewards from 0/1 to 0, 0.2, 0.4, etc...
        """
        self.json_data = json
        self.vocab = vocab
        self.mode = mode
        self.rescale_rewards = rescale_rewards
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
        (
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
        if self.rescale_rewards:
            reward = self._rescale_rewards(entry['reward'], entry['quality'])
        else:
            reward = entry['reward']

        next_state = entry['next_state']
        if entry['next_actions']:
            next_candidates = [a['candidate'] for a in entry['next_actions']]
            next_custom_encs = [a['custom_enc'] for a in entry['next_actions']]
        else:
            next_candidates = None
            next_custom_encs = None

        # if self.mode == 'mlp':
        #     # in a simple MLP setting, only need the custom encoding, the reward, the next possible custom encodings
        #     return torch.Tensor(custom_enc), reward, next_custom_encs
        #     #          ~(enc_size)            ~(1)   ~(n_actions, enc_size)
        # else:

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

        if next_state:
            next_state, n_next_turn, l_next_turn = self.convert_big_string_to_idx(
                next_state,
                start_tag='<sos>' if self.mode == 'rnn+mlp' else '<sot>',
                end_tag='<eos>' if self.mode == 'rnn+mlp' else '<eot>'
            )  # list of Tensor. Each Tensor is an utterance
        else:
            next_state = None
            n_next_turn = 0
            l_next_turn = []

        if next_candidates:
            next_candidates, n_next_candidate, l_next_candidate = self.convert_big_string_to_idx(
                next_candidates,
                start_tag='<sos>' if self.mode == 'rnn+mlp' else '<sot>',
                end_tag='<eos>' if self.mode == 'rnn+mlp' else '<eot>'
            )  # list of Tensors. Each Tensor is a candidate
        else:
            next_candidates = None
            n_next_candidate = 0
            l_next_candidate = []

        return article, n_sent, l_sent, \
            context, n_turn, l_turn, \
            candidate, len(candidate), \
            torch.Tensor(custom_enc), reward, \
            next_state, n_next_turn, l_next_turn, \
            next_candidates, n_next_candidate, l_next_candidate, \
            next_custom_encs


    def __len__(self):
        return len(self.ids)


Q_NETWORK_MODE = None


def collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples:
    :param data: list of tuples depending on the q-network mode
    :return: what the neural networks expects
    """
    articles, n_sents, l_sents, \
        contexts, n_turns, l_turns, \
        candidates, n_tokens, \
        custom_encs, rewards, \
        next_states, n_next_turns, l_next_turns, \
        next_candidates, n_next_candidates, l_next_candidates, \
        next_custom_encs = zip(*data)
    #############################################
    # articles : tuple of list of sentences. each sentence is a Tensor. ~(bs, n_sents, n_tokens)
    # contexts : tuple of list of turns. each turn is a Tensor. ~(bs, n_turns, n_tokens)
    # candidates : tuple of Tensors. ~(bs, n_tokens)
    # custom_encs : tuple of Tensors. ~(bs, enc)
    # next_states : tuple of list of turns. each turn is a Tensor. ~(bs, n_turns, n_tokens)
    # next_candidates : tuple of list of candidate. each candidate is a Tensor ~(bs, n_actions, n_tokens)
    # next_custom_encs : tuple of list of Tensors. ~(bs, n_actions, enc)
    #############################################

    assert Q_NETWORK_MODE is not None
    if Q_NETWORK_MODE == 'mlp':

        # Merge custom encodings (from tuple of 1D tensor to 2D tensor)
        custom_encs = torch.stack(custom_encs, 0)  # ~ (batch, custom_hs)

        # Compute mask of non-final states
        non_final_mask = map(lambda s: s is not None, next_custom_encs)

        # filter out None custom encs
        non_final_next_custom_encs = filter(lambda s: s is not None, next_custom_encs)
        # ~(bs-, n_actions, enc_size)

        return articles, contexts, candidates, \
               custom_encs, torch.Tensor(rewards), non_final_mask, non_final_next_custom_encs
        #       ~(bs, enc)        ~(bs,)              ~(bs)        ~(bs-, n_actions, enc)

    else:

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

        ######################

        # Compute mask of non-final states
        non_final_mask = map(lambda s: s is not None, next_states)

        # filter out 0 values from n_next_turn, n_next_candidates
        n_non_final_next_turns = filter(lambda t: t!=0, n_next_turns)  # ~(bs-)
        n_non_final_next_candidates = filter(lambda t: t!=0, n_next_candidates)  # ~(bs-)

        # Flatten length of each sentence (from tuple of 1D list to 1D list)
        # we want the number of tokens for each sentence
        l_non_final_next_turns = [length for turns in l_next_turns for length in turns]  # ~(bs- x #turns/context)
        l_non_final_next_candidates = [length for candidates in l_next_candidates for length in candidates]  # ~(bs- x #candidates)

        # None of the examples have a next state!!
        if len(l_non_final_next_candidates) == 0:
            assert len(l_non_final_next_turns) == 0
            logger.info("WARING: none of the examples in this batch have a next state!")
            return articles, articles_tensor, torch.LongTensor(n_sents), torch.LongTensor(l_sents), \
                   contexts, contexts_tensor, torch.LongTensor(n_turns), torch.LongTensor(l_turns), \
                   candidates_tensor, torch.LongTensor(n_tokens), \
                   custom_encs, torch.Tensor(rewards), \
                   non_final_mask, \
                   None, None, None, \
                   None, None, None, \
                   None

        # Merge next state utterances (from tuple of list of 1D Tensor to 2D tensor):
        # from tuple of list of turns =to=> (bs- x n_turns, max_len)
        non_final_next_state_tensor = torch.zeros(len(l_non_final_next_turns),
                                                  max(l_non_final_next_turns)).long()
        i = 0
        for cntxt in next_states:
            if cntxt:
                for turn in cntxt:
                    end = l_non_final_next_turns[i]
                    non_final_next_state_tensor[i, :end] = turn[:end]
                    i += 1

        # Merge next state candidates (from tuple of list of 1D Tensor to 2D tensor):
        # from tuple of list of candidates =to=> (bs- x n_actions, max_len)
        non_final_next_candidates_tensor = torch.zeros(len(l_non_final_next_candidates),
                                                       max(l_non_final_next_candidates)).long()
        i = 0
        for candidates in next_candidates:
            if candidates:
                for action in candidates:
                    end = l_non_final_next_candidates[i]
                    non_final_next_candidates_tensor[i, :end] = action[:end]
                    i += 1

        # filter out None custom encs
        non_final_next_custom_encs = filter(lambda s: s is not None, next_custom_encs)
        # ~(bs-, n_actions, enc_size)

        return articles, articles_tensor, torch.LongTensor(n_sents), torch.LongTensor(l_sents), \
            contexts, contexts_tensor, torch.LongTensor(n_turns), torch.LongTensor(l_turns), \
            candidates_tensor, torch.LongTensor(n_tokens), \
            custom_encs, torch.Tensor(rewards), \
            non_final_mask, \
            non_final_next_state_tensor, torch.LongTensor(n_non_final_next_turns), torch.LongTensor(l_non_final_next_turns), \
            non_final_next_candidates_tensor, torch.LongTensor(n_non_final_next_candidates), torch.LongTensor(l_non_final_next_candidates), \
            non_final_next_custom_encs


def get_loader(json, vocab, q_net_mode, rescale_rewards, batch_size, shuffle, num_workers):
    global Q_NETWORK_MODE
    Q_NETWORK_MODE = q_net_mode

    conv = ConversationDataset(json, vocab, q_net_mode, rescale_rewards)
    data_loader = data.DataLoader(dataset=conv,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader
