import features as F

import inspect
import logging
import time
import json
import pymongo
import argparse
import cPickle as pkl
import copy
import sys
import re
import random
import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter

ALL_FEATURES = []
for name, obj in inspect.getmembers(F):
    if inspect.isclass(obj) and name not in ['SentimentIntensityAnalyzer', 'Feature']:
        ALL_FEATURES.append(name)

ALL_COLORS = [
    'red', 'green', 'blue', 'yellow', 'orange',
    'magenta', 'cyan', 'pink', 'black', 'white',
    'purple', 'grey', 'brown', 'silver', 'gold'
]
WELCOME_MSG = "hello! i hope you're doing well. i am doing fantastic today! " \
              "let me go through the article real quick and we will start talking about it."

N_TURNS = 10  # number of turns per conversation


logger = logging.getLogger(__name__)

logger.info("using ALL features: %s" % ALL_FEATURES)




class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_data(total):
    """
    Create a toy domain dataset with small colorful dataset
    Build a mapping from colorful article to list of dictionaries, where each dictionary is made of:
    - state: list of colorful context turns
    - action: proposed colorful candidate turn &
              custom_enc: list of hand-crafted encoding of (article, context, candidate)
    - reward: 0 or 1
    - next_state: list of colorful context turns + candidate turn + user turn,
                  or None if action wasn't taken
    - next_actions: list of possible coloful candidates & hand-crafted encodings after next_state,
                    or None if action wasn't taken
    - quality: score of the entire coloful conversation int from 1 to 5
    :param total: number of examples we want in total
    :return: the mapping
    """

    data = {}  # map from article_string to list of (context_array, candidate_string, reward_int)
    feature_objects, custom_hs = F.initialize_features(ALL_FEATURES)  # list of feature instances

    n_conv = 0
    n_ex = 0
    n_quality = [0, 0, 0, 0, 0]  # number of conversations for each quality score: from 1 to 5

    # while we don't have enough examples
    while n_ex < total:
        n_conv += 1
        logger.info("conversation %d ..." % n_conv)

        # Sample random colors for this conversation
        this_colors = random.sample(ALL_COLORS, 4)  # alternate 4 colors for the article
        sub_colors = random.sample(this_colors, 2)  # alternate 2 colors for the conversation
        wrong_colors = [c for c in this_colors if c not in sub_colors]

        # define colorful article
        article_length = random.randint(3, 10)  # random length between 3*4 and 10*4
        article = ' '.join(this_colors*article_length)
        if article not in data:
            data[article] = []

        # random conversation quality:
        conv_quality = np.random.choice(range(5), p=[0.1, 0.2, 0.4, 0.2, 0.1])
        n_quality[conv_quality - 1] += 1

        context = [WELCOME_MSG]

        resp_length = random.randint(1, 10)  # random number of words in the response
        wrong_lengths = [11 - resp_length, (resp_length / 2) + 1]
        for turn_idx in range(N_TURNS):

            if turn_idx > 0:
                # user response
                user_resp = ' '.join([sub_colors[0]]*resp_length)
                context.append(user_resp)

            candidates = [
                # the correct response:
                ' '.join([sub_colors[1]] * resp_length),
                # the wrong responses:
                ' '.join([sub_colors[1]] * wrong_lengths[0]),
                ' '.join([sub_colors[1]] * wrong_lengths[1]),
                ' '.join([sub_colors[0]] * resp_length),
                ' '.join([wrong_colors[0]] * resp_length),
                ' '.join([wrong_colors[1]] * resp_length),
                ' '.join([wrong_colors[0], wrong_colors[1]] * (resp_length / 2 + 1))
            ]

            next_resp_length = random.randint(1, 10)  # random number of words in the response
            next_wrong_lengths = [11 - next_resp_length, (next_resp_length / 2) + 1]

            # loop through each candidate
            for cand_idx, candidate in enumerate(candidates):

                # Compute custom encoding
                custom_encoding = list(
                    F.get(
                        feature_objects, custom_hs,
                        article, context, candidate
                    )
                )

                # chosen candidate
                if cand_idx == 0:
                    # no next turn
                    if turn_idx + 1 >= N_TURNS:
                        next_state = None
                        next_actions = None
                    # create custom next turn
                    else:
                        #  Next state = context + chosen candidate + user turn
                        next_state = copy.deepcopy(context)
                        next_state.append(candidate)
                        next_state.append(
                            ' '.join([sub_colors[0]] * next_resp_length)
                        )
                        # next candidates:
                        next_candidates = [
                            # the correct response:
                            ' '.join([sub_colors[1]] * next_resp_length),
                            # the wrong responses:
                            ' '.join([sub_colors[1]] * next_wrong_lengths[0]),
                            ' '.join([sub_colors[1]] * next_wrong_lengths[1]),
                            ' '.join([sub_colors[0]] * next_resp_length),
                            ' '.join([wrong_colors[0]] * next_resp_length),
                            ' '.join([wrong_colors[1]] * next_resp_length),
                            ' '.join([wrong_colors[0], wrong_colors[1]] * (next_resp_length / 2 + 1))
                        ]
                        # Compute possible next actions
                        next_actions = []
                        for next_candidate in next_candidates:
                            # Compute next custom encoding
                            next_custom_encoding = list(
                                F.get(
                                    feature_objects, custom_hs,
                                    article, next_state, next_candidate
                                )
                            )
                            # append next candidate
                            next_actions.append({
                                'candidate': next_candidate,
                                'custom_enc': next_custom_encoding
                            })

                # wrong candidate
                else:
                    next_state = None
                    next_actions = None

                data[article].append({
                    'state': copy.deepcopy(context),
                    'action': {
                        'candidate': candidate,
                        'custom_enc': custom_encoding
                    },
                    'reward': 1 if (cand_idx == 0) else 0,
                    'next_state': next_state,
                    'next_actions': next_actions,
                    'quality': conv_quality
                })
                n_ex += 1  # increment example counter

            # after all options, append the chosen text to context
            context.append(candidates[0])
            # and set the next turn lengths
            resp_length = next_resp_length
            wrong_lengths = next_wrong_lengths

        # end of conversation

    # end of accepted HITs

    return data, n_conv, n_ex, n_quality


def split(json_data, n_ex, valid_prop, test_prop):
    """
    :param json_data: mapping from article to list of dictionary
        where each dictionary is made of 'context', 'candidate', 'reward',
        'custom_enc'
    :param n_ex: total number of examples across articles
    :param valid_prop: proportion of data to make the validation set
    :param test_prop: proportion of data to make the test set
    :return: train, valid, test tuples where each tuple is made of:
        (json_data, n_examples)
    """
    # maximum nuber of items in each category
    min_n_valid = int(n_ex * valid_prop)
    min_n_test = int(n_ex * test_prop)

    train_data = {}
    valid_data = {}
    test_data = {}
    n_train = 0
    n_valid = 0
    n_test = 0

    # we add examples one batch at a time. each examples in an article will be
    #  added to the same set to avoid train/valid/test overlap
    for article, examples in json_data.items():
        # first add all examples of the same article to test set
        if n_test < min_n_test:
            if article not in test_data:
                test_data[article] = []
            test_data[article].extend(examples)
            n_test += len(examples)
        # now add all examples of the same article to valid set
        elif n_valid < min_n_valid:
            if article not in valid_data:
                valid_data[article] = []
            valid_data[article].extend(examples)
            n_valid += len(examples)
        # add the remaining examples to the training set
        else:
            if article not in train_data:
                train_data[article] = []
            train_data[article].extend(examples)
            n_train += len(examples)

    return (train_data, n_train), (valid_data, n_valid), (test_data, n_test)


def build_vocab(data, threshold):
    counter = Counter()  # count number of words

    # take words from welcome message
    welcome_msg_tokens = word_tokenize(WELCOME_MSG)
    counter.update(welcome_msg_tokens)
    # take words from the colorful vocab
    counter.update(ALL_COLORS)

    # if the word frequency is less than 'threshold', then word is discarded
    words = [w for w, cnt in counter.items() if cnt >= threshold]

    # create a vocabulary instance
    vocab = Vocabulary()

    # add special tokens
    vocab.add_word('<pad>')
    vocab.add_word('<unk>')

    vocab.add_word('<sos>')  # start-of-sentence
    vocab.add_word('<eos>')  # end-of-sentence to split sentences within an article

    vocab.add_word('<sot>')  # start-of-turn
    vocab.add_word('<eot>')  # end-of-turn to split turns within a dialog

    # add all other words
    for w in words:
        vocab.add_word(w)

    return vocab


def main():
    parser = argparse.ArgumentParser(description='Create pickle data for training, testing ranker neural net')
    parser.add_argument('-nx', '--number_examples', type=int, default=1000,
                        help="number of examples in total")
    parser.add_argument('-vp', '--valid_proportion', type=float, default=0.1,
                        help="proportion of data to make validation set")
    parser.add_argument('-tp', '--test_proportion', type=float, default=0.1,
                        help="proportion of data to make testing set")
    parser.add_argument('-vt', '--vocab_threshold', type=int, default=1,
                        help="minimum number of times a word must occur to be in vocabulary")
    args = parser.parse_args()
    logger.info("")
    logger.info(args)

    logger.info("")
    logger.info("Build conversations from scratch and build data...")
    json_data, n_conv, n_ex, n_quality = build_data(args.number_examples)
    logger.info("Got %d unique articles from %d conversations. Total: %d examples" % (
            len(json_data), n_conv, n_ex
    ))
    logger.info(" - 'very bad'  conversations: %d / %d = %f" % (n_quality[0], n_conv, n_quality[0] / float(n_conv)))
    logger.info(" - 'bad'       conversations: %d / %d = %f" % (n_quality[1], n_conv, n_quality[1] / float(n_conv)))
    logger.info(" - 'medium'    conversations: %d / %d = %f" % (n_quality[2], n_conv, n_quality[2] / float(n_conv)))
    logger.info(" - 'good'      conversations: %d / %d = %f" % (n_quality[3], n_conv, n_quality[3] / float(n_conv)))
    logger.info(" - 'very good' conversations: %d / %d = %f" % (n_quality[4], n_conv, n_quality[4] / float(n_conv)))

    # print some instances to debug.
    logger.info("")
    for idx, article in enumerate(json_data):
        if idx == 10:
            logger.info(article)
            to_print = map(
                    lambda ele: {
                        'ctxt': ele['state'],
                        'cand': ele['action']['candidate'],
                        'r': ele['reward'],
                        'next_state': ele['next_state'],
                        'next_actions': [i['candidate'] for i in ele['next_actions']] if ele['next_actions'] else 'None'
                    },
                    json_data[article]
            )
            logger.info(json.dumps(to_print, indent=4, sort_keys=True))
            logger.info('')
            logger.info('')
            logger.info('')

    # split data into train, valid, test sets
    logger.info("")
    logger.info("Split into train, valid, test sets")
    train, valid, test = split(json_data, n_ex, args.valid_proportion, args.test_proportion)
    logger.info("[train] %d unique articles. Total: %d examples" % (
            len(train[0]), train[1]
    ))
    logger.info("[valid] %d unique articles. Total: %d examples" % (
            len(valid[0]), valid[1]
    ))
    logger.info("[test] %d unique articles. Total: %d examples" % (
            len(test[0]), test[1]
    ))

    logger.info("")
    logger.info("Saving to json file...")
    file_path = "./data/q_ranker_colorful_data.json"
    with open(file_path, 'wb') as f:
        json.dump(
                {
                    'train': [train[0], train[1]],
                    'valid': [valid[0], valid[1]],
                    'test': [test[0], test[1]]
                },
                f
        )
    logger.info("done.")

    # Build vocab on training set
    logger.info("")
    logger.info("Building vocab on training set...")
    vocab = build_vocab(train[0], args.vocab_threshold)
    logger.info("Total vocabulary size: %d" % len(vocab))

    logger.info("Saving vocab to pkl file...")
    file_path = "./data/q_ranker_colorful_vocab.pkl"
    with open(file_path, 'wb') as f:
        pkl.dump(vocab, f)
    logger.info("done.")


if __name__ == '__main__':
    main()
