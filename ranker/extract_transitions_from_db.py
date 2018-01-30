import json
import pymongo
import argparse
import cPickle as pkl
import time
from collections import defaultdict, Counter
import copy
import sys
import pyprind
import nltk

import features as F
import inspect

ALL_FEATURES = []
for name, obj in inspect.getmembers(F):
    if inspect.isclass(obj) and name not in ['SentimentIntensityAnalyzer', 'Feature']:
        ALL_FEATURES.append(name)

import logging
logger = logging.getLogger(__name__)

logger.info("using ALL features: %s" % ALL_FEATURES)


DB_PORT = 8091
DB_CLIENT = '132.206.3.23'
# NICO_ID = "A30GSVXFJOXNTS"
# KOUSTUV_ID = "A1W0QQF93UM08"


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


def build_data():
    """
    Collect messages from db.dialogs
    Build a mapping from article to dictionary, where each dictionary is made of:
    - context: list of utterances
    - candidate: proposed utterance
    - reward: 0 or 1
    - custom_enc: hand-crafted encoding of (article, context, candidate)
    :return: the mapping
    """
    client = pymongo.MongoClient(DB_CLIENT, DB_PORT)
    db = client.rllchat_mturk

    data = {}  # map from article_string to list of (context_array, candidate_string, reward_int)
    feature_objects, custom_hs = F.initialize_features(ALL_FEATURES)  # list of feature instances

    n_conv = 0
    n_ex = 0

    results = list(db.dialogs.find({'review': 'accepted'}))

    # bar = pyprind.ProgBar(len(results), monitor=True, stream=sys.stdout)
    # loop through each conversation that have 'review':'accepted'
    for conv in results:
        n_conv += 1
        logger.info("conversation %d / %d ..." % (n_conv, len(results)))

        # get the conversation article
        article = conv['chat_state']['context']  # conversation article
        article = article.lower().strip()
        # store the article is not already there
        if article not in data:
            data[article] = []

        context = [ "hello! i hope you're doing well. i am doing fantastic today! " \
                    + "let me go through the article real quick and we will start talking about it." ]

        # loop through each turn for this conversation
        for turn_idx, turn in enumerate(conv['chat_state']['turns']):

            choice_idx = turn['choice']  # choice index
            # skip turn if invalid choice
            if choice_idx == '-1' or choice_idx == -1:
                continue

            # add user response to previous turn to the context
            if turn_idx > 0:
                for option_idx, option in turn['options'].items():
                    if option_idx != '/end' and option['model_name'] == 'fact_gen':
                        user_resp = option['context'][-2].lower().strip()
                        context.append(user_resp)
                        break

            # if chose to finish, this is the last turn so we can break
            if choice_idx == '/end':
                break

            # loop through each candidate option
            for option_idx, option in turn['options'].items():
                # don't consider the /end option
                if option_idx == '/end':
                    continue

                candidate = option['text'].lower().strip()

                if choice_idx == option_idx:
                    r = 1
                else:
                    r = 0

                data[article].append({
                        'context': copy.deepcopy(context),
                        'candidate': candidate,
                        'reward': r,
                        'custom_enc':list(
                            F.get(
                                feature_objects, custom_hs,
                                article, context, candidate
                            )
                        )
                })

                n_ex += 1  # increment example counter
                # bar.update()  # update progress bar

            # after all options, append the chosen text to context
            chosen_text = turn['options'][choice_idx]['text'].lower().strip()
            context.append(chosen_text)

        # end of conversation

    # end of accepted HITs

    return data, n_conv, n_ex


def split(json_data, n_conv, n_ex, valid_prop, test_prop):
    """
    :param json_data: mapping from article to list of dictionary
        where each dictionary is made of 'context', 'candidate', 'reward',
        'custom_enc'
    :param n_conv: total number of conversations accross articles
    :param n_ex: total number of examples accross articles
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

    for article, examples in data.items():
        # take word from the article
        article_tokens = nltk.tokenize.word_tokenize(article)
        counter.update(article_tokens)
        # TODO: take words from the conversations: candidates + user turns

    # if the word frequency is less than 'threshold', then word is discarded
    words = [w for w, cnt in counter.items() if cnt >= threshold]

    # create a vocabulary instance
    vocab = Vocabulary()

    # add special tokens
    vocab.add_word('<pad>')
    vocab.add_word('<unk>')
    # vocab.add_word('<eos>')  # end-of-sentence to split sentences within an article
    # vocab.add_word('<eot>')  # end-of-turn to split turns within a dialog

    # add all other words
    for w in words:
        vocab.add_word(w)

    return vocab


def main():
    parser = argparse.ArgumentParser(description='Create pickle data for training, testing ranker neural net')
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
    logger.info("Get conversations from database and build data...")
    json_data, n_conv, n_ex = build_data()
    logger.info("Got %d unique articles from %d conversations. Total: %d examples" % (
            len(json_data), n_conv, n_ex
    ))

    # print some instances to debug.
    '''
    logger.info("")
    for idx, article in enumerate(json_data):
        if idx == 0 or idx == 20:
            logger.info(article)
            to_print = map(
                    lambda ele: {
                        'ctxt':ele['context'],
                        'cand':ele['candidate'],
                        'r':ele['reward']},
                    json_data[article]
            )
            logger.info(json.dumps(to_print, indent=4, sort_keys=True))
            logger.info('')
            logger.info('')
            logger.info('')
    '''

    # split data into train, valid, test sets
    logger.info("")
    logger.info("Split into train, valid, test sets")
    train, valid, test = split(json_data, n_conv, n_ex, args.valid_proportion, args.test_proportion)
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
    unique_id = str(time.time())
    file_path = "./data/amt_data_db_%s.json" % unique_id
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
    file_path = "./data/amt_vocab_%s.pkl" % unique_id
    with open(file_path, 'wb') as f:
        pkl.dump(vocab, f)
    logger.info("done.")



if __name__ == '__main__':
    main()

