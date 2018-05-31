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
import pyprind
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
LEMM = WordNetLemmatizer()

ALL_FEATURES = []
for name, obj in inspect.getmembers(F):
    if inspect.isclass(obj) and name not in ['SentimentIntensityAnalyzer', 'Feature']:
        ALL_FEATURES.append(name)


logger = logging.getLogger(__name__)

logger.info("using ALL features: %s" % ALL_FEATURES)


DB_PORT = 8091
DB_CLIENT = '132.206.3.23'
# NICO_ID = "A30GSVXFJOXNTS"
# KOUSTUV_ID = "A1W0QQF93UM08"

WELCOME_MSG = "hello! i hope you're doing well. i am doing fantastic today! " \
              "let me go through the article real quick and we will start talking about it."


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


def lemmatize(string):
    tokens = word_tokenize(string)
    result = map(lambda w: LEMM.lemmatize(w), tokens)
    return ' '.join(result)


def _get_user_response(turn, lemm):
    """
    Get the user response right before this turn
    :param turn: turn with a bunch of options
    :param lemm: lemmatize the conversations?
    :return: user message just before
    """
    for option_idx, option in turn['options'].items():
        if option_idx != '/end' and option['model_name'] == 'fact_gen':
            user_resp = option['context'][-2].lower().strip()
            if lemm:
                user_resp = lemmatize(user_resp)
            return user_resp

    raise ValueError("Can't find a user response\nTurn: %s" % json.dumps(turn, indent=2, sort_keys=True))

def build_data(lemm):
    """
    Collect messages from db.dialogs
    Build a mapping from article to list of dictionaries, where each dictionary is made of:
    - state: list of context turns
    - action: proposed candidate turn &
              custom_enc: list of hand-crafted encoding of (article, context, candidate)
    - reward: 0 or 1
    - next_state: list of context turns + candidate turn + user turn,
                  or None if action wasn't taken
    - next_actions: list of possible candidates & hand-crafted encodings after next_state,
                    or None if actio wasn't taken
    - quality: score of the entire conversation int from 1 to 5
    :param lemm: lemmatize the article and conversations?
    :return: the mapping
    """
    client = pymongo.MongoClient(DB_CLIENT, DB_PORT)
    db = client.rllchat_mturk

    data = {}  # map from article_string to list of (context_array, candidate_string, reward_int)
    feature_objects, custom_hs = F.initialize_features(ALL_FEATURES)  # list of feature instances

    n_score = [0, 0]  # number of examples for each vote: 0 or 1
    n_ex = 0
    n_quality = [0, 0, 0, 0, 0]  # number of conversations for each quality score: from 1 to 5

    results = list(db.dialogs.find({'review': 'accepted'}))

    # bar = pyprind.ProgBar(len(results), monitor=True, stream=sys.stdout)
    # loop through each conversation that have 'review':'accepted'
    for conv_idx, conv in enumerate(results):
        logger.info("conversation %d / %d ..." % (conv_idx+1, len(results)))

        # count conversation quality:
        conv_quality = int(conv['chat_state']['metrics']['quality'])
        n_quality[conv_quality-1] += 1

        # get the conversation article
        article = conv['chat_state']['context']  # conversation article
        article = article.lower().strip()
        if lemm:
            article = lemmatize(article)
        # store the article is not already there
        if article not in data:
            data[article] = []

        if lemm:
            context = [ lemmatize(WELCOME_MSG) ]
        else:
            context = [ WELCOME_MSG ]

        # loop through each turn for this conversation
        turn_idx = 0
        first_turn = True
        for turn in conv['chat_state']['turns']:

            choice_idx = turn['choice']  # choice index
            # skip turn if invalid choice
            if choice_idx == '-1' or choice_idx == -1:
                turn_idx += 1
                continue

            # add user response to previous turn to the context
            if not first_turn:
                user_resp = _get_user_response(turn, lemm)
                context.append(user_resp)

            # if chose to finish, this is the last turn so we can break
            if choice_idx == '/end':
                break

            # loop through each candidate option
            for option_idx, option in turn['options'].items():
                # don't consider the /end option
                if option_idx == '/end':
                    continue

                # Store candidate response
                candidate = option['text'].lower().strip()
                # remove wrong formating if there is any
                candidate = re.sub('[0-9]->', '', candidate)
                candidate = candidate.replace('<br />', '')

                if lemm:
                    candidate = lemmatize(candidate)

                # Compute custom encoding
                custom_encoding = list(
                    F.get(
                        feature_objects, custom_hs,
                        article, context, candidate
                    )
                )

                ###
                # Compute next state & next action
                ###
                if choice_idx == option_idx:
                    next_turn_idx = turn_idx+1
                    while next_turn_idx < len(conv['chat_state']['turns']) and conv['chat_state']['turns'][next_turn_idx]['choice'] == -1:
                        next_turn_idx += 1

                    # user took this candidate, we have data for next state & actions
                    if next_turn_idx >= len(conv['chat_state']['turns']):
                        logger.info("This is the last turn (%d)! Next state & action will be None" % next_turn_idx)
                        next_state = None
                        next_actions = None

                    else:
                        next_turn = conv['chat_state']['turns'][next_turn_idx]

                        #  Next state = context + chosen candidate + user turn
                        next_state = copy.deepcopy(context)
                        next_state.append(candidate)
                        next_state.append(
                            _get_user_response(next_turn, lemm)
                        )

                        # Compute possible next actions
                        next_actions = []
                        # loop through each next candidate option
                        for next_option_idx, next_option in next_turn['options'].items():
                            # don't consider the /end option
                            if next_option_idx == '/end':
                                continue

                            # Store candidate response
                            next_candidate = next_option['text'].lower().strip()
                            if lemm:
                                next_candidate = lemmatize(next_candidate)

                            # Compute next custom encoding
                            next_custom_encoding = list(
                                F.get(
                                    feature_objects, custom_hs,
                                    article, next_state, next_candidate
                                )
                            )

                            next_actions.append({
                                'candidate': next_candidate,
                                'custom_enc': next_custom_encoding
                            })

                else:
                    ###
                    # Candidate not taken, no next data for this conversation
                    ###
                    next_state = None
                    next_actions = None

                data[article].append({
                    'chat_id': option['chat_unique_id'],
                    'state': copy.deepcopy(context),
                    'action': {
                        'candidate': candidate,
                        'model_name': option['model_name'],
                        'conf': option['conf'],
                        'score': option['score'],
                        'custom_enc': custom_encoding
                    },
                    'reward': 1 if (choice_idx == option_idx) else 0,
                    'next_state': next_state,
                    'next_actions': next_actions,
                    'quality': conv_quality
                })
                n_ex += 1  # increment example counter

                if choice_idx == option_idx:
                    n_score[1] += 1
                else:
                    n_score[0] += 1

            # after all options, append the chosen text to context & increment counter
            chosen_text = turn['options'][choice_idx]['text'].lower().strip()
            if lemm:
                chosen_text = lemmatize(chosen_text)
            context.append(chosen_text)

            turn_idx += 1
            if first_turn:
                first_turn = False

        # end of conversation

    # end of accepted HITs

    n_conv = len(results)
    assert n_score[0] + n_score[1] == n_ex

    return data, n_conv, n_ex, n_score, n_quality


def split(json_data, n_ex, valid_prop, test_prop, oversample=False):
    """
    :param json_data: mapping from article to list of dictionary
        where each dictionary is made of 'context', 'candidate', 'reward',
        'custom_enc'
    :param n_ex: total number of examples across articles
    :param valid_prop: proportion of data to make the validation set
    :param test_prop: proportion of data to make the test set
    :param oversample: duplicate positive examples as many time as negative examples in the training set
    :return: train, valid, test tuples where each tuple is made of:
        (json_data, n_examples)
    """
    # maximum nuber of items in each category
    min_n_valid = int(n_ex * valid_prop)
    min_n_test = int(n_ex * test_prop)

    train_data = {}
    valid_data = {}
    test_data = {}
    n_train, n_pos, n_neg = 0, 0, 0
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
            # go through each examples individually and count the positive & negative ones
            for ex in examples:
                train_data[article].append(ex)  # add each example at least once
                if ex['reward'] == 1:
                    n_pos += 1  # count++
                    n_train += 1
                    # if oversample, copy the positive example as many times as necessary
                    while oversample and n_pos < n_neg:
                        train_data[article].append(ex)
                        n_pos += 1
                        n_train += 1
                else:
                    n_neg += 1
                    n_train += 1

    return (train_data, n_train, n_pos, n_neg), (valid_data, n_valid), (test_data, n_test)


def build_vocab(data, threshold):
    counter = Counter()  # count number of words

    # take words from welcome message
    welcome_msg_tokens = word_tokenize(WELCOME_MSG)
    counter.update(welcome_msg_tokens)

    for article, examples in data.items():
        # take words from the article
        article_tokens = word_tokenize(article)
        counter.update(article_tokens)

        conversation_ids = []
        for ex in examples:
            # take words from each candidate response
            candidate_tokens = word_tokenize(ex['action']['candidate'])
            counter.update(candidate_tokens)

            # check that this conversation has not been seen before
            if ex['chat_id'] not in conversation_ids:
                conversation_ids.append(ex['chat_id'])
                # take words from the the user messages
                for sentence in ex['state'][2::2]:
                    sent_tokens = word_tokenize(sentence)
                    counter.update(sent_tokens)

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
    parser.add_argument('-l', '--lemmatize', action='store_true',
                        help="lemmatize article and conversations")
    parser.add_argument('-vp', '--valid_proportion', type=float, default=0.1,
                        help="proportion of data to make validation set")
    parser.add_argument('-tp', '--test_proportion', type=float, default=0.1,
                        help="proportion of data to make testing set")
    parser.add_argument('-vt', '--vocab_threshold', type=int, default=1,
                        help="minimum number of times a word must occur to be in vocabulary")
    parser.add_argument('-os', '--oversample', action='store_true',
                        help="duplicate positive examples to have balanced training set")
    args = parser.parse_args()
    logger.info("")
    logger.info(args)

    logger.info("")
    logger.info("Get conversations from database and build data...")
    json_data, n_conv, n_ex, n_score, n_quality = build_data(args.lemmatize)
    logger.info("Got %d unique articles from %d conversations. Total: %d examples (%d +1s & %d -1s)" % (
            len(json_data), n_conv, n_ex, n_score[1], n_score[0]
    ))
    logger.info(" - 'very bad'  conversations: %d / %d = %f" % (n_quality[0], n_conv, n_quality[0] / float(n_conv)))
    logger.info(" - 'bad'       conversations: %d / %d = %f" % (n_quality[1], n_conv, n_quality[1] / float(n_conv)))
    logger.info(" - 'medium'    conversations: %d / %d = %f" % (n_quality[2], n_conv, n_quality[2] / float(n_conv)))
    logger.info(" - 'good'      conversations: %d / %d = %f" % (n_quality[3], n_conv, n_quality[3] / float(n_conv)))
    logger.info(" - 'very good' conversations: %d / %d = %f" % (n_quality[4], n_conv, n_quality[4] / float(n_conv)))

    # print some instances to debug.
    logger.info("")
    for idx, article in enumerate(json_data):
        if idx > 437 and idx < 440:
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
    logger.info("Split into train, valid, test sets...")
    train, valid, test = split(json_data, n_ex, args.valid_proportion, args.test_proportion, args.oversample)
    logger.info("[train] %d unique articles. Total: %d examples (%d positive -vs- %d negative)" % (
            len(train[0]), train[1], train[2], train[3]
    ))
    logger.info("[valid] %d unique articles. Total: %d examples" % (
            len(valid[0]), valid[1]
    ))
    logger.info("[test] %d unique articles. Total: %d examples" % (
            len(test[0]), test[1]
    ))

    logger.info("")
    logger.info("Saving to json & pkl file...")
    unique_id = str(time.time())

    if args.oversample:
        file_path = "./data/q_ranker_amt_data++_%s.json" % unique_id
    else:
        file_path = "./data/q_ranker_amt_data_%s.json" % unique_id
    with open(file_path, 'wb') as f:
        json.dump(
                {
                    'train': [train[0], train[1]],
                    'valid': [valid[0], valid[1]],
                    'test': [test[0], test[1]]
                },
                f
        )
    with open(file_path.replace('.json', '.pkl'), 'wb') as f:
         pkl.dump(
                {
                    'train': [train[0], train[1]],
                    'valid': [valid[0], valid[1]],
                    'test': [test[0], test[1]]
                },
                f,
                protocol=pkl.HIGHEST_PROTOCOL
        )

    logger.info("done.")

    # Build vocab on training set
    logger.info("")
    logger.info("Building vocab on training set...")
    vocab = build_vocab(train[0], args.vocab_threshold)
    logger.info("Total vocabulary size: %d" % len(vocab))

    logger.info("Saving vocab to pkl file...")
    file_path = "./data/q_ranker_amt_vocab_%s.pkl" % unique_id
    with open(file_path, 'wb') as f:
        pkl.dump(vocab, f, protocol=pkl.HIGHEST_PROTOCOL)
    logger.info("done.")


if __name__ == '__main__':
    main()
