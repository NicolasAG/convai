import logging
import time
import json
import pymongo
import argparse
import cPickle as pkl
import copy
import sys
import pyprind
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

DB_PORT = 8091
DB_CLIENT = '132.206.3.23'
# NICO_ID = "A30GSVXFJOXNTS"
# KOUSTUV_ID = "A1W0QQF93UM08"

WELCOME_MSG = "hello! i hope you're doing well. i am doing fantastic today! " \
              "let me go through the article real quick and we will start talking about it."


def _get_user_response(turn):
    """
    Get the user response right before this turn
    :param turn: turn with a bunch of options
    :param lemm: lemmatize the conversations?
    :return: user message just before
    """
    for option_idx, option in turn['options'].items():
        if option_idx != '/end' and option['model_name'] == 'fact_gen':
            user_resp = option['context'][-2].lower().strip()
            return user_resp

    raise ValueError("Can't find a user response\nTurn: %s" % json.dumps(turn, indent=2, sort_keys=True))


def build_data():
    """
    Create a list of dictionaries of the form {'article':<str>, 'context':<list of str>, 'candidate':<str>, 'r':<-1,0,1>, 'R':<0-5>}
    :return: list of training instances. each instance is a dictionary
    """

    client = pymongo.MongoClient(DB_CLIENT, DB_PORT)
    db = client.rllchat_mturk

    data = []  # list of training instances

    n_conv = 0
    n_ex = 0
    n_quality = [0, 0, 0, 0, 0]  # number of conversations for each quality score: from 1 to 5

    results = list(db.dialogs.find({'review': 'accepted'}))

    # bar = pyprind.ProgBar(len(results), monitor=True, stream=sys.stdout)
    # loop through each conversation that have 'review':'accepted'
    for conv in results:
        n_conv += 1
        logger.info("conversation %d / %d ..." % (n_conv, len(results)))

        # count conversation quality:
        conv_quality = int(conv['chat_state']['metrics']['quality'])
        n_quality[conv_quality-1] += 1

        # get the conversation article
        article = conv['chat_state']['context']  # conversation article
        article = article.lower().strip()

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
                user_resp = _get_user_response(turn)
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

                data.append({
                    'article': article,
                    'context': context,
                    'candidate': candidate,
                    'r': 1 if choice_idx == option_idx else -1,
                    'R': conv_quality,
                    'policy': 'AMT_DATA',
                    'model': option['model_name']
                })
                n_ex += 1  # increment example counter

            # after all options, append the chosen text to context & increment counter
            chosen_text = turn['options'][choice_idx]['text'].lower().strip()
            context.append(chosen_text)
            turn_idx += 1
            if first_turn:
                first_turn = False

        # end of conversation

    # end of accepted HITs

    return data, n_conv, n_ex, n_quality


def main():

    logger.info("Get conversations from database and build data...")
    json_data, n_conv, n_ex, n_quality = build_data()
    logger.info("Got %d unique articles from %d conversations. Total: %d examples" % (
            len(json_data), n_conv, n_ex
    ))
    logger.info(" - 'very bad'  conversations: %d / %d = %f" % (n_quality[0], n_conv, n_quality[0] / float(n_conv)))
    logger.info(" - 'bad'       conversations: %d / %d = %f" % (n_quality[1], n_conv, n_quality[1] / float(n_conv)))
    logger.info(" - 'medium'    conversations: %d / %d = %f" % (n_quality[2], n_conv, n_quality[2] / float(n_conv)))
    logger.info(" - 'good'      conversations: %d / %d = %f" % (n_quality[3], n_conv, n_quality[3] / float(n_conv)))
    logger.info(" - 'very good' conversations: %d / %d = %f" % (n_quality[4], n_conv, n_quality[4] / float(n_conv)))

    '''
    array of dictionaries:
    {
        'article': article,
        'context': c,
        'candidate': m['text'].strip().lower(),
        'r': score_map[int(m['evaluation'])],
        'R': full_eval,
        'policy': m['policy'],
        'model': m['model']
    }
    '''


    # print some instances to debug.
    logger.info("")
    for idx, instance in enumerate(json_data):
        if idx > 429 and idx < 440:
            to_print = {
                'article': instance['article'],
                'ctxt': instance['context'],
                'cand': instance['candidate'],
                'r': instance['r'],
                'R': instance['R']
            }
            logger.info(json.dumps(to_print, indent=2, sort_keys=True))
            logger.info("")
            logger.info("")
            logger.info("")


    logger.info("\nSaving to json file...")
    unique_id = str(time.time())
    file_path = "./data/supervised_ranker_amt_data_%s.json" % unique_id
    with open(file_path, 'wb') as f:
        json.dump(json_data, f)
    logger.info("done.")


if __name__ == '__main__':
    main()
