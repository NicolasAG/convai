"""
This script previously extracted test amt data and label the chosen message
in each conversation interaction according to some specific short term
ranker, previously trained for the ConvAI competition.
"""
import argparse
import logging
import time
import tensorflow as tf
import numpy as np
import cPickle as pkl

import features
from estimators import Estimator, ACTIVATIONS, OPTIMIZERS, SHORT_TERM_MODE


logger = logging.getLogger(__name__)


logging.info("creating ranker feature instances...")
start_creation_time = time.time()
feature_objects, feature_dim = features.initialize_features(feature_list_short)
logging.info("created all feature instances in %s sec" %
             (time.time() - start_creation_time))

def get_test_data():
    """
    Load raw data. Takes a while!
    """

    # oversampled --> useless for test
    #data_f = "./data/q_ranker_amt_data++_1525301962.86.pkl"
    # regular
    data_f = "./data/q_ranker_amt_data_1524939554.0.pkl"

    logger.info("")
    logger.info("Loading data from %s..." % data_f)
    with open(data_f.replace('.json', '.pkl'), 'rb') as f:
        raw_data = pkl.load(f)

    test_data = raw_data['test'][0]
    test_size = raw_data['test'][1]
    logger.info("got %d test examples" % test_size)

    return test_data


# --> PROBABLY D'ONT NEED IT!! :D
def load_previous_model():
    logger.info("")
    logger.info("Loading model %s ..." % args.model_prefix)
    # Load previously saved model arguments
    with open("%sargs.pkl" % args.model_prefix, 'rb') as handle:
        data, \
        hidden_dims, hidden_dims_extra, activation, \
        optimizer, learning_rate, \
        model_path, model_id, model_name, \
        batch_size, dropout_rate, pretrained = pkl.load(handle)

    # reconstruct model_path just in case it has been moved:
    model_path = args.model_prefix.split(model_id)[0]
    if model_path.endswith('/'):
        model_path = model_path[:-1]  # ignore the last '/'

    return data, \
        hidden_dims, hidden_dims_extra, activation, \
        optimizer, learning_rate, \
        model_path, model_id, model_name, \
        batch_size, dropout_rate, pretrained


# --> PROBABLY D'ONT NEED IT!! :D
def build_graph(data, hidden_dims, hidden_dims_extra, activation, optimizer, learning_rate, model_path, model_id, model_name):
    logger.info("")
    logger.info("Building network...")

    model_graph = tf.Graph()
    with model_graph.as_default():
        estimator = Estimator(
            data,
            hidden_dims, hidden_dims_extra, activation,
            optimizer, learning_rate,
            model_path, model_id, model_name
        )
    return model_graph, estimator


# --> PROBABLY D'ONT NEED IT!! :D
def get_ranker_prediction(feature_objects, feature_dim, article, context, candidate, sess, graph, estimator):
    # calculate NN estimation
    logging.info("Start feature calculation")
    raw_features = features.get(
        feature_objects,
        feature_dim,
        article,
        context,
        candidate
    )
    logging.info("Done feature calculation")

    # reshape raw_features to fit the ranker format
    assert len(raw_features) == feature_dim
    candidate_vector = raw_features.reshape(1, feature_dim)  # make an array of shape (1, input)

    # Get predictions for this candidate response:
    logging.info("Scoring the candidate response")
    with sess.as_default():
        with graph.as_default():
            # get predicted class (0: downvote, 1: upvote), and confidence (ie: proba of upvote)
            vote, conf = estimator.predict(SHORT_TERM_MODE, candidate_vector)
            # sanity check with batch size of 1
            assert len(vote) == len(conf) == 1

    conf = conf[0]  # 0.0 < Pr(upvote) < 1.0
    return conf


def main():
    """
    Build a map from chat ID to list of (context, candidates, rewards, predictions) list
    :return: a map of this form:
    chat_id : [
        {
            'context'    : [list of strings],
            'candidates' : [list of strings],
            'rewards'    : [list of ints],
            'predictions': [list of floats]
        },
        {
            ...
        },
        ...
    ],
    chat_id : [
        ...
    ],
    """

    # Load AMT test data
    raw_data = get_test_data()

    # Load competition short term ranker --> PROBABLY D'ONT NEED IT!! :D
    model_data,\
        hidden_dims, hidden_dims_extra,\
        activation, optimizer, learning_rate,\
        model_path, model_id, model_name,\
        batch_size, dropout_rate, pretrained = load_previous_model()

    # Build features required for this ranker --> PROBABLY D'ONT NEED IT!! :D
    feature_list = model_data[-1]
    logger.info("")
    logging.info("creating ranker feature instances...")
    feature_objects, feature_dim = features.initialize_features(feature_list)

    # Build short term ranker network --> PROBABLY D'ONT NEED IT!! :D
    graph, estimator = build_graph(model_data, hidden_dims, hidden_dims_extra, activation,
                               optimizer, learning_rate, model_path, model_id, model_name)

    # Create a TF session --> PROBABLY D'ONT NEED IT!! :D
    sess = tf.Session(graph=graph)

    # Reset short term ranker parameters --> PROBABLY D'ONT NEED IT!! :D
    with sess.as_default():
        with graph.as_default():
            logger.info("Reset short term network parameters...")
            estimator.load(sess, model_path, model_id, model_name)


    chats = {}

    for article, entries in raw_data.iteritems():
        for entry in entries:
            '''
            'chat_id': <string>,
            'state': <list of strings> ie: context,
            'action': {
                'candidate': <string>  ie: candidate,
                'custom_enc': <list of float>,
                'model_name': <string of model name>,
                'score': <string of float between 0.0 & 1.0 or int 0 when not evaluated>,
            },
            'reward': <int {0,1}>,
            'next_state': <list of strings || None> ie: next_context,
            'next_actions': <list of actions || None> ie: next possible actions
            'quality': <int {1,2,3,4,5}>,
            '''
            idx = -1
            # if chat already exists,
            if entry['chat_id'] in chats:
                for i, c in enumerate(chats[entry['chat_id']]):
                    # if context already exists, add this candidate response
                    if c['context'] == entry['state']:
                        c['candidates'].append(entry['action']['candidate'])
                        c['rewards'].append(entry['reward'])
                        idx = i
                        break
                # if context doesn't exists, add it as a new one with this candidate response
                if idx == -1:
                    chats[entry['chat_id']].append({
                        'context': entry['state'],
                        'candidates': [entry['action']['candidate']],
                        'rewards': [entry['reward']]
                    })
            # if chat doesn't exists, add a new one
            else:
                chats[entry['chat_id']] = [{
                    'context': entry['state'],
                    'candidates': [entry['action']['candidate']],
                    'rewards': [entry['reward']]
                }]
                idx = 0  # it's the first and last chat so idx=0 or idx=-1 are both ok

            ############################
            # predict chosen candidate #
            ############################

            # step1: if context length = 1, chose randomly between NQG and EntitySentence

            # step2: TODO: continue...

            try:
                chats[entry['chat_id']][idx]['predictions'].append(predictions.data[1])
            except KeyError:
                chats[entry['chat_id']][idx]['predictions'] = [predictions.data[1]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_prefix", help="path to the short term ranker used in competition")

    args = parser.parse_args()

    main()

