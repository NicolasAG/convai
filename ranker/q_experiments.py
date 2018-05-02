
def default_params(mode, r):
    """
    Return default dictionary of parameters
    :param mode: 'mlp' or 'rnn+mlp' or 'rnn+rnn+mlp'
    :param r: classify immediate reward (true) or estimate q values (false)
    :return: dictionary of parameters
    """
    return {
        'data_f': "./data/q_ranker_amt_data_1524939554.0.pkl",
        'vocab_f': "./data/q_ranker_amt_vocab_1524939554.0.pkl",
        'mode': mode,
        'predict_rewards': r,
        'gpu': 0,
        'verbose': False,
        'debug': False,
        # training parameters:
        'optimizer': 'adam',  # ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta']
        'learning_rate': 0.001,
        'fix_embeddings': False,
        'patience': 20,
        'epochs': 10000,  # max epochs
        'batch_size': 128,
        # Q-parameters:
        'gamma': 0.99,  # discount factor
        'update_frequence': 20000,  # update target DQN
        # RNN params:
        'rnn_gate': 'gru',  # ['rnn', 'gru', 'lstm']
        'sentence_hs': 300,
        'sentence_bidir': False,
        'sentence_dropout': 0.2,

        'article_hs': 300,
        'article_bidir': False,
        'article_dropout': 0.2,

        'utterance_hs': 300,
        'utterance_bidir': False,
        'utterance_dropout': 0.2,

        'context_hs': 300,
        'context_bidir': False,
        'context_dropout': 0.2,
        # MLP params:
        'mlp_activation': 'prelu',  # ['sigmoid', 'relu', 'prelu']
        'mlp_dropout': 0.2
    }

def mlp_r_exp0():
    params = default_params('mlp', True)

    return params

# TODO: define a bunch of experiments!
