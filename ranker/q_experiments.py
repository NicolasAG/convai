
def to_dict(args):
    return {
        'data_f': args.data_f,
        'vocab_f': args.vocab_f,
        'mode': args.mode,
        'predict_rewards': args.predict_rewards,
        'model_name': args.model_name,
        'experiment': None,
        'gpu': args.gpu,
        'verbose': args.verbose,
        'debug': args.debug,
        # training parameters:
        'optimizer': args.optimizer,  # ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta']
        'learning_rate': args.learning_rate,
        'fix_embeddings': args.fix_embeddings,
        'patience': args.patience,
        'epochs': args.epochs,  # max epochs
        'batch_size': args.batch_size,
        # Q-parameters:
        'gamma': args.gamma,  # discount factor
        'update_frequence': args.update_frequence,  # update target DQN
        # RNN params:
        'rnn_gate': args.rnn_gate,  # ['rnn', 'gru', 'lstm']
        'use_custom_encs': args.use_custom_encs,

        'sentence_hs': args.sentence_hs,
        'sentence_bidir': args.sentence_bidir,
        'sentence_dropout': args.sentence_dropout,

        'article_hs': args.article_hs,
        'article_bidir': args.article_bidir,
        'article_dropout': args.article_dropout,

        'utterance_hs': args.utterance_hs,
        'utterance_bidir': args.utterance_bidir,
        'utterance_dropout': args.utterance_dropout,

        'context_hs': args.context_hs,
        'context_bidir': args.context_bidir,
        'context_dropout': args.context_dropout,
        # MLP params:
        'mlp_activation': args.mlp_activation,  # ['sigmoid', 'relu', 'prelu']
        'mlp_dropout': args.mlp_dropout
    }


def default_params(mode, r):
    """
    Return default dictionary of parameters
    :param mode: 'mlp' or 'rnn+mlp' or 'rnn+rnn+mlp'
    :param r: classify immediate reward (true) or estimate q values (false)
    :return: dictionary of parameters
    """
    return {
        # 'data_f': "./data/q_ranker_amt_data_1524939554.0.pkl",
        # 'vocab_f': "./data/q_ranker_amt_vocab_1524939554.0.pkl",
        'data_f': "./data/q_ranker_amt_data++_1525301962.86.json",  # oversampled positive examples
        'vocab_f': "./data/q_ranker_amt_vocab_1525301962.86.pkl",
        'mode': mode,
        'predict_rewards': r,
        'model_name': None,
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
        'update_frequence': 2000,  # update target DQN
        # RNN params:
        'rnn_gate': 'gru',  # ['rnn', 'gru', 'lstm']
        'use_custom_encs': True,

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


#######################
# CLASSIFY R WITH MLP #
#######################

def mlp_r_exp0():
    params = default_params('mlp', True)
    params['model_name'] = 'SmallR/Small_R-Network[exp00os]'
    return params


#######################
# CLASSIFY R WITH RNN #
#######################

def rnnmlp_r_exp0():
    params = default_params('rnn+mlp', True)
    params['model_name'] = 'DeepR/Deep_R-Network[exp00os]'

    params['fix_embeddings'] = True
    params['sentence_hs'] = 100
    params['utterance_hs'] = 100

    return params


######################
# PREDICT Q WITH MLP #
######################

def mlp_q_exp0():
    params = default_params('mlp', False)
    params['model_name'] = 'SmallQ/Small_Q-Network[exp00os]'
    return params


######################
# PREDICT Q WITH RNN #
######################

def rnnmlp_q_exp0():
    params = default_params('rnn+mlp', False)
    params['model_name'] = 'DeepQ/Deep_Q-Network[exp00os]'
    return params
