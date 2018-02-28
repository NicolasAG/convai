import torch
from torch.autograd import Variable
from q_networks import to_var, QNetwork, DeepQNetwork
from q_data_loader import get_loader
from extract_transitions_from_db import Vocabulary
from embedding_metrics import w2v

import numpy as np
import cPickle as pkl
import argparse
import logging
import copy
import json
import time
import sys
import os

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s %(name)s.%(funcName)s l%(lineno)s [%(levelname)s]: %(message)s"
# )

logger = logging.getLogger(__name__)


def get_data(data_f, vocab_f):
    """
    Load data to train Q Network.
    :param data_f: path to the json data
    :param vocab_f: path to the pkl Vocabulary
    :return: vocabulary, training examples, validation examples,
                testing examples, embeddings
    """
    logger.info("")
    logger.info("Loading data...")
    with open(data_f, 'rb') as f:
        raw_data = json.load(f)
    train_data = raw_data['train'][0]
    train_size = raw_data['train'][1]
    valid_data = raw_data['valid'][0]
    valid_size = raw_data['valid'][1]
    test_data = raw_data['test'][0]
    test_size = raw_data['test'][1]

    logger.info("got %d train examples" % train_size)
    logger.info("got %d valid examples" % valid_size)
    logger.info("got %d test examples" % test_size)

    logger.info("")
    logger.info("Loading vocabulary...")
    with open(vocab_f, 'rb') as f:
        vocab = pkl.load(f)
    logger.info("number of unique tokens: %d" % len(vocab))

    logger.info("")
    logger.info("Get data loaders...")
    train_loader = get_loader(
        json=train_data, vocab=vocab, q_net_mode=args.mode,
        batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    valid_loader = get_loader(
        json=valid_data, vocab=vocab, q_net_mode=args.mode,
        batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    test_loader = get_loader(
        json=test_data, vocab=vocab, q_net_mode=args.mode,
        batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    logger.info("done.")

    logger.info("")
    logger.info("Building word embeddings...")
    embeddings = np.random.uniform(-0.1, 0.1, size=(len(vocab), w2v.vector_size))
    pretrained = 0
    for word, idx in vocab.word2idx.items():
        if word in w2v:
            embeddings[idx] = w2v[word]
            pretrained += 1
    logger.info("Got %d/%d = %.6f pretrained embeddings" % (
            pretrained, len(vocab), float(pretrained)/len(vocab)
    ))

    # take arbitrarily the first training example to find the size of the custom encoding
    first_article = train_data.keys()[0]
    custom_hs = len(train_data[first_article][0]['custom_enc'])

    return train_loader, valid_loader, test_loader, vocab, embeddings, custom_hs


def sample_parameters(t):
    """
    randomly choose a set of parameters t times
    """

    activations = ['swish', 'relu', 'sigmoid']
    optimizers = ['sgd', 'adam', 'rmsprop', 'adagrad', 'adadelta']
    learning_rates = [0.01, 0.001, 0.0001]
    dropout_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    batch_sizes = [32, 64, 128, 256, 512, 1024]

    activs, optims, lrs, drs, bss = [], [], [], [], []
    # sample parameters
    for _ in range(t):
        activs.append(np.random.choice(activations))
        optims.append(np.random.choice(optimizers))
        lrs.append(np.random.choice(learning_rates))
        drs.append(np.random.choice(dropout_rates))
        bss.append(np.random.choice(batch_sizes))

    return activs, optims, lrs, drs, bss


def check_param_ambiguity():
    if args.sentence_hs != args.utterance_hs:
        logger.info("WARNING: ambiguity between sentence (%d) and utterance (%d) hs. Using %d" % (
                args.sentence_hs, args.utterance_hs, args.sentence_hs
        ))
    if args.sentence_bidir != args.utterance_bidir:
        logger.info("WARNING: ambiguity between sentence (%s) and utterance (%s) bidir. Using %s" % (
                args.sentence_bidir, args.utterance_bidir, args.sentence_bidir
        ))
    if args.sentence_dropout != args.utterance_dropout:
        logger.info("WARNING: ambiguity between sentence (%s) and utterance (%s) dropout. Using %s" % (
                args.sentence_dropout, args.utterance_dropout, args.sentence_dropout
        ))

    if args.article_hs != args.context_hs:
        logger.info("WARNING: ambiguity between article (%d) and context (%d) hs. Using %d" % (
                args.article_hs, args.context_hs, args.article_hs
        ))
    if args.article_bidir != args.context_bidir:
        logger.info("WARNING: ambiguity between article (%s) and context (%s) bidir. Using %s" % (
                args.article_bidir, args.context_bidir, args.article_bidir
        ))
    if args.article_dropout != args.context_dropout:
        logger.info("WARNING: ambiguity between article (%s) and context (%s) dropout. Using %s" % (
                args.article_dropout, args.context_dropout, args.article_dropout
        ))


def one_epoch(dqn, huber, mse, data_loader, optimizer=None):
    """
    Performs one epoch over the specified data
    :param dqn: q-network to perform forward pass
    :param huber: huber loss function
    :param mse: mse loss function
    :param data_loader: data iterator
    :param optimizer: optimizer to perform gradient descent.
                        if None, do not train.
    :return: average huber loss, mse loss
    """
    epoch_huber_loss = 0.0
    epoch_mse_loss = 0.0
    nb_batches = 0.0

    if args.mode == 'mlp':
        for i, (custom_encs, rewards) in enumerate(data_loader):
            # IF in training mode
            if optimizer:
                # Reset gradients
                optimizer.zero_grad()

            # Convert Tensors to Variables
            custom_encs = to_var(custom_encs)
            rewards = to_var(rewards)

            # Forward pass: predict q-values
            q_values = dqn(custom_encs)

            # Compute loss
            huber_loss = huber(q_values, rewards)
            mse_loss = mse(q_values, rewards)
            if args.verbose:
                logger.info("step %.3d - huber loss %.6f - mse loss %.6f" % (
                    i + 1, huber_loss.data[0], mse_loss.data[0]
                ))

            # IF in training mode
            if optimizer:
                # Compute loss gradients w.r.t parameters
                huber_loss.backward()
                # Update parameters
                optimizer.step()

            epoch_huber_loss += huber_loss.data[0]
            epoch_mse_loss += mse_loss.data[0]
            nb_batches += 1

    else:
        for i, (articles_tensors, n_sents, l_sents,
                contexts_tensors, n_turns, l_turns,
                candidates_tensors, n_tokens,
                custom_encs, rewards) in enumerate(data_loader):
            # IF in training mode
            if optimizer:
                # Reset gradients
                optimizer.zero_grad()

            # Convert Tensors to Variables
            articles_tensors = to_var(articles_tensors)
            contexts_tensors = to_var(contexts_tensors)
            candidates_tensors = to_var(candidates_tensors)
            custom_encs = to_var(custom_encs)
            rewards = to_var(rewards)

            # Forward pass: predict q-values
            q_values = dqn(
                articles_tensors, n_sents, l_sents,
                contexts_tensors, n_turns, l_turns,
                candidates_tensors, n_tokens,
                custom_encs
            )

            # Compute loss
            huber_loss = huber(q_values, rewards)
            mse_loss = mse(q_values, rewards)
            if args.verbose:
                logger.info("step %.3d - huber loss %.6f - mse loss %.6f" % (
                    i + 1, huber_loss.data[0], mse_loss.data[0]
                ))

            # IF in training mode
            if optimizer:
                # Compute loss gradients w.r.t parameters
                huber_loss.backward()
                # Update parameters
                optimizer.step()

            epoch_huber_loss += huber_loss.data[0]
            epoch_mse_loss += mse_loss.data[0]
            nb_batches += 1

    epoch_huber_loss /= nb_batches
    epoch_mse_loss /= nb_batches

    return epoch_huber_loss, epoch_mse_loss


def one_episode(dqn, huber, mse, data_loader, optimizer=None):
    """
    Performs one epoch over the specified data
    :param dqn: q-network to perform forward pass
    :param huber: huber loss function
    :param mse: mse loss function
    :param data_loader: data iterator
    :param optimizer: optimizer to perform gradient descent.
                        if None, do not train.
    :return: average huber loss, mse loss
    """
    epoch_huber_loss = 0.0
    epoch_mse_loss = 0.0
    nb_batches = 0.0

    if args.mode == 'mlp':
        # TODO: find a way to get (state, action, reward, next_state)

        for i, (custom_encs, rewards) in enumerate(data_loader):
            # IF in training mode
            if optimizer:
                # Reset gradients
                optimizer.zero_grad()

            # Convert Tensors to Variables
            custom_encs = to_var(custom_encs)
            rewards = to_var(rewards)

            # Forward pass: predict q-values
            q_values = dqn(custom_encs)

            # Compute loss
            huber_loss = huber(q_values, rewards)
            mse_loss = mse(q_values, rewards)
            if args.verbose:
                logger.info("step %.3d - huber loss %.6f - mse loss %.6f" % (
                    i + 1, huber_loss.data[0], mse_loss.data[0]
                ))

            # IF in training mode
            if optimizer:
                # Compute loss gradients w.r.t parameters
                huber_loss.backward()
                # Update parameters
                optimizer.step()

            epoch_huber_loss += huber_loss.data[0]
            epoch_mse_loss += mse_loss.data[0]
            nb_batches += 1

    else:
        # TODO: find a way to get (state, action, reward, next_state)
        # state = (article, context)
        # action = (candidate +custom_enc?)
        # next_state = (article, context+candidate)
        # reward = reward
        for i, (articles_tensors, n_sents, l_sents,
                contexts_tensors, n_turns, l_turns,
                candidates_tensors, n_tokens,
                custom_encs, rewards) in enumerate(data_loader):
            # IF in training mode
            if optimizer:
                # Reset gradients
                optimizer.zero_grad()

            # Convert Tensors to Variables
            articles_tensors = to_var(articles_tensors)
            contexts_tensors = to_var(contexts_tensors)
            candidates_tensors = to_var(candidates_tensors)
            custom_encs = to_var(custom_encs)
            rewards = to_var(rewards)

            # Forward pass: predict q-values <--> select_action
            q_values = dqn(
                articles_tensors, n_sents, l_sents,
                contexts_tensors, n_turns, l_turns,
                candidates_tensors, n_tokens,
                custom_encs
            )

            # Compute loss
            huber_loss = huber(q_values, rewards)
            mse_loss = mse(q_values, rewards)
            if args.verbose:
                logger.info("step %.3d - huber loss %.6f - mse loss %.6f" % (
                    i + 1, huber_loss.data[0], mse_loss.data[0]
                ))

            # IF in training mode
            if optimizer:
                # Compute loss gradients w.r.t parameters
                huber_loss.backward()
                # Update parameters
                optimizer.step()

            epoch_huber_loss += huber_loss.data[0]
            epoch_mse_loss += mse_loss.data[0]
            nb_batches += 1

    epoch_huber_loss /= nb_batches
    epoch_mse_loss /= nb_batches

    return epoch_huber_loss, epoch_mse_loss


def main():
    train_loader, valid_loader, test_loader,\
        vocab, embeddings, custom_hs = get_data(args.data_f, args.vocab_f)

    logger.info("")
    logger.info("Building Q-Network...")
    if args.mode == 'mlp':
        model_name = "QNetwork"
        dqn = QNetwork(custom_hs, args.mlp_activation, args.mlp_dropout)

    else:
        if args.mode == 'rnn+mlp':
            model_name = 'DeepQNetwork'
            check_param_ambiguity()
        elif args.mode == 'rnn+rnn+mlp':
            model_name = 'VeryDeepQNetwork'
        else:
            raise NotImplementedError("ERROR: Unknown mode: %s" % args.mode)

        dqn = DeepQNetwork(
            args.mode, embeddings, args.fix_embeddings,
            args.sentence_hs, args.sentence_bidir, args.sentence_dropout,
            args.article_hs, args.article_bidir, args.article_dropout,
            args.utterance_hs, args.utterance_bidir, args.utterance_dropout,
            args.context_hs, args.context_bidir, args.context_dropout,
            args.rnn_gate,
            custom_hs, args.mlp_activation, args.mlp_dropout
        )

    logger.info(dqn)

    model_id = time.time()

    # save parameters
    with open("./models/q_estimator/%s_%s_args.pkl" % (model_name, model_id), 'wb') as f:
        pkl.dump(args, f)

    if torch.cuda.is_available():
        logger.info("")
        logger.info("cuda available! Moving variables to cuda %d..." % args.gpu)
        dqn.cuda()

    # get list of parameters to train
    params = filter(lambda p: p.requires_grad, dqn.parameters())
    # create optimizer ('adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta')
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params=params, lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=params, lr=args.learning_rate)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(params=params, lr=args.learning_rate)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(params=params, lr=args.learning_rate)
    elif args.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(params=params, lr=args.learning_rate)
    else:
        logger.info("ERROR: unknown optimizer: %s" % args.optimizer)
        return

    huber = torch.nn.SmoothL1Loss()  # MSE used in -1 < . < 1 ; Absolute used elsewhere
    mse = torch.nn.MSELoss()

    start_time = time.time()
    best_valid = 100000.
    patience = args.patience

    train_losses = []
    valid_losses = []

    logger.info("")
    logger.info("Training model...")

    # TODO: update to do Q-learning like in : http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    for epoch in range(args.epochs):
        logger.info("***********************************")
        # Perform one epoch and return average losses:
        train_huber_loss, train_mse_loss = one_epoch(
            dqn, huber, mse, train_loader, optimizer=optimizer
        )
        # Save train losses
        train_losses.append((train_huber_loss, train_mse_loss))
        logger.info("Epoch: %d - huber loss: %g - mse loss: %g" % (
            epoch+1, train_huber_loss, train_mse_loss
        ))

        logger.info("computing validation losses...")
        # Compute validation losses
        valid_huber_loss, valid_mse_loss = one_epoch(
            dqn, huber, mse, valid_loader, optimizer=None
        )
        # Save validation losses
        valid_losses.append((valid_huber_loss, valid_mse_loss))
        logger.info("Valid huber loss: %g - best huber loss: %g" % (
            valid_huber_loss, best_valid
        ))

        if valid_huber_loss < best_valid:
            # Reset best validation loss
            best_valid = valid_huber_loss
            # Save network
            torch.save(
                dqn.state_dict(),
                "./models/q_estimator/%s_%s_dqn.pt" % (model_name, model_id)
            )
            # Reset patience
            patience = args.patience
            logger.info("Saved new model.")
        else:
            patience -= 1
            logger.info("No improvement. patience: %d" % patience)

        if patience <= 0:
            break

    logger.info("Finished training. Time elapsed: %g seconds" % (
        time.time() - start_time
    ))

    # TODO: plot train & validation losses
    # valid = red  -- or - -
    # train = blue -- or - -

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected. Got %s' % v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_f", type=str, help="Path to json data file")
    parser.add_argument("vocab_f", type=str, help="Path to pkl vocabbulary file")
    parser.add_argument("mode", choices=['mlp', 'rnn+mlp', 'rnn+rnn+mlp'],
                        default='mlp', help="type of neural network to train")
    parser.add_argument("-g",  "--gpu", type=int, default=0, help="GPU number to use")
    parser.add_argument("-v", "--verbose", type=str2bool, default='no', help="Be verbose")
    # training parameters:
    parser.add_argument("--optimizer", choices=['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta'],
                        default='adam', help="Optimizer to use")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                        help="Learning rate for the optimizer")
    parser.add_argument("--fix_embeddings", type=str2bool, default='no',
                        help="keep word_embeddings fixed during training")
    parser.add_argument("-p", "--patience", type=int, default=20,
                        help="Number of training steps to wait before stopping when validation accuracy doesn't increase")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of training passes to do on the full train set")
    parser.add_argument("-bs", "--batch_size", type=int, default=128,
                        help="batch size during training")

    # TODO: check if using the next one
    parser.add_argument("-pm", "--previous_model", default=None,
                        help="path and prefix_ of the model to continue training from")
    # TODO: check if using the previous one

    # network architecture:
    parser.add_argument("--rnn_gate", choices=['rnn', 'gru', 'lstm'],
                        default="gru", help="RNN gate type.")
    ## sentence rnn
    parser.add_argument("--sentence_hs", type=int, default=300,
                        help="encodding size of each sentence")
    parser.add_argument("--sentence_bidir", type=str2bool, default='no',
                        help="sentence rnn is bidirectional")
    parser.add_argument("--sentence_dropout", type=float, default=0.1,
                        help="dropout probability in sentence rnn")
    ## article rnn
    parser.add_argument("--article_hs", type=int, default=500,
                        help="encodding size of each article")
    parser.add_argument("--article_bidir", type=str2bool, default='no',
                        help="article rnn is bidirectional")
    parser.add_argument("--article_dropout", type=float, default=0.1,
                        help="dropout probability in article rnn")
    ## utterance rnn
    parser.add_argument("--utterance_hs", type=int, default=300,
                        help="encodding size of each utterance")
    parser.add_argument("--utterance_bidir", type=str2bool, default='no',
                        help="utterance rnn is bidirectional")
    parser.add_argument("--utterance_dropout", type=float, default=0.1,
                        help="dropout probability in utterance rnn")
    ## context rnn
    parser.add_argument("--context_hs", type=int, default=500,
                        help="encodding size of each context")
    parser.add_argument("--context_bidir", type=str2bool, default='no',
                        help="context rnn is bidirectional")
    parser.add_argument("--context_dropout", type=float, default=0.1,
                        help="dropout probability in context rnn")
    ## mlp
    parser.add_argument("--mlp_activation", choices=['sigmoid', 'relu', 'swish'],
                        type=str, default='relu', help="Activation function")
    parser.add_argument("--mlp_dropout", type=float, default=0.1,
                        help="dropout probability in mlp")
    args = parser.parse_args()
    logger.info("")
    logger.info("%s" % args)
    logger.info("")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

    main()

