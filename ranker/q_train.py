import torch
from torch.autograd import Variable
from q_networks import to_var, to_tensor, QNetwork, DeepQNetwork
from q_data_loader import get_loader
from extract_amt_for_q_ranker import Vocabulary
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
        json=train_data, vocab=vocab, q_net_mode=args.mode, rescale_rewards=not args.predict_rewards,
        batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    valid_loader = get_loader(
        json=valid_data, vocab=vocab, q_net_mode=args.mode, rescale_rewards=not args.predict_rewards,
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = get_loader(
        json=test_data, vocab=vocab, q_net_mode=args.mode, rescale_rewards=not args.predict_rewards,
        batch_size=args.batch_size, shuffle=False, num_workers=0
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
    custom_hs = len(train_data[first_article][0]['action']['custom_enc'])

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

def one_epoch(dqn, loss, data_loader, optimizer=None, test=False):
    """
    Performs one epoch over the specified data
    :param dqn: q-network to perform forward pass
    :param loss: cross entropy loss function
    :param data_loader: data iterator
    :param optimizer: optimizer to perform gradient descent.
                        if None, do not train.
    :param test: print some predictions
    :return: average loss & accuracy dictionary: acc, TP, TN, FP, FN
    """
    if args.mode == 'mlp':
        epoch_loss, epoch_accuracy, nb_batches = _one_mlp_epoch(
            dqn, loss, data_loader, optimizer, test
        )
    else:
        epoch_loss, epoch_accuracy, nb_batches = _one_rnn_epoch(
            dqn, loss, data_loader, optimizer, test
        )

    epoch_loss /= nb_batches
    epoch_accuracy['acc'] /= nb_batches
    epoch_accuracy['TP'] /= nb_batches
    epoch_accuracy['TN'] /= nb_batches
    epoch_accuracy['FP'] /= nb_batches
    epoch_accuracy['FN'] /= nb_batches

    return epoch_loss, epoch_accuracy

def _one_mlp_epoch(dqn, loss, data_loader, optimizer, test):
    """
    Performs one epoch over the specified data
    :param dqn: q-network to perform forward pass
    :param loss: cross entropy loss function
    :param data_loader: data iterator
    :param optimizer: optimizer to perform gradient descent.
                        if None, do not train.
    :param test: print some predictions
    :return: epoch loss & accuracy
    """
    epoch_loss = 0.0
    epoch_accuracy = {'acc': 0., 'TP': 0., 'TN': 0., 'FP': 0., 'FN': 0.}
    nb_batches = 0.0

    # variable used in test mode (test=True)
    current_article = None
    current_context = None

    '''
    data loader returns:
        articles, contexts, candidates,
        custom_encs, torch.Tensor(rewards), non_final_mask, non_final_next_custom_encs
    '''
    for step, (articles, contexts, candidates,
               custom_encs, rewards, _, _) in enumerate(data_loader):
        # articles : tuple of list of sentences. each sentence is a Tensor. ~(bs, n_sents, n_tokens)
        # contexts : tuple of list of turns. each turn is a Tensor. ~(bs, n_turns, n_tokens)
        # candidates : tuple of Tensors. ~(bs, n_tokens)
        # custom_encs : Tensor ~(bs, enc)
        # rewards : Tensor ~(bs,)

        # Convert Tensors to Variables
        custom_encs = to_var(custom_encs)  # ~(bs, enc)
        rewards = to_var(rewards.long())  # ~(bs)

        # Forward pass: predict rewards
        predictions = dqn(custom_encs)  # ~(bs, 2)

        # Compute loss
        tmp_loss = loss(predictions, rewards)

        # Compute accuracy, TP, TN, FP, FN
        tmp_tp, tmp_tn, tmp_fp, tmp_fn = 0., 0., 0., 0.
        for idx, r in enumerate(rewards.data):
            _, pred = torch.max(predictions[idx], 0)
            pred = pred.data
            if r == 0 and pred.eq(0)[0]:
                tmp_tn += 1.
            elif r == 0 and pred.eq(1)[0]:
                tmp_fp += 1.
            elif r == 1 and pred.eq(0)[0]:
                tmp_fn += 1.
            elif r == 1 and pred.eq(1)[0]:
                tmp_tp += 1.
            else:
                print "WARNING: unknown reward (%s) or prediction (%s)" % (r, pred)
        tmp_acc = (tmp_tn + tmp_tp) / (tmp_tp + tmp_tn + tmp_fp + tmp_fn)
        epoch_accuracy['acc'] += tmp_acc
        if tmp_tn + tmp_fp > 0:
            epoch_accuracy['TN'] += (tmp_tn / (tmp_tn + tmp_fp))  # true negative rate = specificity
            epoch_accuracy['FP'] += (tmp_fp / (tmp_tn + tmp_fp))  # false positive rate = fall-out
        if tmp_fn + tmp_tp > 0:
            epoch_accuracy['FN'] += (tmp_fn / (tmp_fn + tmp_tp))  # false negative rate
            epoch_accuracy['TP'] += (tmp_tp / (tmp_fn + tmp_tp))  # true positive rate = sensitivity

        if args.verbose:
            logger.info("step %.3d - loss %.6f - acc %g" % (
                step + 1, tmp_loss.data[0], tmp_acc
            ))

        # IF in training mode
        if optimizer:
            # Reset gradients
            optimizer.zero_grad()
            # Compute loss gradients w.r.t parameters
            tmp_loss.backward()
            # Update parameters
            optimizer.step()

        # IF in testing mode, print the first 40 examples of this batch with proba 0.5
        if test and np.random.choice([0,1]) == 1:
            print "batch %d" % (step+1)
            end = min(len(rewards.data), 40)
            for b_idx in range(end):
                print "  article: ", map(lambda sent: sent.numpy(), articles[b_idx])
                print "  context: ", map(lambda turn: turn.numpy(), contexts[b_idx])
                print "  candidate: ", candidates[b_idx].numpy()
                print "  reward: ", rewards.data[b_idx]
                print "  prediction: ", predictions.data[b_idx].cpu().numpy()
                print "  *********************************"
        epoch_loss += tmp_loss.data[0]
        nb_batches += 1

    return epoch_loss, epoch_accuracy, nb_batches

def _one_rnn_epoch(dqn, loss, data_loader, optimizer, test):
    """
    Performs one epoch over the specified data
    :param dqn: q-network to perform forward pass
    :param loss: cross entropy loss function
    :param data_loader: data iterator
    :param optimizer: optimizer to perform gradient descent.
                        if None, do not train.
    :param test: print some predictions
    :return: epoch loss & accuracy
    """
    epoch_loss = 0.0
    epoch_accuracy = {'acc': 0., 'TP': 0., 'TN': 0., 'FP': 0., 'FN': 0.}
    nb_batches = 0.0

    # variable used in test mode (test=True)
    current_article = None
    current_context = None

    '''
    data loader returns:
        articles, articles_tensor, n_sents, l_sents, \
        contexts, contexts_tensor, n_turns, l_turns, \
        candidates_tensor, n_tokens, \
        custom_encs, torch.Tensor(rewards), \
        non_final_mask, \
        non_final_next_state_tensor, n_non_final_next_turns, l_non_final_next_turns, \
        non_final_next_candidates_tensor, n_non_final_next_candidates, l_non_final_next_candidates, \
        non_final_next_custom_encs
    '''
    for step, (articles, articles_tensors, n_sents, l_sents,
            contexts, contexts_tensors, n_turns, l_turns,
            candidates_tensors, n_tokens,
            custom_encs, rewards,
            _, _, _, _, _, _, _, _) in enumerate(data_loader):
        # articles : tuple of list of sentences. each sentence is a Tensor. ~(bs, n_sents, n_tokens)
        # articles_tensor : Tensor ~(bs x n_sents, max_len)
        # n_sents : Tensor ~(bs)
        # l_sents : Tensor ~(bs x n_sents)
        # contexts : tuple of list of turns. each turn is a Tensor. ~(bs, n_turns, n_tokens)
        # contexts_tensor : Tensor ~(bs x n_turns, max_len)
        # n_turns : Tensor ~(bs)
        # l_turns : Tensir ~(bs x n_turns)
        # candidates_tensor : Tensor ~(bs, max_len)
        # n_tokens : Tensor ~(bs)
        # custom_encs : Tensor ~(bs, enc)
        # rewards : Tensor ~(bs,)

        # Convert Tensors to Variables
        articles_tensors = to_var(articles_tensors)  # ~(bs x n_sents, max_len)
        n_sents = to_var(n_sents)  # ~(bs)
        l_sents = to_var(l_sents)  # ~(bs x n_sents)
        contexts_tensors = to_var(contexts_tensors)  # ~(bs x n_turns, max_len)
        n_turns = to_var(n_turns)  # ~(bs)
        l_turns = to_var(l_turns)  # ~(bs x n_turns)
        candidates_tensors = to_var(candidates_tensors)  # ~(bs, max_len)
        n_tokens = to_var(n_tokens)  # ~(bs)
        custom_encs = to_var(custom_encs)  # ~(bs, enc)
        rewards = to_var(rewards.long())  # ~(bs)

        # Forward pass: predict q-values
        predictions = dqn(
            articles_tensors, n_sents, l_sents,
            contexts_tensors, n_turns, l_turns,
            candidates_tensors, n_tokens,
            custom_encs
        )  # ~ (bs, 2)

        # Compute loss
        tmp_loss = loss(predictions, rewards)

        # Compute accuracy, TP, TN, FP, FN
        tmp_tp, tmp_tn, tmp_fp, tmp_fn = 0., 0., 0., 0.
        for idx, r in enumerate(rewards.data):
            _, pred = torch.max(predictions[idx], 0)
            pred = pred.data
            if r == 0 and pred.eq(0)[0]:
                tmp_tn += 1.
            elif r == 0 and pred.eq(1)[0]:
                tmp_fp += 1.
            elif r == 1 and pred.eq(0)[0]:
                tmp_fn += 1.
            elif r == 1 and pred.eq(1)[0]:
                tmp_tp += 1.
            else:
                print "WARNING: unknown reward (%s) or prediction (%s)" % (r, pred)
        tmp_acc = (tmp_tn + tmp_tp) / (tmp_tp + tmp_tn + tmp_fp + tmp_fn)
        epoch_accuracy['acc'] += tmp_acc
        if tmp_tn + tmp_fp > 0:
            epoch_accuracy['TN'] += (tmp_tn / (tmp_tn + tmp_fp))  # true negative rate = specificity
            epoch_accuracy['FP'] += (tmp_fp / (tmp_tn + tmp_fp))  # false positive rate = fall-out
        if tmp_fn + tmp_tp > 0:
            epoch_accuracy['FN'] += (tmp_fn / (tmp_fn + tmp_tp))  # false negative rate
            epoch_accuracy['TP'] += (tmp_tp / (tmp_fn + tmp_tp))  # true positive rate = sensitivity

        if args.verbose:
            logger.info("step %.3d - loss %.6f - accuracy %g" % (
                step + 1, tmp_loss.data[0], tmp_acc
            ))

        # IF in training mode
        if optimizer:
            # Reset gradients
            optimizer.zero_grad()
            # Compute loss gradients w.r.t parameters
            tmp_loss.backward()
            # Update parameters
            optimizer.step()

        # IF in testing mode, print the first 40 examples of this batch with proba 0.5
        if test and np.random.choice([0, 1]) == 1:
            print "batch %d" % (step + 1)
            end = min(len(rewards.data), 40)
            for b_idx in range(end):
                print "  article: ", map(lambda sent: sent.numpy(), articles[b_idx])
                print "  context: ", map(lambda turn: turn.numpy(), contexts[b_idx])
                print "  candidate: ", candidates_tensors.data[b_idx].cpu().numpy()
                print "  reward: ", rewards.data[b_idx]
                print "  prediction: ", predictions.data[b_idx].cpu().numpy()
                print "  *********************************"
        epoch_loss += tmp_loss.data[0]
        nb_batches += 1

    return epoch_loss, epoch_accuracy, nb_batches


def one_episode(itt, dqn, target_dqn, huber, data_loader, optimizer=None, test=False):
    """
    Performs one episode over the specified data
    :param itt: number of past iterations
    :param dqn: q-network to perform forward pass
    :param dqn_target: target network
    :param huber: huber loss function
    :param data_loader: data iterator
    :param optimizer: optimizer to perform gradient descent.
                        if None, do not train.
    :param test: print some predictions
    :return: average huber loss, number of batch seen
    """
    if args.mode == 'mlp':
        epoch_huber_loss, nb_batches = _one_mlp_episode(
            itt, dqn, target_dqn, huber, data_loader, optimizer, test
        )
    else:
        epoch_huber_loss, nb_batches = _one_rnn_episode(
            itt, dqn, target_dqn, huber, data_loader, optimizer, test
        )

    epoch_huber_loss /= nb_batches

    return epoch_huber_loss, nb_batches

def _one_mlp_episode(itt, dqn, target_dqn, huber, data_loader, optimizer, test):
    """
    Performs one epoch over the specified data
    :param itt: number of past iterations
    :param dqn: q-network to perform forward pass
    :param dqn_target: target network
    :param huber: huber loss function
    :param data_loader: data iterator
    :param optimizer: optimizer to perform gradient descent.
                        if None, do not train.
    :param test: print some predictions
    :return: epoch huber loss, number of batch seen
    """
    epoch_huber_loss = 0.0
    nb_batches = 0.0

    for step, (articles, contexts, candidates,
               custom_encs, rewards,
               non_final_mask, non_final_next_custom_encs) in enumerate(data_loader):
        # articles : tuple of list of sentences. each sentence is a Tensor. ~(bs, n_sents, n_tokens)
        # contexts : tuple of list of turns. each turn is a Tensor. ~(bs, n_turns, n_tokens)
        # candidates : tuple of Tensors. ~(bs, n_tokens)
        # custom_encs : Tensor ~(bs, enc)
        # rewards : Tensor ~(bs,)
        # non_final_mask : boolean list ~(bs)
        # non_final_next_custom_encs : tuple of list of Tensors: ~(bs-, n_actions, enc)

        # Convert array to Tensor
        non_final_mask = to_tensor(non_final_mask, torch.ByteTensor)
        # Convert Tensors to Variables
        custom_encs = to_var(custom_encs)  # ~(bs, enc)
        rewards = to_var(rewards)  # ~(bs,)

        # Forward pass: predict current state-action value
        q_values = dqn(custom_encs)  # ~(bs)

        # ALL next states are None
        if not non_final_mask.any():
            expected_state_action_values = rewards

        # At least one state has a next state:
        else:
            # Compute Q(s', a') for all next state-action pairs.
            # [-> Double DQN: use real dqn for argmax_a and use target dqn to measure q(s', a*) <-]
            max_actions = []  # ~(bs-, enc)
            for actions in non_final_next_custom_encs:
                # We don't want to backprop through the expected action values and volatile
                # will save us on temporarily changing the model parameters'
                # requires_grad to False!
                actions = to_var(to_tensor(actions), volatile=True)  # ~(actions, enc)
                next_qs = dqn(actions)  # ~(actions,)
                max_q_val, max_idx = next_qs.data.max(0)
                # append custom_enc of the max action according to current dqn
                max_actions.append(actions.data[max_idx].view(-1))

            next_state_action_values = to_var(torch.zeros(non_final_mask.size()))
            next_state_action_values[non_final_mask] = target_dqn(
                to_var(torch.stack(max_actions), volatile=True)
            )  # ~(bs-)
            # Now, we don't want to mess up the loss with a volatile flag, so let's
            # clear it. After this, we'll just end up with a Variable that has
            # requires_grad=False
            next_state_action_values.volatile = False
            # Compute the expected Q values
            expected_state_action_values = (next_state_action_values * args.gamma) + rewards

        # Compute loss
        huber_loss = huber(q_values, expected_state_action_values)
        if args.verbose:
            logger.info("step %.3d - huber loss %.6f" % (
                step + 1, huber_loss.data[0]
            ))

        # IF in training mode
        if optimizer:
            # Reset gradients
            optimizer.zero_grad()
            # Compute loss gradients w.r.t parameters
            huber_loss.backward()
            # Clip gradients
            for param in dqn.parameters():
                param.grad.data.clamp_(-1, 1)
            # Update parameters
            optimizer.step()

            ## update target dqn
            if (itt + step) % args.update_frequence == 0:
                logger.info("iteration %d: updating target DQN." % (itt + step))
                target_dqn.load_state_dict(dqn.state_dict())

        # IF in testing mode, print the first 40 examples of this batch with proba 0.5
        if test and np.random.choice([0, 1]) == 1:
            print "batch %d" % (step + 1)
            end = min(len(rewards.data), 40)
            for b_idx in range(end):
                print "  article: ", map(lambda sent: sent.numpy(), articles[b_idx])
                print "  context: ", map(lambda turn: turn.numpy(), contexts[b_idx])
                print "  candidate: ", candidates[b_idx].numpy()
                print "  reward: ", rewards.data[b_idx]
                print "  q(s,a): ", q_values.data[b_idx].cpu().numpy()
                # print "  q(s', a*): ", next_state_action_values.data[b_idx].cpu().numpy()
                print "  expected state action values: ", expected_state_action_values.data[b_idx]
                print "  *********************************"
        epoch_huber_loss += huber_loss.data[0]
        nb_batches += 1

    return epoch_huber_loss, nb_batches

def _one_rnn_episode(itt, dqn, target_dqn, huber, data_loader, optimizer, test):
    """
    Performs one epoch over the specified data
    :param itt: number of past iterations
    :param dqn: q-network to perform forward pass
    :param dqn_target: target network
    :param huber: huber loss function
    :param data_loader: data iterator
    :param optimizer: optimizer to perform gradient descent.
                        if None, do not train.
    :param test: print some predictions
    :return: epoch huber loss, number of batch seen
    """
    epoch_huber_loss = 0.0
    nb_batches = 0.0

    for step, (articles, articles_tensors, n_sents, l_sents,
            contexts, contexts_tensors, n_turns, l_turns,
            candidates_tensors, n_tokens,
            custom_encs, rewards,
            non_final_mask,
            non_final_next_state_tensor, n_non_final_next_turns, l_non_final_next_turns,
            non_final_next_candidates_tensor, n_non_final_next_candidates, l_non_final_next_candidates,
            non_final_next_custom_encs) in enumerate(data_loader):
        # articles : tuple of list of sentences. each sentence is a Tensor. ~(bs, n_sents, n_tokens)
        # articles_tensor : Tensor ~(bs x n_sents, max_len)
        # n_sents : Tensor ~(bs)
        # l_sents : Tensor ~(bs x n_sents)
        # contexts : tuple of list of turns. each turn is a Tensor. ~(bs, n_turns, n_tokens)
        # contexts_tensor : Tensor ~(bs x n_turns, max_len)
        # n_turns : Tensor ~(bs)
        # l_turns : Tensir ~(bs x n_turns)
        # candidates_tensor : Tensor ~(bs, max_len)
        # n_tokens : Tensor ~(bs)
        # custom_encs : Tensor ~(bs, enc)
        # rewards : Tensor ~(bs,)
        # non_final_mask : boolean list ~(bs)
        # ...
        # non_final_next_custom_encs : tuple of list of Tensors: ~(bs-, n_actions, enc)

        # Convert array to Tensor
        non_final_mask = to_tensor(non_final_mask, torch.ByteTensor)
        # Convert Tensors to Variables
        # state:
        articles_tensors_t = to_var(articles_tensors)  # ~(bs x n_sentences, max_len)
        n_sents_t = to_var(n_sents)  # ~(bs)
        l_sents_t = to_var(l_sents)  # ~(bs x n_sentences)
        contexts_tensors_t = to_var(contexts_tensors)  # ~(bs x n_turns, max_len)
        n_turns_t = to_var(n_turns)  # ~(bs)
        l_turns_t = to_var(l_turns)  # ~(bs x n_turns)
        # action:
        candidates_tensors_t = to_var(candidates_tensors)  # ~(bs, max_len)
        n_tokens_t = to_var(n_tokens)  # ~(bs)
        custom_encs_t = to_var(custom_encs)  # ~(bs, enc)
        # reward:
        rewards_t = to_var(rewards)  # ~(bs)

        # Forward pass: predict current state-action value
        q_values = dqn(
            articles_tensors_t, n_sents_t, l_sents_t,
            contexts_tensors_t, n_turns_t, l_turns_t,
            candidates_tensors_t, n_tokens_t,
            custom_encs_t
        ) # ~(bs,)

        # print "articles:", articles_tensors_t.size()
        # print "- n_sentences:", n_sents_t.size()
        # print "- l_sentences:", l_sents_t.size()
        # print "contexts:", contexts_tensors_t.size()
        # print "- n_turns:", n_turns_t.size()
        # print "- l_turns:", l_turns_t.size()
        # print "candidate:", candidates_tensors_t.size()
        # print "- n_tokens:", n_tokens_t.size()
        # print "custom_encodings:", custom_encs_t.size()
        # print "--"
        # print "q_values:", q_values.size()
        # print "--"

        # ALL next states are None
        if not non_final_mask.any():
            expected_state_action_values = rewards_t

        # At least one state has a next state:
        else:

            # Next state:
            # Filter out article for which next state is None!
            n_sents_tp1 = []  # ~(bs-)
            l_sents_tp1 = []  # ~(bs- x n_sent) where each tensor is a sentence
            for idx, article in enumerate(articles):
                # article = list of tensors
                if non_final_mask[idx] == 1:
                    assert n_sents[idx] == len(article)
                    n_sents_tp1.append(len(article))
                    l_sents_tp1.extend([len(sent) for sent in article])
            articles_tensors_tp1 = torch.zeros(len(l_sents_tp1), max(l_sents_tp1)).long()  # ~(bs- x n_sent, max_len)
            i = 0
            for idx, artcl in enumerate(articles):
                # article = list of tensors
                if non_final_mask[idx] == 1:
                    for sent in artcl:
                        assert len(sent) == l_sents_tp1[i]
                        end = l_sents_tp1[i]
                        articles_tensors_tp1[i, :end] = sent[:end]
                        i += 1

            # We don't want to backprop through the expected action values and volatile
            # will save us on temporarily changing the model parameters'
            # requires_grad to False!
            articles_tensors_tp1 = to_var(articles_tensors_tp1, volatile=True)  # ~(bs- x n_sentences, max_len)
            n_sents_tp1 = to_var(to_tensor(n_sents_tp1, torch.LongTensor), volatile=True)  # ~(bs-)
            l_sents_tp1 = to_var(to_tensor(l_sents_tp1, torch.LongTensor), volatile=True)  # ~(bs- x n_sentences)

            contexts_tensors_tp1 = to_var(non_final_next_state_tensor,
                                          volatile=True)  # ~(bs- x n_turns, max_len)
            n_turns_tp1 = to_var(n_non_final_next_turns, volatile=True)  # ~(bs-)
            l_turns_tp1 = to_var(l_non_final_next_turns, volatile=True)  # ~(bs- x n_turns)

            # Compute Q(s', a') for all next state-action pairs.
            # [--Double DQN: use real dqn for argmax_a and use target dqn to measure q(s', a*)--]
            max_actions = {
                'candidate': [],  # ~(bs-, max_len)
                'n_tokens': [],   # ~(bs-)
                'custom_enc': []  # ~(bs-, enc)
            }
            past_n_sentences = 0
            past_n_turns = 0
            past_n_actions = 0
            for idx in range(len(non_final_next_custom_encs)):
                # We don't want to backprop through the expected action values and volatile
                # will save us on temporarily changing the model parameters'
                # requires_grad to False!

                tmp_n_actions = n_non_final_next_candidates[idx]

                ### article
                # grab number of sentences for that article (x n_actions)
                tmp_n_sents = to_var(
                    to_tensor([n_sents_tp1.data[idx]] * tmp_n_actions, torch.LongTensor),
                    volatile=True
                )  # ~(n_actions)
                # grab length of each sentence for that article (x n_actions)
                tmp_l_sents = l_sents_tp1.data[
                                  past_n_sentences: int(past_n_sentences+n_sents_tp1.data[idx])
                              ]  # ~(n_sentences)
                tmp_l_sents = to_var(
                    torch.cat([tmp_l_sents] * tmp_n_actions),
                    volatile=True
                )  # ~(n_actions x n_sentences)
                # grab articles_for_which_next_state_is_not_None[idx] (x n_actions)
                tmp_articles_tensors = articles_tensors_tp1.data[
                                           past_n_sentences: int(past_n_sentences+n_sents_tp1.data[idx])
                                       ]  # ~(n_sent, max_len)
                tmp_articles_tensors = to_var(
                    torch.cat([tmp_articles_tensors] * tmp_n_actions),
                    volatile=True
                )  # ~(n_actions x n_sentences, max_len)

                past_n_sentences += int(n_sents_tp1.data[idx])

                ### context
                # grab number of turns for that context (x n_actions)
                tmp_n_turns = to_var(
                    to_tensor([n_turns_tp1.data[idx]] * tmp_n_actions, torch.LongTensor),
                    volatile=True
                )  # ~(n_actions)
                # grab length of each turn for that context (x n_actions)
                tmp_l_turns = l_turns_tp1.data[
                                  past_n_turns: int(past_n_turns+n_turns_tp1.data[idx])
                              ]  # ~(n_turns)
                tmp_l_turns = to_var(
                    torch.cat([tmp_l_turns] * tmp_n_actions),
                    volatile=True
                )  # ~(n_actions x n_turns)
                # grab context for which next state is not None[idx] (x n_actions)
                tmp_contexts_tensors = contexts_tensors_tp1.data[
                                           past_n_turns: int(past_n_turns+n_turns_tp1.data[idx])
                                       ]  # ~(n_turns, max_len)
                tmp_contexts_tensors = to_var(
                    torch.cat([tmp_contexts_tensors] * tmp_n_actions),
                    volatile=True
                )  # ~(n_actions x n_turns, max_len)

                past_n_turns += int(n_turns_tp1.data[idx])

                ### candidate
                # grab each candidate turns
                tmp_candidates_tensors = to_var(
                    non_final_next_candidates_tensor[past_n_actions: past_n_actions+tmp_n_actions],
                    volatile=True
                )  # ~(n_actions, max_len)
                # grab number of tokens for each candidate turns
                tmp_n_tokens = to_var(
                    l_non_final_next_candidates[past_n_actions: past_n_actions+tmp_n_actions],
                    volatile=True
                )  # ~(n_actions)

                past_n_actions += tmp_n_actions

                ### custom enc
                # grab candidates custom encodings
                tmp_custom_encs = to_var(
                    to_tensor(non_final_next_custom_encs[idx]),
                    volatile=True
                )  # ~(n_actions, enc)

                tmp_q_val = dqn(
                    tmp_articles_tensors, tmp_n_sents, tmp_l_sents,
                    tmp_contexts_tensors, tmp_n_turns, tmp_l_turns,
                    tmp_candidates_tensors, tmp_n_tokens,
                    tmp_custom_encs
                )
                max_q_val, max_idx = tmp_q_val.data.max(0)
                # append custom_enc of the max action according to current dqn
                max_actions['candidate'].append(tmp_candidates_tensors.data[max_idx].view(-1))
                max_actions['n_tokens'].append(tmp_n_tokens.data[max_idx])
                max_actions['custom_enc'].append(tmp_custom_encs.data[max_idx].view(-1))

            next_q_values = target_dqn(
                articles_tensors_tp1, n_sents_tp1, l_sents_tp1,
                contexts_tensors_tp1, n_turns_tp1, l_turns_tp1,
                to_var(torch.stack(max_actions['candidate']), volatile=True),
                to_var(torch.cat(max_actions['n_tokens']), volatile=True),
                to_var(torch.stack(max_actions['custom_enc']), volatile=True)
            )  # ~(bs)
            next_state_action_values = to_var(torch.zeros(non_final_mask.size()))
            next_state_action_values[non_final_mask] = next_q_values

            # print "non_final articles:", articles_tensors_tp1.size()
            # print "- n_sentences:", n_sents_tp1.size()
            # print "- l_sentences:", l_sents_tp1.size()
            # print "next_contexts:", contexts_tensors_tp1.size()
            # print "- n_turns:", n_turns_tp1.size()
            # print "- l_turns:", l_turns_tp1.size()
            # print "best candidate:", torch.stack(max_actions['candidate']).size()
            # print "- n_tokens:", len(max_actions['n_tokens'])
            # print "custom_encodings:", torch.stack(max_actions['custom_enc']).size()
            # print "--"
            # print "next_q_values:", next_q_values.size()
            # print ""

            # Now, we don't want to mess up the loss with a volatile flag, so let's
            # clear it. After this, we'll just end up with a Variable that has
            # requires_grad=False
            next_state_action_values.volatile = False
            # Compute the expected Q values
            expected_state_action_values = (next_state_action_values * args.gamma) + rewards_t

        # Compute loss
        huber_loss = huber(q_values, expected_state_action_values)
        if args.verbose:
            logger.info("step %.3d - huber loss %.6f" % (
                step + 1, huber_loss.data[0]
            ))

        # IF in training mode
        if optimizer:
            # Reset gradients
            optimizer.zero_grad()
            # Compute loss gradients w.r.t parameters
            huber_loss.backward()
            # Update parameters
            optimizer.step()

            ## update target dqn
            if (itt+step) % args.update_frequence == 0:
                logger.info("iteration %d: updating target DQN." % (itt+step))
                target_dqn.load_state_dict(dqn.state_dict())

        # IF in testing mode, print the first 40 examples of this batch with proba 0.5
        if test and np.random.choice([0, 1]) == 1:
            print "batch %d" % (step + 1)
            end = min(len(rewards_t.data), 40)
            for b_idx in range(end):
                print "  article: ", map(lambda sent: sent.numpy(), articles[b_idx])
                print "  context: ", map(lambda turn: turn.numpy(), contexts[b_idx])
                print "  candidate: ", candidates_tensors_t.data[b_idx].cpu().numpy()
                print "  reward: ", rewards_t.data[b_idx]
                print "  q(s,a): ", q_values.data[b_idx].cpu().numpy()
                # print "  q(s', a*): ", next_state_action_values.data[b_idx].cpu().numpy()
                print "  expected state action values: ", expected_state_action_values.data[b_idx]
                print "  *********************************"
        epoch_huber_loss += huber_loss.data[0]
        nb_batches += 1

    return epoch_huber_loss, nb_batches


def main():
    #######################
    # Load Data
    #######################

    # Load toy domain dataset
    if args.debug:
        train_loader, valid_loader, test_loader, \
        vocab, embeddings, custom_hs = get_data("./data/q_ranker_colorful_data.json",
                                                "./data/q_ranker_colorful_vocab.pkl")
    # Load regular dataset
    else:
        train_loader, valid_loader, test_loader,\
            vocab, embeddings, custom_hs = get_data(args.data_f, args.vocab_f)

    #######################
    # Build DQN
    #######################
    logger.info("")
    logger.info("Building Q-Network...")
    model_name = "colorful_" if args.debug else ""
    # MLP network
    if args.mode == 'mlp':
        # model name
        model_name += "RNetwork" if args.predict_rewards else "QNetwork"
        # output dimension
        if args.predict_rewards:
            out = 2
        else:
            out = 1
        # dqn
        dqn = QNetwork(custom_hs, args.mlp_activation, args.mlp_dropout, out)
        dqn_target = QNetwork(custom_hs, args.mlp_activation, args.mlp_dropout, out)
    # RNNs + MLP network
    else:
        # model name
        if args.mode == 'rnn+mlp':
            model_name += "DeepRNetwork" if args.predict_rewards else "DeepQNetwork"
            check_param_ambiguity()
        elif args.mode == 'rnn+rnn+mlp':
            model_name += "VeryDeepRNetwork" if args.predict_rewards else "VeryDeepQNetwork"
        else:
            raise NotImplementedError("ERROR: Unknown mode: %s" % args.mode)
        # output dimension
        if args.predict_rewards:
            out = 2
        else:
            out = 1
        # dqn
        dqn = DeepQNetwork(
            args.mode, embeddings, args.fix_embeddings,
            args.sentence_hs, args.sentence_bidir, args.sentence_dropout,
            args.article_hs, args.article_bidir, args.article_dropout,
            args.utterance_hs, args.utterance_bidir, args.utterance_dropout,
            args.context_hs, args.context_bidir, args.context_dropout,
            args.rnn_gate,
            custom_hs, args.mlp_activation, args.mlp_dropout, out
        )
        dqn_target = DeepQNetwork(
            args.mode, embeddings, args.fix_embeddings,
            args.sentence_hs, args.sentence_bidir, args.sentence_dropout,
            args.article_hs, args.article_bidir, args.article_dropout,
            args.utterance_hs, args.utterance_bidir, args.utterance_dropout,
            args.context_hs, args.context_bidir, args.context_dropout,
            args.rnn_gate,
            custom_hs, args.mlp_activation, args.mlp_dropout, out
        )

    logger.info(dqn)

    model_id = time.time()

    # save parameters
    with open("./models/q_estimator/%s_%s_args.pkl" % (model_name, model_id), 'wb') as f:
        pkl.dump(args, f)

    # moving networks to GPU
    if torch.cuda.is_available():
        logger.info("")
        logger.info("cuda available! Moving variables to cuda %d..." % args.gpu)
        dqn.cuda()
        dqn_target.cuda()

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

    # Define losses
    huber = torch.nn.SmoothL1Loss()  # MSE used in -1 < . < 1 ; Absolute used elsewhere
    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()  # used for classification of immediate reward

    #######################
    # Start Training
    #######################
    start_time = time.time()
    best_valid = 100000.
    best_valid_acc = 0.
    patience = args.patience

    train_losses = []  # list of losses for each epoch
    train_accurs = []  # list of accuracy obj for each epoch
    valid_losses = []  # list of losses for each epoch
    valid_accurs = []  # list of accuracy obj for each epoch

    logger.info("")
    logger.info("Training model...")

    iterations_done = 0
    for epoch in range(args.epochs):
        ###
        # Perform one epoch and return average losses:
        ###
        logger.info("***********************************")

        # predict rewards: use cross-entropy loss
        if args.predict_rewards:
            train_loss, train_acc = one_epoch(
                dqn, ce, train_loader, optimizer=optimizer
            )

            # Save train accuracies
            train_accurs.append(train_acc)

            logger.info("Epoch: %d - train loss: %g - train acc: %g" % (
                epoch + 1, train_loss, train_acc['acc']
            ))

        # predict q values: use huber loss (or MSE)
        else:
            train_loss, steps = one_episode(
                iterations_done, dqn, dqn_target, huber, train_loader, optimizer=optimizer
            )
            iterations_done += steps

            logger.info("Epoch: %d - train loss: %g" % (
                epoch + 1, train_loss
            ))

        # Save train losses (always)
        train_losses.append(train_loss)

        ###
        # Compute validation losses
        ###
        logger.info("computing validation losses...")
        improved = False

        # predict rewards: use cross-entropy loss
        if args.predict_rewards:
            valid_loss, valid_acc = one_epoch(
                dqn, ce, valid_loader
            )

            # save validation accuracies
            valid_accurs.append(valid_acc)

            logger.info("Valid loss: %g - Valid acc: %g - best valid accuracy: %g" % (
                valid_loss, valid_acc['acc'], best_valid_acc
            ))

            # Early stopping on valid accuracy
            if valid_acc['acc'] > best_valid_acc:
                # Reset best validation accuracy
                best_valid_acc = valid_acc['acc']
                improved = True

        # predict q values: use huber loss (or MSE)
        else:
            valid_loss, _ = one_episode(
                0, dqn, dqn_target, huber, valid_loader
            )

            logger.info("Valid loss: %g - best valid loss: %g" % (
                valid_loss, best_valid
            ))

            # Early stopping on valid loss
            if valid_loss < best_valid:
                # Reset best validation loss
                best_valid = valid_loss
                improved = True

        # Save validation losses (always)
        valid_losses.append(valid_loss)

        ###
        # Early stopping
        ###
        if improved:
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

    logger.info("Saving timings...")
    with open("./models/q_estimator/%s_%s_timings.json" % (model_name, model_id), 'wb') as f:
        json.dump(
            {'train_losses': train_losses,
             'train_accurs': train_accurs,
             'valid_losses': valid_losses,
             'valid_accurs': valid_accurs},
            f
        )

    #######################
    # Testing
    #######################
    # if args.debug:
    #     # predict rewards
    #     if args.predict_rewards:
    #         test_loss, test_acc = one_epoch(dqn, ce, test_loader, test=True)
    #     # predict q values
    #     else:
    #         test_loss, _ = one_episode(0, dqn, dqn_target, huber, test_loader, test=True)
    #     logger.info("\nTest huber loss: %g" % test_loss)


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
    parser.add_argument("--predict_rewards", type=str2bool, default='no',
                        help="Decide if we predict rewards of q-values")
    parser.add_argument("-g",  "--gpu", type=int, default=0, help="GPU number to use")
    parser.add_argument("-v", "--verbose", type=str2bool, default='no', help="Be verbose")
    parser.add_argument("-d", "--debug", type=str2bool, default='no', help="use toy domain & print on test set")
    # training parameters:
    parser.add_argument("--optimizer", choices=['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta'],
                        default='adam', help="Optimizer to use")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                        help="Learning rate for the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor")
    parser.add_argument("--fix_embeddings", type=str2bool, default='no',
                        help="keep word_embeddings fixed during training")
    parser.add_argument("-p", "--patience", type=int, default=20,
                        help="Number of training steps to wait before stopping when validation accuracy doesn't increase")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of training passes to do on the full train set")
    parser.add_argument("-bs", "--batch_size", type=int, default=128,
                        help="batch size during training")
    parser.add_argument("--update_frequence", type=int, default=100,
                        help="number of iterations to do before updating target dqn")

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
                        type=str, default='swish', help="Activation function")
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

