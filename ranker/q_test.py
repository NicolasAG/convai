import torch
import torch.nn.functional as F
from torch.autograd import Variable
from q_networks import to_var, to_tensor, QNetwork, DeepQNetwork
from q_data_loader import get_loader, collate_fn
from extract_amt_for_q_ranker import Vocabulary
from embedding_metrics import w2v
from q_experiments import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pkl
import argparse
import logging
import copy
import json
import time
import sys
import os

logger = logging.getLogger(__name__)


def get_raw_data(old_args=None):
    """
    Load raw data. Takes a while!
    """

    if old_args:
        # Load toy domain dataset
        if old_args['debug']:
            data_f = "./data/q_ranker_colorful_data.json"
        # Load regular dataset
        else:
            data_f = old_args['data_f']
    # Load default dataset
    else:
        # oversampled --> useless for test
        #data_f = "./data/q_ranker_amt_data++_1525301962.86.pkl"
        # regular
        data_f = "./data/q_ranker_amt_data_1524939554.0.pkl"

    logger.info("")
    logger.info("Loading data from %s..." % data_f)
    with open(data_f.replace('.json', '.pkl'), 'rb') as f:
        raw_data = pkl.load(f)

    return raw_data


def get_data(old_args, raw_data):
    """
    Load data to train Q Network.
    :return: vocabulary, testing examples, embeddings
    """
    # Load toy domain dataset
    if old_args['debug']:
        vocab_f = "./data/q_ranker_colorful_vocab.pkl"
    # Load regular dataset
    else:
        vocab_f = old_args['vocab_f']

    test_data = raw_data['test'][0]
    test_size = raw_data['test'][1]
    logger.info("got %d test examples" % test_size)

    logger.info("")
    logger.info("Loading vocabulary...")
    with open(vocab_f, 'rb') as f:
        vocab = pkl.load(f)
    logger.info("number of unique tokens: %d" % len(vocab))

    logger.info("")
    logger.info("Get data loaders...")
    test_loader, test_conv = get_loader(
        json=test_data, vocab=vocab, q_net_mode=old_args['mode'], rescale_rewards=not old_args['predict_rewards'],
        batch_size=old_args['batch_size'], shuffle=False, num_workers=0
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

    # take arbitrarily the first testing example to find the size of the custom encoding
    first_article = test_data.keys()[0]
    custom_hs = len(test_data[first_article][0]['action']['custom_enc'])

    return test_conv, test_loader, vocab, embeddings, custom_hs


def one_epoch(dqn, loss, data_loader, mode):
    """
    Performs one epoch over the specified data
    :param dqn: q-network to perform forward pass
    :param loss: cross entropy loss function
    :param data_loader: data iterator
    :return: average loss & accuracy dictionary: acc, TP, TN, FP, FN
    """
    if mode == 'mlp':
        epoch_loss, epoch_accuracy, nb_batches = _one_mlp_epoch(dqn, loss, data_loader)
    else:
        epoch_loss, epoch_accuracy, nb_batches = _one_rnn_epoch(dqn, loss, data_loader)

    epoch_loss /= nb_batches
    epoch_accuracy['acc'] /= nb_batches
    epoch_accuracy['TP'] /= nb_batches
    epoch_accuracy['TN'] /= nb_batches
    epoch_accuracy['FP'] /= nb_batches
    epoch_accuracy['FN'] /= nb_batches

    return epoch_loss, epoch_accuracy

def _one_mlp_epoch(dqn, loss, data_loader):
    """
    Performs one epoch over the specified data
    :param dqn: q-network to perform forward pass
    :param loss: cross entropy loss function
    :param data_loader: data iterator
    :return: epoch loss & accuracy
    """
    epoch_loss = 0.0
    epoch_accuracy = {'acc': 0., 'TP': 0., 'TN': 0., 'FP': 0., 'FN': 0.}
    nb_batches = 0.0

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

        # print the first 40 examples of this batch with proba 0.5
        if args.verbose and np.random.choice([0,1]) == 1:
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

def _one_rnn_epoch(dqn, loss, data_loader):
    """
    Performs one epoch over the specified data
    :param dqn: q-network to perform forward pass
    :param loss: cross entropy loss function
    :param data_loader: data iterator
    :return: epoch loss & accuracy
    """
    epoch_loss = 0.0
    epoch_accuracy = {'acc': 0., 'TP': 0., 'TN': 0., 'FP': 0., 'FN': 0.}
    nb_batches = 0.0

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

        # print the first 40 examples of this batch with proba 0.5
        if args.verbose and np.random.choice([0, 1]) == 1:
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


def one_episode(dqn, dqn_target, gamma, huber, data_loader, mode):
    """
    Performs one episode over the specified data
    :param dqn: q-network to perform forward pass
    :param huber: huber loss function
    :param data_loader: data iterator
    :return: average huber loss, number of batch seen
    """
    if mode == 'mlp':
        epoch_huber_loss, nb_batches = _one_mlp_episode(dqn, dqn_target, gamma, huber, data_loader)
    else:
        epoch_huber_loss, nb_batches = _one_rnn_episode(dqn, dqn_target, gamma, huber, data_loader)

    epoch_huber_loss /= nb_batches

    return epoch_huber_loss, nb_batches

def _one_mlp_episode(dqn, dqn_target, gamma, huber, data_loader):
    """
    Performs one epoch over the specified data
    :param dqn: q-network to perform forward pass
    :param huber: huber loss function
    :param data_loader: data iterator
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
            next_state_action_values[non_final_mask] = dqn_target(
                to_var(torch.stack(max_actions), volatile=True)
            )  # ~(bs-)
            # Now, we don't want to mess up the loss with a volatile flag, so let's
            # clear it. After this, we'll just end up with a Variable that has
            # requires_grad=False
            next_state_action_values.volatile = False
            # Compute the expected Q values
            expected_state_action_values = (next_state_action_values * gamma) + rewards

        # Compute loss
        huber_loss = huber(q_values, expected_state_action_values)
        if args.verbose:
            logger.info("step %.3d - huber loss %.6f" % (
                step + 1, huber_loss.data[0]
            ))

        # print the first 40 examples of this batch with proba 0.5
        if args.verbose and np.random.choice([0, 1]) == 1:
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

def _one_rnn_episode(dqn, dqn_target, gamma, huber, data_loader):
    """
    Performs one epoch over the specified data
    :param dqn: q-network to perform forward pass
    :param huber: huber loss function
    :param data_loader: data iterator
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

            next_q_values = dqn_target(
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
            expected_state_action_values = (next_state_action_values * gamma) + rewards_t

        # Compute loss
        huber_loss = huber(q_values, expected_state_action_values)
        if args.verbose:
            logger.info("step %.3d - huber loss %.6f" % (
                step + 1, huber_loss.data[0]
            ))

        # print the first 40 examples of this batch with proba 0.5
        if args.verbose and np.random.choice([0, 1]) == 1:
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

    if os.path.exists('%s_args.pkl' % args.model_prefix):
        with open('%s_args.pkl' % args.model_prefix, 'rb') as f:
            old_args = pkl.load(f)
            old_params = to_dict(old_args)
    else:
        with open('%s_params.json' % args.model_prefix, 'rb') as f:
            old_params = json.load(f)

    logger.info("old parameters: %s" % old_params)

    # TODO: next line is super long and takes a lot of memory.
    #  remove when testing more than one model
    #raw_data = get_raw_data(old_args=old_params)
    raw_data = get_raw_data()

    test_conv, test_loader, vocab, embeddings, custom_hs = get_data(old_params, raw_data)

    #######################
    # Build DQN
    #######################
    logger.info("")
    logger.info("Building Q-Network...")
    # MLP network
    if old_params['mode'] == 'mlp':
        # output dimension
        if old_params['predict_rewards']:
            out = 2
        else:
            out = 1
        # dqn
        dqn = QNetwork(custom_hs, old_params['mlp_activation'], old_params['mlp_dropout'], out)
        dqn_target = QNetwork(custom_hs, old_params['mlp_activation'], old_params['mlp_dropout'], out)
    # RNNs + MLP network
    else:
        # output dimension
        if old_params['predict_rewards']:
            out = 2
        else:
            out = 1
        # dqn
        dqn = DeepQNetwork(
            old_params['mode'], embeddings, old_params['fix_embeddings'],
            old_params['sentence_hs'], old_params['sentence_bidir'], old_params['sentence_dropout'],
            old_params['article_hs'], old_params['article_bidir'], old_params['article_dropout'],
            old_params['utterance_hs'], old_params['utterance_bidir'], old_params['utterance_dropout'],
            old_params['context_hs'], old_params['context_bidir'], old_params['context_dropout'],
            old_params['rnn_gate'],
            custom_hs, old_params['mlp_activation'], old_params['mlp_dropout'], out
        )
        dqn_target = DeepQNetwork(
            old_params['mode'], embeddings, old_params['fix_embeddings'],
            old_params['sentence_hs'], old_params['sentence_bidir'], old_params['sentence_dropout'],
            old_params['article_hs'], old_params['article_bidir'], old_params['article_dropout'],
            old_params['utterance_hs'], old_params['utterance_bidir'], old_params['utterance_dropout'],
            old_params['context_hs'], old_params['context_bidir'], old_params['context_dropout'],
            old_params['rnn_gate'],
            custom_hs, old_params['mlp_activation'], old_params['mlp_dropout'], out
        )

    dqn.load_state_dict(torch.load("%s_dqn.pt" % args.model_prefix))
    dqn_target.load_state_dict(torch.load("%s_dqn.pt" % args.model_prefix))

    logger.info(dqn)

    # moving networks to GPU
    if torch.cuda.is_available():
        logger.info("")
        logger.info("cuda available! Moving variables to cuda %d..." % args.gpu)
        dqn.cuda()
        dqn_target.cuda()

    # Define losses
    huber = torch.nn.SmoothL1Loss()  # MSE used in -1 < . < 1 ; Absolute used elsewhere
    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()  # used for classification of immediate reward

    ############################
    # Plot train & valid stats #
    ############################
    logger.info("")
    logger.info("Plotting timings...")
    with open("%s_timings.json" % args.model_prefix, 'rb') as f:
        timings = json.load(f)

    train_losses = timings['train_losses']
    train_accs = [e['acc'] for e in timings['train_accurs']]
    valid_losses = timings['valid_losses']
    valid_accs = [e['acc'] for e in timings['valid_accurs']]

    if old_params['predict_rewards']:
        # Predicting immediate Reward
        assert len(train_losses) == len(valid_losses) == len(train_accs) == len(valid_accs)
    else:
        # Predicting Q value
        assert len(train_losses) == len(valid_losses)
        assert len(train_accs) == len(valid_accs) == 0

    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='train')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, 'r-', label='valid')
    plt.title("Training and Validation loss over time")
    plt.xlabel("epochs")
    plt.ylabel("CrossEntropy Loss" if old_params['predict_rewards'] else "Huber Loss")
    plt.legend(loc='best')
    plt.savefig("%s_losses.png" % args.model_prefix)
    plt.close()

    best_valid_loss_idx = np.argmin(valid_losses)
    logger.info("best valid loss: %g achieved at epoch %d" %
                (valid_losses[best_valid_loss_idx], best_valid_loss_idx))
    logger.info("training loss at this epoch: %g" % train_losses[best_valid_loss_idx])

    if old_params['predict_rewards']:
        plt.plot(range(1, len(train_accs) + 1), train_accs, 'b-', label='train')
        plt.plot(range(1, len(valid_accs) + 1), valid_accs, 'r-', label='valid')
        plt.title("Training and Validation accuracy over time")
        plt.xlabel("epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        plt.savefig("%s_accuracy.png" % args.model_prefix)
        plt.close()

        best_valid_acc_idx = np.argmax(valid_accs)
        logger.info("best valid acc: %g achieved at epoch %d" %
                    (valid_accs[best_valid_acc_idx], best_valid_acc_idx))
        logger.info("training acc at this epoch: %g" % train_accs[best_valid_acc_idx])

    logger.info("done.")

    #######################
    # Start Testing
    #######################
    start_time = time.time()
    logger.info("")
    logger.info("Testing model in batches...")

    # put in inference mode (Dropout OFF)
    dqn.eval()
    dqn_target.eval()

    # predict rewards: use cross-entropy loss
    if old_params['predict_rewards']:
        loss, accuracy = one_epoch(dqn, ce, test_loader, old_params['mode'])
        # report test accuracy of r-ranker
        logger.info("Test loss: %g - test accuracy:\n%s" % (loss, json.dumps(accuracy, indent=2)))

    # predict q values: use huber loss (or MSE)
    else:
        loss, _ = one_episode(dqn, dqn_target, old_params['gamma'],
                              huber, test_loader, old_params['mode'])
        logger.info("Test loss: %g" % loss)


    logger.info("Finished testing. Time elapsed: %g seconds" % (
        time.time() - start_time
    ))


    #####################################################################
    # PREDICT VALUES FOR EACH TEST EXAMPLE & GROUP BY CONV_ID + CONTEXT #
    #####################################################################
    start_time = time.time()
    logger.info("")
    logger.info("Testing model one example at a time & generating report.json")

    chats = {}  # map from chat ID to list of (context - candidate) pairs
    '''
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
    '''
    ###
    # BUILD REPORT: regroup each convo+context and present their candidate, score, predictions
    ###
    for item, data in enumerate(test_conv):
        article, idx = test_conv.ids[item]
        entry = test_conv.json_data[article][idx]
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

        #logger.info("item: %s" % item)
        #logger.info("")
        articles, n_sents, l_sents, \
            contexts, n_turns, l_turns, \
            candidates, n_tokens, \
            custom_encs, rewards, \
            next_states, n_next_turns, l_next_turns, \
            next_candidates, n_next_candidates, l_next_candidates, \
            next_custom_encs = data
        #logger.info("articles: %s" % articles)
        #logger.info("n_sents: %s" % n_sents)
        #logger.info("l_sents: %s" % l_sents)
        #logger.info("contexts: %s" % contexts)
        #logger.info("n_turns: %s" % n_turns)
        #logger.info("l_turns: %s" % l_turns)
        #logger.info("candidates: %s" % candidates)
        #logger.info("n_tokens: %s" % n_tokens)
        #logger.info("custom_encs: %s" % custom_encs)
        #logger.info("rewards: %s" % rewards)
        #logger.info("next_states: %s" % next_states)
        #logger.info("n_next_turns: %s" % n_next_turns)
        #logger.info("l_next_turns: %s" % l_next_turns)
        #logger.info("next_candidates: %s" % next_candidates)
        #logger.info("n_next_candidates: %s" % n_next_candidates)
        #logger.info("l_next_candidates: %s" % l_next_candidates)
        #logger.info("next_custom_encs: %s" % next_custom_encs)

        # batch size is 1 so need to encapsulate data in tuple
        data = ([articles, n_sents, l_sents,
                 contexts, n_turns, l_turns,
                 candidates, n_tokens,
                 custom_encs, rewards,
                 next_states, n_next_turns, l_next_turns,
                 next_candidates, n_next_candidates, l_next_candidates,
                 next_custom_encs
        ],)

        # put data in batches & create masks
        logger.setLevel(logging.WARNING)  # ignore warning of no next state since batch size is 1 anyway
        data = collate_fn(data)  # batch size is 1!!
        logger.setLevel(logging.DEBUG)    # reset logger level

        # Classification of candidate responses:
        if old_params['predict_rewards']:
            # with a simple MLP
            if old_params['mode'] == 'mlp':
                _, _, _, custom_encs, rewards, _, _ = data
                # articles : tuple of list of sentences. each sentence is a Tensor. ~(1, n_sents, n_tokens)
                # contexts : tuple of list of turns. each turn is a Tensor. ~(1, n_turns, n_tokens)
                # candidates : tuple of Tensors. ~(1, n_tokens)
                # custom_encs : Tensor ~(1, enc)
                # rewards : Tensor ~(1,)

                # reward not scaled for classification
                assert rewards[0] == entry['reward']

                # Convert Tensors to Variables
                custom_encs = to_var(custom_encs)  # ~(1, enc)
                # Forward pass: predict rewards
                predictions = dqn(custom_encs)  # ~(1, 2)
                predictions = F.softmax(predictions[0], dim=0)

                try:
                    chats[entry['chat_id']][idx]['predictions'].append(predictions.data[1])
                except KeyError:
                    chats[entry['chat_id']][idx]['predictions'] = [predictions.data[1]]

            # with RNNs and MLP
            else:
                _, articles_tensors, n_sents, l_sents,\
                    _, contexts_tensors, n_turns, l_turns,\
                    candidates_tensors, n_tokens, custom_encs, rewards,\
                    _, _, _, _, _, _, _, _ = data
                # articles : tuple of list of sentences. each sentence is a Tensor. ~(1, n_sents, n_tokens)
                # articles_tensor : Tensor ~(1 x n_sents, max_len)
                # n_sents : Tensor ~(1)
                # l_sents : Tensor ~(1 x n_sents)
                # contexts : tuple of list of turns. each turn is a Tensor. ~(1, n_turns, n_tokens)
                # contexts_tensor : Tensor ~(1 x n_turns, max_len)
                # n_turns : Tensor ~(1)
                # l_turns : Tensir ~(1 x n_turns)
                # candidates_tensor : Tensor ~(1, max_len)
                # n_tokens : Tensor ~(1)
                # custom_encs : Tensor ~(1, enc)
                # rewards : Tensor ~(1,)

                # Convert Tensors to Variables
                articles_tensors = to_var(articles_tensors)  # ~(1 x n_sents, max_len)
                n_sents = to_var(n_sents)  # ~(1)
                l_sents = to_var(l_sents)  # ~(1 x n_sents)
                contexts_tensors = to_var(contexts_tensors)  # ~(1 x n_turns, max_len)
                n_turns = to_var(n_turns)  # ~(1)
                l_turns = to_var(l_turns)  # ~(1 x n_turns)
                candidates_tensors = to_var(candidates_tensors)  # ~(1, max_len)
                n_tokens = to_var(n_tokens)  # ~(1)
                custom_encs = to_var(custom_encs)  # ~(1, enc)

                # reward not scaled for classification
                assert rewards[0] == entry['reward']

                # Forward pass: predict q-values
                predictions = dqn(
                    articles_tensors, n_sents, l_sents,
                    contexts_tensors, n_turns, l_turns,
                    candidates_tensors, n_tokens,
                    custom_encs
                )  # ~ (1, 2)
                predictions = F.softmax(predictions[0], dim=0)

                try:
                    chats[entry['chat_id']][idx]['predictions'].append(predictions.data[1])
                except KeyError:
                    chats[entry['chat_id']][idx]['predictions'] = [predictions.data[1]]

        # Prediction of Q-value
        else:
            # with a simple MLP
            if old_params['mode'] == 'mlp':
                _, _, _, custom_encs, rewards, _, _ = data
                # articles : tuple of list of sentences. each sentence is a Tensor. ~(1, n_sents, n_tokens)
                # contexts : tuple of list of turns. each turn is a Tensor. ~(1, n_turns, n_tokens)
                # candidates : tuple of Tensors. ~(1, n_tokens)
                # custom_encs : Tensor ~(1, enc)
                # rewards : Tensor ~(1,)
                # non_final_mask : boolean list ~(1)
                # non_final_next_custom_encs : tuple of list of Tensors: ~(1-, n_actions, enc)

                # reward is scaled for q-prediction
                # comparing floats with '==' is not a good idea, use .isclose() instead.
                assert np.isclose(rewards[0], test_conv._rescale_rewards(entry['reward'], entry['quality']))

                # Convert Tensors to Variables
                custom_encs = to_var(custom_encs)  # ~(1, enc)
                # Forward pass: predict current state-action value
                q_values = dqn(custom_encs)  # ~(1)

                try:
                    chats[entry['chat_id']][idx]['predictions'].append(q_values.data[0][0])
                except KeyError:
                    chats[entry['chat_id']][idx]['predictions'] = [q_values.data[0][0]]

            # with RNNs and MLP
            else:
                _, articles_tensors, n_sents, l_sents,\
                    _, contexts_tensors, n_turns, l_turns,\
                    candidates_tensors, n_tokens,\
                    custom_encs, rewards,\
                    _, _, _, _, _, _, _, _ = data
                # articles : tuple of list of sentences. each sentence is a Tensor. ~(1, n_sents, n_tokens)
                # articles_tensor : Tensor ~(1 x n_sents, max_len)
                # n_sents : Tensor ~(1)
                # l_sents : Tensor ~(1 x n_sents)
                # contexts : tuple of list of turns. each turn is a Tensor. ~(1, n_turns, n_tokens)
                # contexts_tensor : Tensor ~(1 x n_turns, max_len)
                # n_turns : Tensor ~(1)
                # l_turns : Tensir ~(1 x n_turns)
                # candidates_tensor : Tensor ~(1, max_len)
                # n_tokens : Tensor ~(1)
                # custom_encs : Tensor ~(1, enc)
                # rewards : Tensor ~(1,)
                # non_final_mask : boolean list ~(1)
                # ...
                # non_final_next_custom_encs : tuple of list of Tensors: ~(1-, n_actions, enc)

                # reward is scaled for q-prediction
                # comparing floats with '==' is not a good idea, use .isclose() instead.
                assert np.isclose(rewards[0], test_conv._rescale_rewards(entry['reward'], entry['quality']))

                # Convert Tensors to Variables
                # state:
                articles_tensors_t = to_var(articles_tensors)  # ~(1 x n_sentences, max_len)
                n_sents_t = to_var(n_sents)  # ~(1)
                l_sents_t = to_var(l_sents)  # ~(1 x n_sentences)
                contexts_tensors_t = to_var(contexts_tensors)  # ~(1 x n_turns, max_len)
                n_turns_t = to_var(n_turns)  # ~(1)
                l_turns_t = to_var(l_turns)  # ~(1 x n_turns)
                # action:
                candidates_tensors_t = to_var(candidates_tensors)  # ~(1, max_len)
                n_tokens_t = to_var(n_tokens)  # ~(1)
                custom_encs_t = to_var(custom_encs)  # ~(1, enc)

                # Forward pass: predict current state-action value
                q_values = dqn(
                    articles_tensors_t, n_sents_t, l_sents_t,
                    contexts_tensors_t, n_turns_t, l_turns_t,
                    candidates_tensors_t, n_tokens_t,
                    custom_encs_t
                )  # ~(1,)

                try:
                    chats[entry['chat_id']][idx]['predictions'].append(q_values.data[0][0])
                except KeyError:
                    chats[entry['chat_id']][idx]['predictions'] = [q_values.data[0][0]]

    logger.info("Finished 1-by-1 testing. Time elapsed: %g seconds" % (
        time.time() - start_time
    ))

    ###
    # Measuring accuracy
    ###
    logger.info("")
    logger.info("Measuring accuracy at predicting best candidate...")
    correct = 0.0
    total = 0.0

    for chat_id, contexts in chats.iteritems():
        for c in contexts:
            '''
            {
                'context'    : [list of strings],
                'candidates' : [list of strings],
                'rewards'    : [list of ints],
                'predictions': [list of floats]
            },
            '''
            # sum of all candidate rewards is at most 1
            assert np.sum(c['rewards']) <= 1

            if np.argmax(c['rewards']) == np.argmax(c['predictions']):
                correct += 1
            total += 1

    logger.info("Predicted like human behavior: %d / %d = %g" % (
        correct, total, correct / total
    ))

    ###
    # Saving report
    ###
    logger.info("")
    logger.info("Saving report...")
    with open("%s_report.json" % args.model_prefix, 'wb') as f:
        json.dump(chats, f, indent=2)
    logger.info("done.")


    ###
    # TODO
    ###
    # analyse report:
    # - measure recall@1, @2, ..., @5
    # - plot proportion of #of possible candidate: #of2, #of3, ..., #of9
    # - measure #of_candidate avg
    # - plot recall@1 as a function of context length
    # - plot proportion of context length: #of2, #of3, ..., #of10
    # - measure context length avg


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected. Got %s' % v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_prefix", type=str, help="Path to model prefix to test")
    parser.add_argument("-g",  "--gpu", type=int, default=0, help="GPU number to use")
    parser.add_argument("-v", "--verbose", type=str2bool, default='no', help="Be verbose")

    args = parser.parse_args()
    logger.info("")
    logger.info("%s" % args)
    logger.info("")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu

    # To handle multiple workers on data loader!!
    torch.multiprocessing.set_sharing_strategy('file_system')

    main()

