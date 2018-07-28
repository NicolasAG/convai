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
import spacy
import copy
import json
import time
import sys
import os
import re

logger = logging.getLogger(__name__)

nlp = spacy.load('en')

config = {
    "wh_words": ['who', 'where', 'when', 'why', 'what', 'how', 'whos', 'wheres', 'whens', 'whys', 'whats', 'hows']
}

# list of generic words to detect if user is bored
generic_words_list = []
with open('../data/generic_list.txt') as fp:
    for line in fp:
        generic_words_list.append(line.strip())
generic_words_list = set(generic_words_list)  # remove duplicates


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
        #data_f = "./data/q_ranker_amt_data_1524939554.0.pkl"
        data_f = "./data/q_ranker_amt_data_1527760664.38.pkl"

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


def one_epoch(dqn, loss, data_loader, old_params):
    """
    Performs one epoch over the specified data
    :param dqn: q-network to perform forward pass
    :param loss: cross entropy loss function
    :param data_loader: data iterator
    :param old_params: parameters of previous model
    :return: average loss & accuracy dictionary: acc, TP, TN, FP, FN
    """
    if old_params['mode'] == 'mlp':
        epoch_loss, epoch_accuracy, nb_batches = _one_mlp_epoch(dqn, loss, data_loader)
    else:
        epoch_loss, epoch_accuracy, nb_batches = _one_rnn_epoch(dqn, loss, data_loader,
                                                                old_params.get('use_custom_encs', True))

    epoch_loss /= nb_batches
    epoch_accuracy['acc'] /= nb_batches
    epoch_accuracy['F1'] /= nb_batches

    epoch_accuracy['TPR'] /= nb_batches  # true positive rate = recall = sensitivity = hit rate
    epoch_accuracy['TNR'] /= nb_batches  # true negative rate = specificity
    epoch_accuracy['PPV'] /= nb_batches  # positive predictive value = precision
    epoch_accuracy['NPV'] /= nb_batches  # negative predictive value

    epoch_accuracy['FNR'] /= nb_batches  # false negative rate = miss rate
    epoch_accuracy['FPR'] /= nb_batches  # false positive rate = fall out
    epoch_accuracy['FDR'] /= nb_batches  # false discovery rate
    epoch_accuracy['FOR'] /= nb_batches  # false omission rate

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
    epoch_accuracy = {'acc': 0., 'F1':0.,
                      'TPR': 0., 'TNR': 0., 'PPV': 0., 'NPV': 0.,
                      'FNR': 0., 'FPR': 0., 'FDR': 0., 'FOR': 0.}
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
        epoch_accuracy['F1'] += (2*tmp_tp / (2*tmp_tp + tmp_fp + tmp_fn))
        if tmp_tn + tmp_fp > 0:
            epoch_accuracy['TNR'] += (tmp_tn / (tmp_tn + tmp_fp))  # true negative rate = specificity
            epoch_accuracy['FPR'] += (tmp_fp / (tmp_tn + tmp_fp))  # false positive rate = fall-out
        if tmp_fn + tmp_tp > 0:
            epoch_accuracy['FNR'] += (tmp_fn / (tmp_fn + tmp_tp))  # false negative rate = miss rate
            epoch_accuracy['TPR'] += (tmp_tp / (tmp_fn + tmp_tp))  # true positive rate = sensitivity = recall = hit rate
        if tmp_tp + tmp_fp > 0:
            epoch_accuracy['PPV'] += (tmp_tp / (tmp_tp + tmp_fp)) # positive predictive value = precision
            epoch_accuracy['FDR'] += (tmp_fp / (tmp_tp + tmp_fp)) # false discovery rate
        if tmp_tn + tmp_fn > 0:
            epoch_accuracy['NPV'] += (tmp_tn / (tmp_tn + tmp_fn)) # negative predicting value
            epoch_accuracy['FOR'] += (tmp_fn / (tmp_tn + tmp_fn)) # false omission rate

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

def _one_rnn_epoch(dqn, loss, data_loader, use_custom_encs):
    """
    Performs one epoch over the specified data
    :param dqn: q-network to perform forward pass
    :param loss: cross entropy loss function
    :param data_loader: data iterator
    :param use_custom_encs: if using custom_encs
    :return: epoch loss & accuracy
    """
    epoch_loss = 0.0
    epoch_accuracy = {'acc': 0., 'F1': 0.,
                      'TPR': 0., 'TNR': 0., 'PPV': 0., 'NPV': 0.,
                      'FNR': 0., 'FPR': 0., 'FDR': 0., 'FOR': 0.}
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

        # custom encoding dimension
        if not use_custom_encs:
            custom_encs = None

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
        epoch_accuracy['F1'] += (2 * tmp_tp / (2 * tmp_tp + tmp_fp + tmp_fn))
        if tmp_tn + tmp_fp > 0:
            epoch_accuracy['TNR'] += (tmp_tn / (tmp_tn + tmp_fp))  # true negative rate = specificity
            epoch_accuracy['FPR'] += (tmp_fp / (tmp_tn + tmp_fp))  # false positive rate = fall-out
        if tmp_fn + tmp_tp > 0:
            epoch_accuracy['FNR'] += (tmp_fn / (tmp_fn + tmp_tp))  # false negative rate = miss rate
            epoch_accuracy['TPR'] += (tmp_tp / (tmp_fn + tmp_tp))  # true positive rate = sensitivity = recall = hit rate
        if tmp_tp + tmp_fp > 0:
            epoch_accuracy['PPV'] += (tmp_tp / (tmp_tp + tmp_fp))  # positive predictive value = precision
            epoch_accuracy['FDR'] += (tmp_fp / (tmp_tp + tmp_fp))  # false discovery rate
        if tmp_tn + tmp_fn > 0:
            epoch_accuracy['NPV'] += (tmp_tn / (tmp_tn + tmp_fn))  # negative predicting value
            epoch_accuracy['FOR'] += (tmp_fn / (tmp_tn + tmp_fn))  # false omission rate

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


def one_episode(dqn, dqn_target, huber, data_loader, old_params):
    """
    Performs one episode over the specified data
    :param dqn: q-network to perform forward pass
    :param dqn_target: target network to compute dqn target
    :param huber: huber loss function
    :param data_loader: data iterator
    :param old_params: parameters of previous model
    :return: average huber loss, number of batch seen
    """
    if old_params['mode'] == 'mlp':
        epoch_huber_loss, nb_batches = _one_mlp_episode(dqn, dqn_target, old_params['gamma'], huber, data_loader)
    else:
        epoch_huber_loss, nb_batches = _one_rnn_episode(dqn, dqn_target, old_params['gamma'], huber, data_loader,
                                                        old_params.get('use_custom_encs', True))

    epoch_huber_loss /= nb_batches

    return epoch_huber_loss, nb_batches

def _one_mlp_episode(dqn, dqn_target, gamma, huber, data_loader):
    """
    Performs one epoch over the specified data
    :param dqn: q-network to perform forward pass
    :param dqn_target: target network to compute DQN target
    :param gamma: discount factor
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

def _one_rnn_episode(dqn, dqn_target, gamma, huber, data_loader, use_custom_encs):
    """
    Performs one epoch over the specified data
    :param dqn: q-network to perform forward pass
    :param dqn_target: target network to compute DQN target
    :param gamma: discount factor
    :param huber: huber loss function
    :param data_loader: data iterator
    :param use_custom_encs: if using custom_encs
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

        # custom encoding dimension
        if not use_custom_encs:
            custom_encs_t = None

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
                if use_custom_encs:
                    # grab candidates custom encodings
                    tmp_custom_encs = to_var(
                        to_tensor(non_final_next_custom_encs[idx]),
                        volatile=True
                    )  # ~(n_actions, enc)
                else:
                    tmp_custom_encs = None

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
                if use_custom_encs:
                    max_actions['custom_enc'].append(tmp_custom_encs.data[max_idx].view(-1))

            if use_custom_encs:
                next_q_values = dqn_target(
                    articles_tensors_tp1, n_sents_tp1, l_sents_tp1,
                    contexts_tensors_tp1, n_turns_tp1, l_turns_tp1,
                    to_var(torch.stack(max_actions['candidate']), volatile=True),
                    to_var(torch.cat(max_actions['n_tokens']), volatile=True),
                    to_var(torch.stack(max_actions['custom_enc']), volatile=True)
                )  # ~(bs)
            else:
                next_q_values = dqn_target(
                    articles_tensors_tp1, n_sents_tp1, l_sents_tp1,
                    contexts_tensors_tp1, n_turns_tp1, l_turns_tp1,
                    to_var(torch.stack(max_actions['candidate']), volatile=True),
                    to_var(torch.cat(max_actions['n_tokens']), volatile=True),
                    None
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


def plot_timings(timings, old_params):
    train_losses = timings['train_losses']
    train_accs = [e['acc'] for e in timings['train_accurs']]
    valid_losses = timings['valid_losses']
    valid_accs = [e['acc'] for e in timings['valid_accurs']]
    if len(train_accs) > 0 and 'F1' in timings['train_accurs'][0]:
        train_f1 = [e['F1'] for e in timings['train_accurs']]
        valid_f1 = [e['F1'] for e in timings['valid_accurs']]
    else:
        train_f1 = None
        valid_f1 = None


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

    if train_f1 and valid_f1:
        plt.plot(range(1, len(train_f1) + 1), train_f1, 'b-', label='train')
        plt.plot(range(1, len(valid_f1) + 1), valid_f1, 'r-', label='valid')
        plt.title("Training and Validation F1 score over time")
        plt.xlabel("epochs")
        plt.ylabel("F1")
        plt.legend(loc='best')
        plt.savefig("%s_f1.png" % args.model_prefix)
        plt.close()

        best_valid_f1_idx = np.argmax(valid_f1)
        logger.info("best valid f1: %g achieved at epoch %d" %
                    (valid_f1[best_valid_f1_idx], best_valid_f1_idx))
        logger.info("training f1 at this epoch: %g" % train_f1[best_valid_f1_idx])


def test_individually(dqn, test_conv, old_params):
    """
    Build a map from chat ID to list of (context, candidates, rewards, predictions) list
    :param test_conv: raw json data of test set
    :return: a map of this form:
    chat_id : [
        {
            'article'      : string,
            'context'      : [list of strings],
            'candidates'   : [list of strings],
            'model_names'  : [list of strings],
            'ranker_confs' : [list of floats],
            'ranker_scores': [list of floats],
            'rewards'      : [list of ints],
            'predictions'  : [list of floats],
            'rule-based'   : [list of floats]
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
    chats = {}

    for item, data in enumerate(test_conv):
        article, idx = test_conv.ids[item]
        entry = test_conv.json_data[article][idx]
        '''
        'chat_id': <string>,
        'state': <list of strings> ie: context,
        'action': {
            'candidate': <string>  ie: candidate,
            'custom_enc': <list of float>,
            'model_name': <string of model name>,
            'conf': <string of float between 0.0 & 1.0 or int 0 when not evaluated>,
            'score': <string of float between 0.0 & 5.0 or int -1 when not evaluated>,
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
                    assert c['article'] is not None
                    c['candidates'].append(entry['action']['candidate'])
                    c['model_names'].append(entry['action']['model_name'])
                    c['ranker_confs'].append(entry['action']['conf'])
                    c['ranker_scores'].append(entry['action']['score'])
                    c['rewards'].append(entry['reward'])
                    #c['predictions'].append(0.0)  # placeholder for now
                    c['rule-based'].append(0.0)  # placeholder for now
                    idx = i
                    break

            # if context doesn't exists, add it as a new one with this candidate response
            if idx == -1:
                chats[entry['chat_id']].append({
                    'article': article,
                    'context': entry['state'],
                    'candidates': [entry['action']['candidate']],
                    'model_names': [entry['action']['model_name']],
                    'ranker_confs': [entry['action']['conf']],
                    'ranker_scores': [entry['action']['score']],
                    'rewards': [entry['reward']],
                    #'predictions': [0.0],  # placeholder for now
                    'rule-based': [0.0]  # placeholder for now
                })

        # if chat doesn't exists, add a new one
        else:
            chats[entry['chat_id']] = [{
                'article': article,
                'context': entry['state'],
                'candidates': [entry['action']['candidate']],
                'model_names': [entry['action']['model_name']],
                'ranker_confs': [entry['action']['conf']],
                'ranker_scores': [entry['action']['score']],
                'rewards': [entry['reward']],
                #'predictions': [0.0],  # placeholder for now
                'rule-based': [0.0]  # placeholder for now
            }]
            idx = 0  # it's the first and last chat so idx=0 or idx=-1 are both ok

        # logger.info("item: %s" % item)
        # logger.info("")
        articles, n_sents, l_sents, \
        contexts, n_turns, l_turns, \
        candidates, n_tokens, \
        custom_encs, rewards, \
        next_states, n_next_turns, l_next_turns, \
        next_candidates, n_next_candidates, l_next_candidates, \
        next_custom_encs = data
        # logger.info("articles: %s" % articles)
        # logger.info("n_sents: %s" % n_sents)
        # logger.info("l_sents: %s" % l_sents)
        # logger.info("contexts: %s" % contexts)
        # logger.info("n_turns: %s" % n_turns)
        # logger.info("l_turns: %s" % l_turns)
        # logger.info("candidates: %s" % candidates)
        # logger.info("n_tokens: %s" % n_tokens)
        # logger.info("custom_encs: %s" % custom_encs)
        # logger.info("rewards: %s" % rewards)
        # logger.info("next_states: %s" % next_states)
        # logger.info("n_next_turns: %s" % n_next_turns)
        # logger.info("l_next_turns: %s" % l_next_turns)
        # logger.info("next_candidates: %s" % next_candidates)
        # logger.info("n_next_candidates: %s" % n_next_candidates)
        # logger.info("l_next_candidates: %s" % l_next_candidates)
        # logger.info("next_custom_encs: %s" % next_custom_encs)

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
        l = logging.getLogger('q_data_loader')
        l.setLevel(logging.ERROR)  # ignore warning of no next state since batch size is 1 anyway
        data = collate_fn(data)  # batch size is 1!!
        l.setLevel(logging.DEBUG)  # reset logger level

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
                _, articles_tensors, n_sents, l_sents, \
                _, contexts_tensors, n_turns, l_turns, \
                candidates_tensors, n_tokens, custom_encs, rewards, \
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

                # custom encoding dimension
                if not old_params.get('use_custom_encs', True):
                    custom_encs = None

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
                _, articles_tensors, n_sents, l_sents, \
                _, contexts_tensors, n_turns, l_turns, \
                candidates_tensors, n_tokens, \
                custom_encs, rewards, \
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

                # custom encoding dimension
                if not old_params.get('use_custom_encs', True):
                    custom_encs_t = None

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

    return chats


def get_rulebased_predictions(chats):

    def topic_match(text):
        # catch responses of the style "what is this article about"
        question_match_1 = ".*what\\s*(is|'?s|does)?\\s?(this|it|the)?\\s?(article)?\\s?(talks?)?\\s?(about)\\s*(\\?)*"
        # catch also responses of the style : "what do you think of this article"
        question_match_2 = ".*what\\sdo\\syou\\sthink\\s(of|about)\\s(this|it|the)?\\s?(article)\\s*\\?*"
        return re.match(question_match_1, text, re.IGNORECASE) or re.match(question_match_2, text, re.IGNORECASE)

    def has_wh_words(nt_words):
        # check if the text contains wh words
        for word in nt_words:
            if word in set(config['wh_words']):
                return True
        return False

    def rank_candidate_responses(c):
        # array containing tuple of (model_name, rank_score)
        consider_models = []
        dont_consider_models = []

        always_consider = ['hred-reddit', 'hred-twitter', 'alicebot']

        for idx, response in enumerate(c['candidates']):
            model = c['model_names'][idx]
            conf = float(c['predictions'][idx])
            score = float(c['ranker_scores'][idx])

            if conf > 0.75 and model in always_consider:
                consider_models.append((model, conf * score))

            if conf < 0.25 and conf != 0.0 and model != 'fact_gen':
                # keep fact generator model for failure handling case
                dont_consider_models.append((model, conf * score))

        consider = None
        dont_consider = None
        if len(consider_models) > 0:
            consider_models = sorted(consider_models, key=lambda x: x[1], reverse=True)
            consider = consider_models[0][0]

        elif len(dont_consider_models) > 0:
            dont_consider = dont_consider_models

        return consider, dont_consider

    def was_it_bored(context):
        bored_cnt = 0
        for utt in context:
            # nouns of the user message
            ntext = nlp(unicode(utt))
            nt_words = [p.lemma_ for p in ntext]

            # check if user said only generic words:
            generic_turn = True
            for word in nt_words:
                if word not in generic_words_list:
                    generic_turn = False
                    break

            # if text contains 2 words or less, add 1 to the bored count
            # also consider the case when the user says only generic things
            if len(utt.strip().split()) <= 2 or generic_turn:
                bored_cnt += 1

        return bored_cnt > 0 and bored_cnt % 2 == 0

    for chat_id, list_of_contexts in chats.iteritems():
        for c in list_of_contexts:
            ''' {
                'article'      : string,
                'context'      : [list of strings],
                'candidates'   : [list of strings],
                'model_names'  : [list of strings],
                'ranker_confs' : [list of string numbers],
                'ranker_scores': [list of string numbers],
                'rewards'      : [list of ints],
                'predictions': [list of floats],
                'rule-based': [list of floats]
            } '''
            assert len(c['candidates']) == len(c['model_names']) == len(c['ranker_confs']) == len(c['ranker_scores']) \
                   == len(c['rewards']) == len(c['predictions']) == len(c['rule-based'])

            # step1: if context length = 1, chose randomly between NQG and EntitySentence
            if len(c['context']) == 1:
                assert len(c['candidates']) == 2, "number of candidates (%d) != 2" % len(c['candidates'])
                assert c['model_names'][0] in ['candidate_question', 'nqg'], "model name (%s) unknown" % c['model_names'][0]
                assert c['model_names'][1] in ['candidate_question', 'nqg'], "model name (%s) unknown" % c['model_names'][1]

                choice = np.random.choice([0, 1])
                c['rule-based'][choice] = 1.0
                continue  # go to next interaction

            # step2: if query falls under dumb questions, respond appropriately
            # if candidate responses contains dumb_qa then there must have been a match
            if 'dumb_qa' in c['model_names']:
                idx = c['model_names'].index('dumb_qa')
                assert c['ranker_confs'][idx] == '0', "ranker conf (%s) != '0'" % c['ranker_confs'][idx]
                assert c['ranker_scores'][idx] == '-1', "ranker score (%s) != '-1'" % c['ranker_scores'][idx]
                assert len(c['candidates']) == 9, "number of candidates (%d) != 9" % len(c['candidates'])

                c['rule-based'][idx] = 1.0
                continue  # go to next interaction

            # step3: if query falls under topic request, respond with the article topic
            if topic_match(c['context'][-1]) and 'topic_model' in c['model_names']:
                idx = c['model_names'].index('topic_model')
                assert c['ranker_confs'][idx] == '0', "ranker confs (%s) != '0'" % c['ranker_confs'][idx]
                assert c['ranker_scores'][idx] == '-1', "ranker scores (%s) != '-1'" % c['ranker_scores'][idx]

                c['rule-based'][idx] = 1.0
                continue  # go to next interaction

            # nouns of the previous user message
            ntext = nlp(unicode(c['context'][-1]))
            nt_words = [p.lemma_ for p in ntext]

            # step4: if query is a question, try to reply with DrQA
            if has_wh_words(nt_words) or ('which' in c['context'][-1] and '?' in c['context'][-1]):
                # nouns from the article
                article_nlp = nlp(unicode(c['article']))
                article_nouns = [p.lemma_ for p in article_nlp if p.pos_ in ['NOUN', 'PROPN']]

                common = list(set(article_nouns).intersection(set(nt_words)))

                # if there is a common noun between question and article select DrQA
                if len(common) > 0 and 'drqa' in c['model_names']:
                    idx = c['model_names'].index('drqa')
                    c['rule-based'][idx] = 1.0
                    continue  # go to next interaction

            ###
            # Ranker based selection
            ###

            # get ignored models
            idx_to_ignore = []
            best_model, dont_consider = rank_candidate_responses(c)
            if dont_consider and len(dont_consider) > 0:
                for model, _ in dont_consider:
                    idx_to_ignore.append(c['model_names'].index(model))

            # Reduce confidence of CAND_QA
            #if 'candidate_question' in c['model_names']:
            #    idx = c['model_names'].index('candidate_question')
            #    c['ranker_confs'][idx] = str(float(c['ranker_confs'][idx]) / 2)

            # step5: Bored model selection
            if was_it_bored(c['context']):
                # list of available models to use if bored
                bored_models = ['nqg', 'fact_gen', 'candidate_question']
                bored_idx = [c['model_names'].index(m) for m in c['model_names'] if m in bored_models]
                bored_idx = [idx for idx in bored_idx if idx not in idx_to_ignore]
                # if user is bored, change the topic by asking a question
                if len(bored_idx) > 0:
                    # assign model selection probability based on estimator confidence
                    for idx in bored_idx:
                        if c['predictions'][idx] == 0.0:
                            c['rule-based'][idx] = min([p for p in c['predictions'] if p > 0.0]) / 10.
                        else:
                            c['rule-based'][idx] = c['predictions'][idx]

                    c['rule-based'] = c['rule-based'] / np.sum(c['rule-based'])
                    c['rule-based'] = c['rule-based'].tolist()
                    continue  # go to next interaction

            # step6: If not bored, then select from best model
            if best_model:
                idx = c['model_names'].index(best_model)
                c['rule-based'][idx] = 1.0
                continue  # go to next interaction

            # step7: Sample from the other models based on confidence probability
            # randomly decide a model to query to get a response:
            models = ['hred-reddit', 'hred-twitter', 'alicebot']
            # if the user asked a question, also consider DrQA
            if has_wh_words(nt_words) or ('which' in c['context'][-1] and '?' in c['context'][-1]):
                models.append('drqa')
            # if the user didn't ask a question, also consider models that ask questions: nqg, and cand_qa
            else:
                models.extend(['nqg', 'candidate_question'])

            available_idx = [c['model_names'].index(m) for m in c['model_names'] if m in models]
            available_idx = [idx for idx in available_idx if idx not in idx_to_ignore]
            if len(available_idx) > 0:
                # assign model selection probability based on estimator confidence
                for idx in available_idx:
                    if c['predictions'][idx] == 0.0:
                        c['rule-based'][idx] = min([p for p in c['predictions'] if p > 0.0]) / 10.
                    else:
                        c['rule-based'][idx] = c['predictions'][idx]

                c['rule-based'] = c['rule-based'] / np.sum(c['rule-based'])
                c['rule-based'] = c['rule-based'].tolist()
                continue  # go to next interaction

            # last step: if still no response, then just send a random fact
            if 'fact_gen' in c['model_names']:
                idx = c['model_names'].index('fact_gen')
                c['rule-based'][idx] = 1.0
                continue  # go to next interaction

            # at this stage only one response should have been selected!
            logger.warning("THIS SHOULD NEVER BE PRINTED... :3")

    return chats


def get_recall(chats, policy):
    assert policy in ['RuleBased', 'Sample', 'Argmax']
    recalls = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    total = 0.0

    for chat_id, contexts in chats.iteritems():
        for c in contexts:
            '''
            {
                'article'      : string,
                'context'      : [list of strings],
                'candidates'   : [list of strings],
                'model_names'  : [list of strings],
                'ranker_confs' : [list of string numbers],
                'ranker_scores': [list of string numbers],
                'rewards'      : [list of ints],
                'predictions': [list of floats],
                'rule-based' : [list of floats]
            },
            '''
            # sum of all candidate rewards is at most 1
            assert np.sum(c['rewards']) <= 1

            # sum of all predictions with ruleBased policy is exactly 1
            assert np.isclose(np.sum(c['rule-based']), 1.0), "SUM(rule-based) = %g != 1.0" % np.sum(c['rule-based'])

            # Get sorted idx of candidate scores
            if policy == 'RuleBased':
                # select response according to rule based
                _, sorted_idx = torch.Tensor(c['rule-based']).sort(descending=True)
            elif policy == 'Argmax':
                # select the max response
                _, sorted_idx = torch.Tensor(c['predictions']).sort(descending=True)
            elif policy == 'Sample':
                # sample response according to probability score
                probas = np.exp(c['predictions']) / np.sum(np.exp(c['predictions']), axis=0)
                sorted_idx = np.random.choice(range(len(probas)),
                                              size=len(probas),
                                              replace=False,
                                              p=probas)
            else:
                raise ValueError("Unkown policy: %s" % policy)

            # idx of chosen candidate
            idx = np.argmax(c['rewards'])

            # Recall @ 1
            if idx in sorted_idx[:1]:
                for i in range(9):
                    recalls[i] += 1
            # Recal @ 2
            elif idx in sorted_idx[:2]:
                for i in range(1, 9):
                    recalls[i] += 1
            # Recal @ 3
            elif idx in sorted_idx[:3]:
                for i in range(2, 9):
                    recalls[i] += 1
            # Recal @ 4
            elif idx in sorted_idx[:4]:
                for i in range(3, 9):
                    recalls[i] += 1
            # Recal @ 5
            elif idx in sorted_idx[:5]:
                for i in range(4, 9):
                    recalls[i] += 1
            # Recal @ 6
            elif idx in sorted_idx[:6]:
                for i in range(5, 9):
                    recalls[i] += 1
            # Recal @ 7
            elif idx in sorted_idx[:7]:
                for i in range(6, 9):
                    recalls[i] += 1
            # Recal @ 8
            elif idx in sorted_idx[:8]:
                for i in range(7, 9):
                    recalls[i] += 1
            # Recal @ 9
            elif idx in sorted_idx[:9]:
                for i in range(8, 9):
                    recalls[i] += 1

            total += 1

    return recalls, total


def get_proportions(chats):
    candidate_counts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    total_counts = 0.0
    context_lengths = [0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0]
    total_lengths = 0.0
    long_contexts = 0.0
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
            assert len(c['candidates']) == len(c['rewards']) == len(c['predictions'])

            candidate_counts[len(c['candidates'])-1] += 1
            total_counts += len(c['candidates'])

            try:
                context_lengths[len(c['context'])-1] += 1
            except IndexError:
                # logger.info("long context: %d" % len(c['context']))
                long_contexts += 1
            total_lengths += len(c['context'])

            total += 1

    return (candidate_counts, total_counts), (context_lengths, long_contexts, total_lengths), total


def recallat1_contextlen(chats, policy):
    assert policy in ['RuleBased', 'Sample', 'Argmax']
    recalls = [0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    totals = [0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for chat_id, contexts in chats.iteritems():
        for c in contexts:
            '''
            {
                'article'      : string,
                'context'      : [list of strings],
                'candidates'   : [list of strings],
                'model_names'  : [list of strings],
                'ranker_confs' : [list of string numbers],
                'ranker_scores': [list of string numbers],
                'rewards'      : [list of ints],
                'predictions': [list of floats],
                'rule-based' : [list of floats]
            },
            '''
            # Get sorted idx of candidate scores
            if policy == 'RuleBased':
                # select response according to rule based
                _, sorted_idx = torch.Tensor(c['rule-based']).sort(descending=True)
            elif policy == 'Argmax':
                # select the max response
                _, sorted_idx = torch.Tensor(c['predictions']).sort(descending=True)
            elif policy == 'Sample':
                # sample response according to probability score
                probas = np.exp(c['predictions']) / np.sum(np.exp(c['predictions']), axis=0)
                sorted_idx = np.random.choice(range(len(probas)),
                                              size=len(probas),
                                              replace=False,
                                              p=probas)
            else:
                raise ValueError("Unkown policy: %s" % policy)

            # idx of chosen candidate
            idx = np.argmax(c['rewards'])

            # Recall @ 1
            if idx in sorted_idx[:1]:
                try:
                    recalls[(len(c['context']) - 1) / 2] += 1.
                except IndexError:
                    recalls[-1] += 1.

            try:
                totals[(len(c['context']) - 1) / 2] += 1.
            except IndexError:
                totals[-1] += 1.

    return recalls, totals


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
        # custom encoding dimension
        if not old_params.get('use_custom_encs', True):
            custom_hs = 0
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

    # restore old parameters
    dqn.load_state_dict(torch.load("%s_dqn.pt" % args.model_prefix))
    dqn_target.load_state_dict(torch.load("%s_dqn.pt" % args.model_prefix))

    # put in inference mode (Dropout OFF)
    dqn.eval()
    dqn_target.eval()

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

    plot_timings(timings, old_params)
    logger.info("done.")

    #######################
    # Start Testing
    #######################
    start_time = time.time()
    logger.info("")
    logger.info("Testing model in batches...")

    # predict rewards: use cross-entropy loss
    if old_params['predict_rewards']:
        loss, accuracy = one_epoch(dqn, ce, test_loader, old_params)
        # report test accuracy of r-ranker
        logger.info("Test loss: %g - test accuracy:\n%s" % (loss, json.dumps(accuracy, indent=2)))

    # predict q values: use huber loss (or MSE)
    else:
        loss, _ = one_episode(dqn, dqn_target, huber, test_loader, old_params)
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

    ###
    # BUILD REPORT: regroup each convo+context and present their candidate, score, predictions
    ###
    chats = test_individually(dqn, test_conv, old_params)  # get DQN scores
    logger.info("Now simulating the old chatbot decision policy...")
    chats = get_rulebased_predictions(chats)               # get rule-based scores

    logger.info("Finished 1-by-1 testing. Time elapsed: %g seconds" % (
        time.time() - start_time
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
    # Measuring recall
    ###
    logger.info("")
    logger.info("Measuring recall at predicting best candidate...")

    # Get recall for different selection strategies
    recalls_rulebased, total_rulebased = get_recall(chats, 'RuleBased')
    recalls_argmax, total_argmax = get_recall(chats, 'Argmax')
    recalls_sample, total_sample = get_recall(chats, 'Sample')

    logger.info("Predicted like human behavior with rulebased selection:")
    for idx, r in enumerate(recalls_rulebased):
        logger.info("- recall@%d: %d / %d = %g" % (
            idx + 1, r, total_rulebased, r / total_rulebased
        ))
    logger.info("Predicted like human behavior with argmax selection:")
    for idx, r in enumerate(recalls_argmax):
        logger.info("- recall@%d: %d / %d = %g" % (
            idx + 1, r, total_argmax, r / total_argmax
        ))
    logger.info("Predicted like human behavior with sampled selection:")
    for idx, r in enumerate(recalls_sample):
        logger.info("- recall@%d: %d / %d = %g" % (
            idx + 1, r, total_sample, r / total_sample
        ))

    barwidth = 0.3

    # - plot recalls: r@1, r@2, ..., r@9
    recalls_rulebased = np.array(recalls_rulebased) / total_rulebased
    recalls_argmax = np.array(recalls_argmax) / total_argmax
    recalls_sample = np.array(recalls_sample) / total_sample
    plt.bar(range(len(recalls_rulebased)),
            recalls_rulebased,
            color='r', width=barwidth,
            label='Rule-Based')
    plt.bar([x + barwidth for x in range(len(recalls_argmax))],
            recalls_argmax,
            color='b', width=barwidth,
            label='Argmax')
    plt.bar([x + 2 * barwidth for x in range(len(recalls_sample))],
            recalls_sample,
            color='g', width=barwidth,
            label='Sampled')
    plt.legend(loc='best')
    plt.title("Recall@k measure")
    plt.xlabel("k")
    plt.xticks(
        [r + barwidth for r in range(len(recalls_rulebased))],
        range(1, len(recalls_rulebased) + 1)
    )
    plt.ylabel("recall")
    plt.savefig("%s_recalls.png" % args.model_prefix)
    plt.close()


    ###
    # Compute recall@1 for each context length
    ###
    logger.info("")
    logger.info("Measuring recall@1 for each context length...")

    # Get recall for different selection strategies
    recalls_rulebased, totals_rulebased = recallat1_contextlen(chats, 'RuleBased')
    recalls_argmax, totals_argmax = recallat1_contextlen(chats, 'Argmax')
    recalls_sample, totals_sample = recallat1_contextlen(chats, 'Sample')

    logger.info("Predicted like human behavior with rulebased selection:")
    for c_len, r in enumerate(recalls_rulebased):
        logger.info("- recall@1 for context of size %d: %d / %d = %s" % (
            2*c_len + 1, r, totals_rulebased[c_len],
            (r / totals_rulebased[c_len] if totals_rulebased[c_len] > 0 else "inf")
        ))
    logger.info("Predicted like human behavior with argmax selection:")
    for c_len, r in enumerate(recalls_argmax):
        logger.info("- recall@1 for context of size %d: %d / %d = %s" % (
            2*c_len + 1, r, totals_argmax[c_len], (r / totals_argmax[c_len] if totals_argmax[c_len] > 0 else "inf")
        ))
    logger.info("Predicted like human behavior with sampled selection:")
    for c_len, r in enumerate(recalls_sample):
        logger.info("- recall@1 for context of size %d: %d / %d = %s" % (
            2*c_len + 1, r, totals_sample[c_len], (r / totals_sample[c_len] if totals_sample[c_len] > 0 else "inf")
        ))

    # - plot recalls:
    recalls_rulebased = np.array(recalls_rulebased) / np.array(totals_rulebased)
    recalls_argmax = np.array(recalls_argmax) / np.array(totals_argmax)
    recalls_sample = np.array(recalls_sample) / np.array(totals_sample)
    plt.bar(range(len(recalls_rulebased)),
            recalls_rulebased,
            color='r', width=barwidth,
            label='Rule-Based')
    plt.bar([x + barwidth for x in range(len(recalls_argmax))],
            recalls_argmax,
            color='b', width=barwidth,
            label='Argmax')
    plt.bar([x + 2 * barwidth for x in range(len(recalls_sample))],
            recalls_sample,
            color='g', width=barwidth,
            label='Sampled')
    plt.legend(loc='best')
    plt.title("Recall@1 for each context length")
    plt.xlabel("#of messages")
    plt.xticks(
        [r + barwidth for r in range(len(recalls_rulebased))],
        range(1, 2*len(recalls_rulebased), 2) + ['>']
    )
    plt.ylabel("recall")
    plt.savefig("%s_recall1_c-len.png" % args.model_prefix)
    plt.close()

    ###
    # Get stats: number of candidates, context lengths, etc...
    # Do it ONCE for all model: these are stats of the test set itself
    ###
    '''
    logger.info("")
    logger.info("Counting number of candidate responses & context lengths...")

    (candidate_counts, total_counts), (context_lengths, long_contexts, total_lengths), total = get_proportions(chats)

    logger.info("Number of candidate responses:")
    for length, count in enumerate(candidate_counts):
        logger.info("- #of examples with %d candidates available: %d / %d = %g" % (
            length + 1, count, total, count / total
        ))
    # - measure #of_candidate avg
    logger.info("Average number of candidates: %g" % (total_counts / total))
    # - plot proportion of #of possible candidate: #of2, #of3, ..., #of9
    candidate_counts = np.array(candidate_counts) / total
    plt.bar(range(len(candidate_counts)), candidate_counts, tick_label=range(1, len(candidate_counts)+1))
    plt.title("Number of candidate responses available")
    plt.xlabel("#of candidates")
    plt.ylabel("Proportion")
    plt.savefig("./models/q_estimator/test_candidates_proportion.png")
    plt.close()

    logger.info("")

    logger.info("Length of context:")
    for length, count in enumerate(context_lengths):
        logger.info("- #of examples with %d messages in context: %d / %d = %g" % (
            length + 1, count, total, count / total
        ))
    logger.info("- #of examples with more messages in context: %d / %d = %g" % (
        long_contexts, total, long_contexts / total
    ))
    # - measure context length avg
    logger.info("Average context size: %g" % (total_lengths / total))
    # - plot proportion of context length: #of2, #of3, ..., #of10
    context_lengths.append(long_contexts)
    context_lengths = np.array(context_lengths) / total
    plt.bar(
        range(len(context_lengths)),
        context_lengths,
        tick_label=range(1, len(context_lengths))+['>']
    )
    plt.title("Context lengths")
    plt.xlabel("#of messages")
    plt.ylabel("Proportion")
    plt.savefig("./models/q_estimator/test_contextlength_proportion.png")
    plt.close()
    '''

    logger.info("done.")



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

