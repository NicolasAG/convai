#!/usr/bin/env bash

##################
# Train Q-RANKER #
##################

### 1) MLP to predict immediate R ###
hyperdash run -n "train PT mlp R ranker" python q_train.py \
                ./data/q_ranker_amt_data_1524939554.0.json \
                ./data/q_ranker_amt_vocab_1524939554.0.pkl \
    mlp \
    --predict_rewards yes \
    --verbose no \
    --gpu 0 \
    --optimizer adam \
    --learning_rate 0.001 \
    --patience 20 \
    --batch_size 128 \
    --mlp_activation swish \
    --mlp_dropout 0.1

### 2) MLP to predict Q-values ###
hyperdash run -n "train PT mlp Q ranker" python q_train.py \
                ./data/q_ranker_amt_data_1524939554.0.json \
                ./data/q_ranker_amt_vocab_1524939554.0.pkl \
    mlp \
    --predict_rewards no \
    --verbose no \
    --gpu 0 \
    --optimizer adam \
    --learning_rate 0.001 \
    --gamma 0.99 \
    --patience 20 \
    --batch_size 128 \
    --update_frequence 2000 \
    --mlp_activation swish \
    --mlp_dropout 0.1

### 3) RNN to predict immediate R ###
# TOFIX: --> OVERFITS!!! done in less than 1 epoch! :o
hyperdash run -n "train PT rnn R ranker" python q_train.py \
                ./data/q_ranker_amt_data_1524939554.0.json \
                ./data/q_ranker_amt_vocab_1524939554.0.pkl \
    rnn+mlp \
    --predict_rewards yes \
    --verbose no \
    --gpu 0 \
    --optimizer adam \
    --learning_rate 0.001 \
    --patience 20 \
    --batch_size 128 \
    --fix_embeddings yes \
    --rnn_gate gru \
    --sentence_hs 100 \
    --article_hs 300 \
    --utterance_hs 100 \
    --context_hs 300 \
    --mlp_activation swish \
    --mlp_dropout 0.1

### 4) RNN to predict Q-values ###
hyperdash run -n "train PT rnn Q ranker" python q_train.py \
                ./data/q_ranker_amt_data_1524939554.0.json \
                ./data/q_ranker_amt_vocab_1524939554.0.pkl \
    rnn+mlp \
    --predict_rewards no \
    --verbose no \
    --gpu 0 \
    --optimizer adam \
    --learning_rate 0.001 \
    --gamma 0.99 \
    --patience 20 \
    --batch_size 128 \
    --update_frequence 10000 \
    --fix_embeddings yes \
    --rnn_gate gru \
    --sentence_hs 100 \
    --article_hs 300 \
    --utterance_hs 100 \
    --context_hs 300 \
    --mlp_activation swish \
    --mlp_dropout 0.1


