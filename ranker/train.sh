#!/usr/bin/env bash

########################################
# Train estimator for immediate reward #
########################################

#                ./data/supervised_ranker_amt_data_1524858322.44.json \

hyperdash run -n "train TF mlp supervised ranker" python train.py \
                ./data/supervised_ranker_db_voted_data_1520108818.78.json \
                ./data/supervised_ranker_round1_voted_data_1520108828.2.json \
    short_term \
    --verbose \
    --gpu 0 \
    --batch_size 128 \
    --patience 20 \
    --hidden_sizes 789 789 394 \
    --activation relu \
    --dropout_rate 0.7 \
    --optimizer rmsprop \
    --learning_rate 0.001

###
# Train estimator for long term reward based on previously trained short term estimator
###
# python train.py ./data/full_data_db_1510012482.99.json ./data/full_data_round1_1510012496.02.json \
#     long_term \
#     --gpu 3 \
#     --patience 20 \
#     --previous_model ./models/short_term/0.640059/1510403841.53_Estimator_
#     --previous_model ./models/short_term/0.639659/1510335852.62_Estimator_
#     --previous_model ./models/short_term/0.639792/1510236120.6_Estimator_
#     --previous_model ./models/short_term/0.641391/1510248853.21_Estimator_


