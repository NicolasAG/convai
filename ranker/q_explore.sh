#!/usr/bin/env bash

MODE=$1

optimizers=( 'adam' 'sgd' 'rmsprop' 'adadelta' )
learningrates=( 0.01 0.001 0.0001 )
activations=( 'sigmoid' 'prelu' )
dropout=( 0.2 0.4 0.6 0.8 )
exp=1

if [ "$MODE" = 'mlp_r' ] ; then
    echo "explore mlp R"
    for op in "${optimizers[@]}"
    do
        for ac in "${activations[@]}"
        do
            for drop in "${dropout[@]}"
            do
                for lr in "${learningrates[@]}"
                do
                    echo "Train MLP R ranker with" $op "(" $lr ") and" $ac "activations and" $drop "dropout -- stop on F1"
                    hyperdash run -n "train PT mlp_f1 R ranker++ exp$exp" python q_train.py \
                        --gpu 1 \
                        --data_f ./data/q_ranker_amt_data++_1525301962.86.json \
                        --vocab_f ./data/q_ranker_amt_vocab_1525301962.86.pkl \
                        --mode mlp \
                        --predict_rewards yes \
                        --optimizer $op \
                        --learning_rate $lr \
                        --mlp_activation $ac \
                        --mlp_dropout $drop \
                        --model_name "SmallR/Small_R-Network[exp$exp]os+F1"
                    exp=$((exp+1))
                done
            done
        done
    done

elif [ "$MODE" = 'mlp_q' ] ; then
    echo "explore mlp Q"
    for op in "${optimizers[@]}"
    do
        for ac in "${activations[@]}"
        do
            for drop in "${dropout[@]}"
            do
                for lr in "${learningrates[@]}"
                do
                    echo "Train MLP Q ranker with" $op "(" $lr ") and" $ac "activations and" $drop "dropout"
                    #echo "SmallQ/Small_Q-Network[exp$exp]os"
                    hyperdash run -n "train PT mlp Q ranker++ exp$exp" python q_train.py \
                        --data_f ./data/q_ranker_amt_data++_1525301962.86.json \
                        --vocab_f ./data/q_ranker_amt_vocab_1525301962.86.pkl \
                        --mode mlp \
                        --predict_rewards no \
                        --optimizer $op \
                        --learning_rate $lr \
                        --mlp_activation $ac \
                        --mlp_dropout $drop \
                        --model_name "SmallQ/Small_Q-Network[exp$exp]os"

                    hyperdash run -n "train PT mlp Q ranker exp$exp" python q_train.py \
                        --data_f ./data/q_ranker_amt_data_1524939554.0.json \
                        --vocab_f ./data/q_ranker_amt_vocab_1524939554.0.pkl \
                        --mode mlp \
                        --predict_rewards no \
                        --optimizer $op \
                        --learning_rate $lr \
                        --mlp_activation $ac \
                        --mlp_dropout $drop \
                        --model_name "SmallQ/Small_Q-Network[exp$exp]"

                    exp=$((exp+1))
                done
            done
        done
    done

elif [ "$MODE" = 'rnn+mlp_r' ] ; then
    echo "explore rnn+mlp R"
    for op in "${optimizers[@]}"
    do
        for ac in "${activations[@]}"
        do
            for drop in "${dropout[@]}"
            do
                for lr in "${learningrates[@]}"
                do
                    echo "Train RNN+MLP R ranker with" $op "(" $lr ") and" $ac "activations and" $drop "dropout"
                    #echo "DeepR/Deep_R-Network[exp$exp]os"
                    hyperdash run -n "train PT rnn+mlp R ranker++ exp$exp" python q_train.py \
                        --data_f ./data/q_ranker_amt_data++_1525301962.86.json \
                        --vocab_f ./data/q_ranker_amt_vocab_1525301962.86.pkl \
                        --mode rnn+mlp \
                        --predict_rewards yes \
                        --optimizer $op \
                        --learning_rate $lr \
                        --mlp_activation $ac \
                        --mlp_dropout $drop \
                        --fix_embeddings no \
                        --sentence_hs 300 \
                        --utterance_hs 300 \
                        --sentence_dropout $drop \
                        --article_dropout $drop \
                        --utterance_dropout $drop \
                        --context_dropout $drop \
                        --model_name "DeepR/Deep_R-Network[exp$exp]os"
                    exp=$((exp+1))
                done
            done
        done
    done

elif [ "$MODE" = 'rnn+mlp_q' ] ; then
    echo "explore rnn+mlp Q"
    for op in "${optimizers[@]}"
    do
        for ac in "${activations[@]}"
        do
            for drop in "${dropout[@]}"
            do
                for lr in "${learningrates[@]}"
                do
                    echo "Train RNN+MLP Q ranker with" $op "(" $lr ") and" $ac "activations and" $drop "dropout"
                    #echo "DeepQ/Deep_Q-Network[exp$exp]os"
                    hyperdash run -n "train PT rnn+mlp Q ranker++ exp$exp" python q_train.py \
                        --data_f ./data/q_ranker_amt_data++_1525301962.86.json \
                        --vocab_f ./data/q_ranker_amt_vocab_1525301962.86.pkl \
                        --mode rnn+mlp \
                        --predict_rewards no \
                        --optimizer $op \
                        --learning_rate $lr \
                        --mlp_activation $ac \
                        --mlp_dropout $drop \
                        --fix_embeddings no \
                        --sentence_hs 300 \
                        --utterance_hs 300 \
                        --sentence_dropout $drop \
                        --article_dropout $drop \
                        --utterance_dropout $drop \
                        --context_dropout $drop \
                        --model_name "DeepQ/Deep_Q-Network[exp$exp]os"

                    hyperdash run -n "train PT rnn+mlp Q ranker exp$exp" python q_train.py \
                        --data_f ./data/q_ranker_amt_data_1524939554.0.json \
                        --vocab_f ./data/q_ranker_amt_vocab_1524939554.0.pkl \
                        --mode rnn+mlp \
                        --predict_rewards no \
                        --optimizer $op \
                        --learning_rate $lr \
                        --mlp_activation $ac \
                        --mlp_dropout $drop \
                        --fix_embeddings no \
                        --sentence_hs 300 \
                        --utterance_hs 300 \
                        --sentence_dropout $drop \
                        --article_dropout $drop \
                        --utterance_dropout $drop \
                        --context_dropout $drop \
                        --model_name "DeepQ/Deep_Q-Network[exp$exp]"
                    exp=$((exp+1))
                done
            done
        done
    done

else
    echo "UNKNOWN MODE:" $MODE
fi
