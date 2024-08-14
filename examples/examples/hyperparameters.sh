#!/bin/bash
declare -a reg_values=("0" "0.1")
declare -a batch_size_values=("100" "500" "1000")
declare -a neg_sample_values=("50" "500")
declare -a learning_rate_values=("0.001" "0.01" "0.1")

for reg in "${reg_values[@]}"; do
    for batch_size in "${batch_size_values[@]}"; do
        for neg_sample in "${neg_sample_values[@]}"; do
            for learning_rate in "${learning_rate_values[@]}"; do
                python3 /workspace/KGEmb-master/hyperparameters.py \
                    --dataset WN18RR_base \
                    --model TransE \
                    --rank 500 \
                    --regularizer N3 \
                    --reg $reg \
                    --optimizer Adam \
                    --max_epochs 100 \
                    --patience 10 \
                    --valid 5 \
                    --batch_size $batch_size \
                    --neg_sample $neg_sample \
                    --init_size 0.001 \
                    --learning_rate $learning_rate \
                    --gamma 0.0 \
                    --bias none \
                    --dtype single
            done
        done
    done
done