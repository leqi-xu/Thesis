#!/bin/bash
cd .. 
source set_env.sh
python3 run.py \
            --dataset WN18RR_base \
            --model TransE \
            --rank 500 \
            --regularizer N3 \
            --reg 0.1 \
            --optimizer Adam \
            --max_epochs 500 \
            --patience 10 \
            --valid 5 \
            --batch_size 100 \
            --neg_sample_size -1 \
            --init_size 0.001 \
            --learning_rate 0.001 \
            --gamma 0.0 \
            --bias none \
            --dtype single \
            --train1
cd examples/
