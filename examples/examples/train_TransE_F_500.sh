#!/bin/bash
cd .. 
source set_env.sh
python3 run.py \
            --dataset FB237_base \
            --model TransE \
            --rank 500 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adagrad \
            --max_epochs 1500 \
            --patience 10 \
            --valid 5 \
            --batch_size 1000 \
            --neg_sample_size -1 \
            --init_size 0.001 \
            --learning_rate 0.001 \
            --gamma 0.0 
cd examples/
