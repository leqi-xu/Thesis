#!/bin/bash
cd .. 
source set_env.sh
python3 run1.py \
            --dataset YAGO3-10_curr \
            --model ComplEx \
            --rank 500 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adagrad \
            --max_epochs 3000 \
            --patience 5 \
            --valid 5 \
            --batch_size 4000 \
            --neg_sample_size -1 \
            --init_size 0.001 \
            --learning_rate 0.005 \
            --gamma 0.0 \
            --bias learn \
            --dtype double \
            --multi_c  
cd examples/
