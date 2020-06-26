#!/bin/bash
python3 train_kfold.py --data_path /dataset \
                --epochs 3 \
                --n_fold 5 \
                --batch_size 32 \
                --bottleneck_ratio 1 \
                --group_width 16 \
                --initial_width 56 \
                --slope 39 \
                --quantized_param 2.4 \
                --network_depth 14 \
                --stride 2 \
                --se_ratio 4
