#!/bin/bash
python3 train_kfold.py --data_path /dataset \
                --epochs 10 \
                --n_fold 5 \
                --batch_size 32 \
                --bottleneck_ratio 1 \
                --group_width 24 \
                --initial_width 80 \
                --slope 42.63 \
                --quantized_param 2.66 \
                --network_depth 21 \
                --stride 2 \
                --se_ratio 4
