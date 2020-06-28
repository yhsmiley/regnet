#!/bin/bash
python3 train_kfold.py --data_path /dataset \
                --epochs 10 \
                --n_fold 5 \
                --batch_size 32 \
                --bottleneck_ratio 1 \
                --group_width 24 \
                --initial_width 48 \
                --slope 20.71 \
                --quantized_param 2.65 \
                --network_depth 27 \
                --stride 2 \
                --se_ratio 4
