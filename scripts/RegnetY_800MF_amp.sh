#!/bin/bash
python3 train_apex.py --data_path /dataset \
                --epochs 100 \
                --batch_size 128 \
                --bottleneck_ratio 1 \
                --group_width 16 \
                --initial_width 56 \
                --slope 39 \
                --quantized_param 2.4 \
                --network_depth 14 \
                --stride 2 \
                --se_ratio 4