#!/bin/bash
python3 train.py --data_path /dataset \
                --epochs 100 \
                --batch_size 32 \
                --bottleneck_ratio 1 \
                --group_width 24 \
                --initial_width 48 \
                --slope 20.71 \
                --quantized_param 2.65 \
                --network_depth 27 \
                --stride 2 \
                --se_ratio 4
