#!/bin/bash
python3 train.py --data_path /dataset \
                --epochs 100 \
                --batch_size 256 \
                --bottleneck_ratio 1 \
                --group_width 8 \
                --initial_width 24 \
                --slope 36 \
                --quantized_param 2.5 \
                --network_depth 13 \
                --stride 2 \
                --se_ratio 4 \
                --apex \
                --fixres