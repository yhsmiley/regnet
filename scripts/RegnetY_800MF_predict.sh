#!/bin/bash
python3 predict.py --test_data_path /dataset/test \
                --model_path /regnet/trained_models/best_amp_checkpoint.pth.tar \
                --batch_size 10 \
                --bottleneck_ratio 1 \
                --group_width 16 \
                --initial_width 56 \
                --slope 39 \
                --quantized_param 2.4 \
                --network_depth 14 \
                --stride 2 \
                --se_ratio 4