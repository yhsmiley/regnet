#!/bin/bash
python3 predict.py --test_data_path /dataset/test \
                --model_path /regnet/trained_models/best_amp_checkpoint.pth.tar \
                --bottleneck_ratio 1 \
                --group_width 8 \
                --initial_width 24 \
                --slope 36 \
                --quantized_param 2.5 \
                --network_depth 13 \
                --stride 2 \
                --se_ratio 4