#!/bin/bash

#python train_with_edge.py --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml --gpus 0,1,2,3,4,5,6,7 --name stablesr_edge_T6_DIV2K_2005_10_04
python train_with_edge_fixed.py --config configs/stableSRNew/v2-finetune_text_T_512_edge_fixed.yaml --gpus 0,1,2,3,4,5,6,7 --name stablesr_edge_fixed