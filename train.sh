#!/bin/bash
python /root/pd/StableSR/main.py --train --base /root/pd/StableSR/configs/stableSRNew/v2-finetune_text_T_512_weiql_0930.yaml --gpus 0,1,2,3,4,5,6,7 --name weiql_t800_0930 --scale_lr False