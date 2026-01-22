#!/usr/bin/env bash
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/mnt/T5_26T/workspace/StableSR/hf_cache
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
/workspace1/StableSR/env/bin/python main.py \
   --train \
  --base configs/memc/v2-finetune_text_T_512_memc.yaml \
   --gpus 4,5,6,7, \
   --name memc \
   --scale_lr False

