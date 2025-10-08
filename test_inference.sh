#!/bin/bash

# Test inference with edge processing
# 测试边缘处理推理

cd /root/dp/StableSR_Edge_v2

# Activate environment
source /root/miniconda/bin/activate sr_edge

# Run inference with a single test image
python scripts/sr_val_ddpm_text_T_vqganfin_old_edge.py \
    --use_edge_processing \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt logs/2025-10-07T02-28-22_stablesr_edge_8_channels/checkpoints/epoch=000030.ckpt \
    --vqgan_ckpt models/ldm/stable-diffusion-v1/epoch=000011.ckpt \
    --init-img inputs/user_upload \
    --outdir outputs/test_edge_fix \
    --ddpm_steps 200 \
    --dec_w 0.5 \
    --colorfix_type adain \
    --n_samples 1 \
    --seed 42

echo "Inference complete. Check outputs/test_edge_fix/"




