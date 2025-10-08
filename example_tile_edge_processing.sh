#!/bin/bash

# Tile-Based Edge Processing 使用示例
# 本脚本展示如何使用支持tile处理的edge增强超分辨率

echo "=============================================="
echo "  Tile-Based Edge Super-Resolution Examples"
echo "=============================================="

# ===== 示例1: 基础使用 - 小图片（不需要tile） =====
echo ""
echo "示例1: 处理小图片（≤512x512）"
echo "----------------------------------------------"
python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile_edge.py \
    --init-img ./128x128_valid_LR \
    --gt-img ./512x512_valid_HR \
    --outdir ./results/example1_small \
    --use_edge_processing \
    --vqgantile_size 1280 \
    --vqgantile_stride 1000 \
    --tile_overlap 32 \
    --input_size 512 \
    --upscale 4.0 \
    --ddpm_steps 200 \
    --colorfix_type adain \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
    --vqgan_ckpt models/ldm/stable-diffusion-v1/epoch=000011.ckpt \
    --seed 42

# ===== 示例2: 大图片处理 - 使用tile =====
echo ""
echo "示例2: 处理大图片（>1280x1280）使用tile"
echo "----------------------------------------------"
# 假设输入是256x256的LR图片，需要上采样到2048x2048（8x）
python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile_edge.py \
    --init-img ./large_lr_images \
    --gt-img ./large_hr_images \
    --outdir ./results/example2_large_tile \
    --use_edge_processing \
    --vqgantile_size 1280 \
    --vqgantile_stride 1000 \
    --tile_overlap 32 \
    --input_size 512 \
    --upscale 8.0 \
    --ddpm_steps 200 \
    --colorfix_type adain \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
    --vqgan_ckpt models/ldm/stable-diffusion-v1/epoch=000011.ckpt \
    --seed 42

# ===== 示例3: 超大图片 - 保守设置 =====
echo ""
echo "示例3: 超大图片处理（4K+）- 保守设置"
echo "----------------------------------------------"
python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile_edge.py \
    --init-img ./4k_lr_images \
    --gt-img ./4k_hr_images \
    --outdir ./results/example3_4k \
    --use_edge_processing \
    --vqgantile_size 1536 \
    --vqgantile_stride 1200 \
    --tile_overlap 48 \
    --input_size 512 \
    --upscale 4.0 \
    --ddpm_steps 200 \
    --colorfix_type wavelet \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
    --vqgan_ckpt models/ldm/stable-diffusion-v1/epoch=000011.ckpt \
    --seed 42

# ===== 示例4: 不使用GT - 从LR生成edge map =====
echo ""
echo "示例4: 不使用GT图片（从LR生成edge map）"
echo "----------------------------------------------"
python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile_edge.py \
    --init-img ./test_lr_images \
    --outdir ./results/example4_no_gt \
    --use_edge_processing \
    --vqgantile_size 1280 \
    --vqgantile_stride 1000 \
    --tile_overlap 32 \
    --input_size 512 \
    --upscale 4.0 \
    --ddpm_steps 200 \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
    --vqgan_ckpt models/ldm/stable-diffusion-v1/epoch=000011.ckpt \
    --seed 42

# ===== 示例5: 快速模式 - 减少步数 =====
echo ""
echo "示例5: 快速模式（降低ddpm_steps）"
echo "----------------------------------------------"
python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile_edge.py \
    --init-img ./quick_test \
    --gt-img ./quick_test_hr \
    --outdir ./results/example5_fast \
    --use_edge_processing \
    --vqgantile_size 1280 \
    --vqgantile_stride 1000 \
    --tile_overlap 16 \
    --input_size 512 \
    --upscale 4.0 \
    --ddpm_steps 100 \
    --colorfix_type nofix \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
    --vqgan_ckpt models/ldm/stable-diffusion-v1/epoch=000011.ckpt \
    --seed 42

# ===== 示例6: 内存受限 - 小tile尺寸 =====
echo ""
echo "示例6: 内存受限环境（减小tile尺寸）"
echo "----------------------------------------------"
python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile_edge.py \
    --init-img ./memory_limited_test \
    --gt-img ./memory_limited_test_hr \
    --outdir ./results/example6_memory_limited \
    --use_edge_processing \
    --vqgantile_size 1024 \
    --vqgantile_stride 768 \
    --tile_overlap 16 \
    --input_size 512 \
    --upscale 4.0 \
    --ddpm_steps 150 \
    --config configs/stableSRNew/v2-finetune_text_T_512_edge.yaml \
    --ckpt models/ldm/stable-diffusion-v1/model.ckpt \
    --vqgan_ckpt models/ldm/stable-diffusion-v1/epoch=000011.ckpt \
    --seed 42

echo ""
echo "=============================================="
echo "  所有示例执行完成！"
echo "  结果保存在 ./results/ 目录"
echo "=============================================="

# ===== 参数说明 =====
cat << EOF

参数说明：
-----------
--init-img          输入LR图片目录
--gt-img            GT图片目录（可选，用于生成更好的edge map）
--outdir            输出目录
--use_edge_processing  启用edge增强处理
--vqgantile_size    VQGAN tile尺寸（像素），越大质量越好但内存占用越多
--vqgantile_stride  VQGAN tile步长（像素），越小重叠越多，质量越好但速度越慢
--tile_overlap      Diffusion latent空间的tile重叠
--input_size        基础输入尺寸
--upscale           上采样倍数
--ddpm_steps        扩散步数，越多质量越好但速度越慢
--colorfix_type     颜色修正: adain (最好) | wavelet | nofix
--seed              随机种子，保证可重复性

推荐配置：
-----------
小图 (≤512):       vqgantile_size=1280, stride=1000, overlap=32, steps=200
中图 (512-1280):   vqgantile_size=1280, stride=1000, overlap=32, steps=200
大图 (1280-2048):  vqgantile_size=1280, stride=1000, overlap=32, steps=200
超大图 (>2048):    vqgantile_size=1536, stride=1200, overlap=48, steps=200

快速模式:          减少steps到100-150
高质量模式:        增加overlap到48-64，减小stride
内存受限:          减小vqgantile_size到1024

EOF

