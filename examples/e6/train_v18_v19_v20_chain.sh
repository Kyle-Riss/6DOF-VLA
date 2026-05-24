#!/bin/bash
# v18 → v19 → v20 → v21 순차 학습 (vision LoRA ablation)
# v18: full (0~25), v19: mid (14~18), v20: high (19~25), v21: low (0~8)
# 각 20k steps (~3.8 epoch, 42495 frames / batch8)

set -e
cd /home/billy/26kp/openpi_upstream_clean

LOG_DIR="/tmp"

echo "=============================="
echo "[CHAIN] v18 학습 시작 (vision LoRA 0~25, full)"
echo "=============================="
uv run scripts/train.py pi05_e6_v18_lora --exp-name e6_2cam_lora_v18 2>&1 | tee "$LOG_DIR/train_v18.log"

echo "=============================="
echo "[CHAIN] v18 완료 → v19 학습 시작 (vision LoRA 14~18, mid only)"
echo "=============================="
uv run scripts/train.py pi05_e6_v19_lora --exp-name e6_2cam_lora_v19 2>&1 | tee "$LOG_DIR/train_v19.log"

echo "=============================="
echo "[CHAIN] v19 완료 → v20 학습 시작 (vision LoRA 19~25, high only)"
echo "=============================="
uv run scripts/train.py pi05_e6_v20_lora --exp-name e6_2cam_lora_v20 2>&1 | tee "$LOG_DIR/train_v20.log"

echo "=============================="
echo "[CHAIN] v20 완료 → v21 학습 시작 (vision LoRA 0~8, low only)"
echo "=============================="
uv run scripts/train.py pi05_e6_v21_lora --exp-name e6_2cam_lora_v21 2>&1 | tee "$LOG_DIR/train_v21.log"

echo "=============================="
echo "[CHAIN] v18/v19/v20/v21 전체 완료"
echo "=============================="
