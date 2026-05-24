#!/bin/bash
# v20 resume(12500) → v21 재시작 → v22 → v23 → v24 순차 학습
# v20: high (19~25, resume from 12500)
# v21: low (0~8, 재시작)
# v22: mid-low (10~13)
# v23: mid (15~19)
# v24: full LoRA (0~26, 27L)

set -e
cd /home/billy/26kp/openpi_upstream_clean

LOG_DIR="/tmp"

echo "=============================="
echo "[CHAIN] v20 resume from 12500 (vision LoRA 19~25, high only)"
echo "=============================="
uv run scripts/train.py pi05_e6_v20_lora --exp-name e6_2cam_lora_v20 --resume 2>&1 | tee "$LOG_DIR/train_v20.log"

echo "=============================="
echo "[CHAIN] v20 완료 → v21 재시작 (vision LoRA 0~8, low only)"
echo "=============================="
uv run scripts/train.py pi05_e6_v21_lora --exp-name e6_2cam_lora_v21 2>&1 | tee "$LOG_DIR/train_v21.log"

echo "=============================="
echo "[CHAIN] v21 완료 → v22 학습 시작 (vision LoRA 10~13, mid-low)"
echo "=============================="
uv run scripts/train.py pi05_e6_v22_lora --exp-name e6_2cam_lora_v22 2>&1 | tee "$LOG_DIR/train_v22.log"

echo "=============================="
echo "[CHAIN] v22 완료 → v23 학습 시작 (vision LoRA 15~19)"
echo "=============================="
uv run scripts/train.py pi05_e6_v23_lora --exp-name e6_2cam_lora_v23 2>&1 | tee "$LOG_DIR/train_v23.log"

echo "=============================="
echo "[CHAIN] v23 완료 → v24 학습 시작 (vision LoRA 0~26, full 27L)"
echo "=============================="
uv run scripts/train.py pi05_e6_v24_lora --exp-name e6_2cam_lora_v24 2>&1 | tee "$LOG_DIR/train_v24.log"

echo "=============================="
echo "[CHAIN] v24 완료 → v25 학습 시작 (vision LoRA 15~19, Mid/Late 경계)"
echo "=============================="
uv run scripts/train.py pi05_e6_v25_lora --exp-name e6_2cam_lora_v25 2>&1 | tee "$LOG_DIR/train_v25.log"

echo "=============================="
echo "[CHAIN] v25 완료 → v26 학습 시작 (vision LoRA 22~25, Late 상위 26제외)"
echo "=============================="
uv run scripts/train.py pi05_e6_v26_lora --exp-name e6_2cam_lora_v26 2>&1 | tee "$LOG_DIR/train_v26.log"

echo "=============================="
echo "[CHAIN] v20~v26 전체 완료"
echo "=============================="
