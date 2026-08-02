#!/bin/bash
# v21, v23, v24, v25, v26 → safetensor 변환 후 새 볼륨2 이동
set -e

BASE="/home/billy/26kp/openpi_upstream_clean"
DEST="/media/billy/새 볼륨3/e6_checkpoints"
LOG="/tmp/convert_v21_v26.log"
cd "$BASE"

declare -A VERSIONS=(
    [v21]="19999:pi05_e6_v21_lora"
    [v23]="19999:pi05_e6_v23_lora"
    [v24]="19999:pi05_e6_v24_lora"
    [v25]="19999:pi05_e6_v25_lora"
    [v26]="7500:pi05_e6_v26_lora"
)

echo "============================== $(date)" | tee -a "$LOG"
echo "[INFO] 변환 시작 (CPU 모드, CUDA 비활성)" | tee -a "$LOG"

for VERSION in v21 v23 v24 v25 v26; do
    IFS=':' read -r STEP CONFIG <<< "${VERSIONS[$VERSION]}"
    CKPT_DIR="checkpoints/pi05_e6_${VERSION}_lora/e6_2cam_lora_${VERSION}/${STEP}"
    OUT_DIR="checkpoints/pi05_e6_${VERSION}_lora/e6_2cam_lora_${VERSION}/${STEP}_pytorch_lora_merged"

    echo "" | tee -a "$LOG"
    echo "============================== $(date)" | tee -a "$LOG"
    echo "[${VERSION}] step=${STEP}, config=${CONFIG}" | tee -a "$LOG"

    if [ ! -d "$CKPT_DIR" ]; then
        echo "[${VERSION}] ❌ 체크포인트 없음: $CKPT_DIR" | tee -a "$LOG"
        continue
    fi

    if [ -d "$OUT_DIR" ]; then
        echo "[${VERSION}] ⚠️ 이미 변환됨: $OUT_DIR (skip)" | tee -a "$LOG"
    else
        echo "[${VERSION}] 변환 중..." | tee -a "$LOG"
        JAX_PLATFORMS=cpu uv run examples/convert_jax_to_pytorch_lora_merged.py \
            --checkpoint-dir "$CKPT_DIR" \
            --config-name "$CONFIG" \
            --output-path "$OUT_DIR" \
            2>&1 | tee -a "$LOG"
        if [ -f "$OUT_DIR/model.safetensors" ]; then
            echo "[${VERSION}] ✅ 변환 완료: $OUT_DIR" | tee -a "$LOG"
        else
            echo "[${VERSION}] ❌ 변환 실패 (model.safetensors 없음)" | tee -a "$LOG"
            continue
        fi
    fi

    # 새 볼륨2 이동
    DEST_DIR="$DEST/e6_${VERSION}_${STEP}"
    if [ -d "$DEST_DIR" ]; then
        echo "[${VERSION}] ⚠️ 이미 이동됨: $DEST_DIR (skip)" | tee -a "$LOG"
        continue
    fi

    if [ ! -d "$DEST" ]; then
        echo "[${VERSION}] ❌ 대상 드라이브 없음: $DEST — 변환만 완료, 이동 나중에 수동으로" | tee -a "$LOG"
        continue
    fi

    echo "[${VERSION}] 복사 중: $OUT_DIR → $DEST_DIR" | tee -a "$LOG"
    mkdir -p "$DEST"
    cp -r "$OUT_DIR" "$DEST_DIR" 2>&1 | tee -a "$LOG"
    echo "[${VERSION}] ✅ 이동 완료: $DEST_DIR" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "============================== $(date)" | tee -a "$LOG"
echo "[DONE] 전체 완료. 로그: $LOG" | tee -a "$LOG"
