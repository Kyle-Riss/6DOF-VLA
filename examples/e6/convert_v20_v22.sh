#!/bin/bash
# v20 (12500) 즉시 변환, v22 (19999) 완료 대기 후 자동 변환
# 새 볼륨3 마운트 시 자동 이동

BASE="/home/billy/26kp/openpi_upstream_clean"
DEST="/media/billy/새 볼륨3/e6_checkpoints"
LOG="/tmp/convert_v20_v22.log"
cd "$BASE"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

convert_and_move() {
    local VERSION=$1
    local STEP=$2
    local CONFIG="pi05_e6_${VERSION}_lora"
    local CKPT_DIR="checkpoints/pi05_e6_${VERSION}_lora/e6_2cam_lora_${VERSION}/${STEP}"
    local OUT_DIR="checkpoints/pi05_e6_${VERSION}_lora/e6_2cam_lora_${VERSION}/${STEP}_pytorch_lora_merged"

    log "==============================="
    log "[${VERSION}] step=${STEP} 변환 시작"

    if [ ! -d "$CKPT_DIR" ]; then
        log "[${VERSION}] ❌ 체크포인트 없음: $CKPT_DIR"
        return 1
    fi

    if [ -d "$OUT_DIR" ] && [ -f "$OUT_DIR/model.safetensors" ]; then
        log "[${VERSION}] ⚠️ 이미 변환됨: $OUT_DIR (skip)"
    else
        log "[${VERSION}] 변환 중 (JAX_PLATFORMS=cpu)..."
        JAX_PLATFORMS=cpu uv run examples/convert_jax_to_pytorch_lora_merged.py \
            --checkpoint-dir "$CKPT_DIR" \
            --config-name "$CONFIG" \
            --output-path "$OUT_DIR" \
            2>&1 | tee -a "$LOG"

        if [ -f "$OUT_DIR/model.safetensors" ]; then
            log "[${VERSION}] ✅ 변환 완료: $OUT_DIR"
        else
            log "[${VERSION}] ❌ 변환 실패 (model.safetensors 없음)"
            return 1
        fi
    fi

    # 새 볼륨3 이동
    local DEST_DIR="$DEST/e6_${VERSION}_${STEP}"
    if [ -d "$DEST_DIR" ]; then
        log "[${VERSION}] ⚠️ 이미 이동됨: $DEST_DIR (skip)"
        return 0
    fi

    if [ ! -d "$DEST" ]; then
        log "[${VERSION}] ⚠️ 새 볼륨3 없음 — 변환만 완료. 이동은 드라이브 연결 후 수동으로."
        return 0
    fi

    log "[${VERSION}] 복사 중: $OUT_DIR → $DEST_DIR"
    mkdir -p "$DEST"
    cp -r "$OUT_DIR" "$DEST_DIR" 2>&1 | tee -a "$LOG"
    log "[${VERSION}] ✅ 새 볼륨3 이동 완료: $DEST_DIR"
}

log "=============================="
log "convert_v20_v22.sh 시작"

# v20: 12500 step 즉시 변환
convert_and_move "v20" "12500"

# v22: 19999 step 대기 후 변환
V22_CKPT="checkpoints/pi05_e6_v22_lora/e6_2cam_lora_v22/19999"
log "==============================="
log "[v22] 19999 step 완료 대기 중..."

while [ ! -f "$V22_CKPT/orbax_checkpoint_handler" ] && [ ! -d "$V22_CKPT/0" ] && [ ! -f "$V22_CKPT/_CHECKPOINT_METADATA" ]; do
    sleep 60
    log "[v22] 대기 중... ($V22_CKPT 미존재)"
done

log "[v22] 체크포인트 감지됨. 30초 후 변환 시작 (저장 완료 대기)"
sleep 30

convert_and_move "v22" "19999"

log "=============================="
log "[DONE] 전체 완료. 로그: $LOG"
