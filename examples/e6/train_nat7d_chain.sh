#!/bin/bash
# Native 7D action-head experiment set (all action_dim=7, pi05_base, 15k steps).
#   Vision LoRA sweep (freeze_filter_v4, random head):
#     v30 (18,26) late  | v29 (10,17) mid | v28 (0,9) lo
#   Head-init comparison @ (18,26):  v30 random  vs  v31 sliced
#   Full VLM frozen (v1/v2 style, no vision LoRA, expert full LoRA):  v32
#   (32D reference = v23, already done.)
#
# Each: train 15k → JAX→PyTorch LoRA-merged conversion.
# keep_period=None + max_to_keep=1 → only latest checkpoint kept per run.
# All 7D → shared norm_stats (assets/pi05_e6_nat7d_lora), computed once.
# Checkpoints → ./checkpoints (repo, TEMPORARY — revert to new volume after; see plan).
set -eo pipefail   # pipefail so a train.py failure through `| tee` actually stops the chain
cd /home/billy/26kp/openpi_upstream_clean
LOG=/tmp

echo "=== [NAT7D] norm_stats (7D, shared) ==="
uv run scripts/compute_norm_stats.py --config-name pi05_e6_v30_nat7d_late_lora 2>&1 | tee "$LOG/norm_nat7d.log"

run () {
  local CONFIG=$1 EXP=$2
  echo "=============================="
  echo "[NAT7D][TRAIN] $CONFIG  ($EXP)"
  echo "=============================="
  uv run scripts/train.py "$CONFIG" --exp-name "$EXP" 2>&1 | tee "$LOG/train_$EXP.log"

  local DIR="./checkpoints/$CONFIG/$EXP"
  local STEP; STEP=$(ls -1 "$DIR" | grep -E '^[0-9]+$' | sort -n | tail -1)
  local FULL="$DIR/$STEP"
  echo "[NAT7D][CONVERT] $CONFIG @ $FULL"
  uv run python examples/convert_jax_to_pytorch_lora_merged.py \
    --checkpoint-dir "$FULL" --config-name "$CONFIG" \
    --output-path "${FULL}_pytorch_lora_merged" 2>&1 | tee "$LOG/convert_$EXP.log"
}

# core comparison first (late-random / sliced / full-frozen), then rest of vision sweep
run pi05_e6_v30_nat7d_late_lora      e6_v30_nat7d_random      # vision (18,26), random head
run pi05_e6_v31_nat7d_sliced_lora    e6_v31_nat7d_sliced      # vision (18,26), sliced head
run pi05_e6_v32_nat7d_vlmfrozen_lora e6_v32_nat7d_vlmfrozen   # no vision LoRA (full frozen)
run pi05_e6_v29_nat7d_mid_lora       e6_v29_nat7d_mid         # vision (10,17), random head
run pi05_e6_v28_nat7d_lo_lora        e6_v28_nat7d_lo          # vision (0,9),  random head

echo "=============================="
echo "[NAT7D] 완료: v30/v31/v32/v29/v28 train+convert. v23(32D)와 대조."
echo "=============================="
