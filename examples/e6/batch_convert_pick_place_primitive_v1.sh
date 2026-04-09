#!/usr/bin/env bash
# Batch-convert all Pick-place-random episodes to one LeRobot dataset with E6 primitive v1
# (per-frame task strings from e6_v1_task_contract.py). Options are fixed below for reproducibility.
#
# Prerequisite: E6_SEGMENTS_CSV must contain a row set for every episode folder under E6_RAW_ROOT
# (run RoboVLA scripts/data/build_episode_primitive_segments.py on the full tree first).
#
# Usage (from anywhere):
#   bash /path/to/openpi_upstream_clean/examples/e6/batch_convert_pick_place_primitive_v1.sh
#
# Optional env overrides:
#   E6_RAW_ROOT          — default: ~/26kp/RoboVLA/Pick-place-random
#   E6_SEGMENTS_CSV      — default: RoboVLA gate2 episode_primitive_segments.csv
#   E6_REPO_ID           — default: billy/dobot_e6_pick_place_random_v1
#   HF_LEROBOT_HOME      — Hugging Face LeRobot cache root (default if unset: ~/.cache/huggingface/lerobot)
#
# After run, prints meta/info.json total_frames and lists tasks from meta/tasks.jsonl.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${OPENPI_ROOT}/.venv/bin/python"
CONVERT="${OPENPI_ROOT}/examples/e6/convert_e6_episode_to_lerobot.py"

if [[ ! -x "$PY" ]]; then
  echo "Missing venv python: ${PY} (cd ${OPENPI_ROOT} && uv sync)" >&2
  exit 1
fi

# --- fixed defaults (do not change casually; keep in sync with team + training config) ---
: "${E6_RAW_ROOT:=${HOME}/26kp/RoboVLA/Pick-place-random}"
: "${E6_SEGMENTS_CSV:=${HOME}/26kp/RoboVLA/docs/gate2/primitive_segments_v1/episode_primitive_segments.csv}"
: "${E6_REPO_ID:=billy/dobot_e6_pick_place_random_v1}"
OBJECT_PHRASE="red object"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}"

if [[ ! -f "$E6_SEGMENTS_CSV" ]]; then
  echo "Missing segments CSV: $E6_SEGMENTS_CSV" >&2
  exit 1
fi
if [[ ! -d "$E6_RAW_ROOT" ]]; then
  echo "Missing raw dataset root: $E6_RAW_ROOT" >&2
  exit 1
fi

EPS_SORTED=()
while IFS= read -r d; do
  EPS_SORTED+=("$d")
done < <(
  find "$E6_RAW_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\t%p\n' 2>/dev/null \
    | awk -F'\t' '$1 ~ /^[0-9]+$/ {print $1"\t"$2}' \
    | sort -n \
    | cut -f2-
)

# Require robot_data.csv
TMP=()
for d in "${EPS_SORTED[@]}"; do
  [[ -f "$d/robot_data.csv" ]] && TMP+=("$(cd "$d" && pwd)")
done
EPS_SORTED=("${TMP[@]}")

if [[ ${#EPS_SORTED[@]} -eq 0 ]]; then
  echo "No numeric episode folders with robot_data.csv under $E6_RAW_ROOT" >&2
  exit 1
fi

echo "OPENPI_ROOT=$OPENPI_ROOT"
echo "E6_RAW_ROOT=$E6_RAW_ROOT"
echo "E6_REPO_ID=$E6_REPO_ID"
echo "E6_SEGMENTS_CSV=$E6_SEGMENTS_CSV"
echo "HF_LEROBOT_HOME=$HF_LEROBOT_HOME"
echo "Episodes: ${#EPS_SORTED[@]}"
echo "OBJECT_PHRASE=$OBJECT_PHRASE"
echo

"$PY" "$CONVERT" \
  --episode-dir "${EPS_SORTED[@]}" \
  --repo-id "$E6_REPO_ID" \
  --primitive-v1 \
  --segments-csv "$E6_SEGMENTS_CSV" \
  --object-phrase "$OBJECT_PHRASE" \
  --clean

META="${HF_LEROBOT_HOME}/${E6_REPO_ID}/meta"
INFO="${META}/info.json"
TASKS="${META}/tasks.jsonl"

echo ""
echo "=== Post-run checks ==="
if [[ -f "$INFO" ]]; then
  echo "--- meta/info.json (total_frames, total_episodes) ---"
  "$PY" - <<PY
import json
from pathlib import Path
p = Path("${INFO}")
info = json.loads(p.read_text())
print("total_frames:", info.get("total_frames"))
print("total_episodes:", info.get("total_episodes"))
PY
else
  echo "Missing $INFO" >&2
fi

if [[ -f "$TASKS" ]]; then
  echo "--- meta/tasks.jsonl (unique task strings) ---"
  "$PY" - <<PY
import json
from pathlib import Path
lines = Path("${TASKS}").read_text(encoding="utf-8").strip().splitlines()
texts = sorted({json.loads(L)["task"] for L in lines if L.strip()})
for t in texts:
    print(t)
print(f"(distinct prompts: {len(texts)}, task_index rows: {len(lines)})")
PY
else
  echo "Missing $TASKS" >&2
fi

echo ""
echo "Dataset root: ${HF_LEROBOT_HOME}/${E6_REPO_ID}"
echo "Next: compute_norm_stats for pi05_e6_v1 / pi05_e6_v1_lora, then train."
