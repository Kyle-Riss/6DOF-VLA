"""
Convert Dobot MagicianE6 v16 raw episodes to LeRobot format for pi05_e6_v16_lora.

v16 changes vs v15:
  - action[6] = gripper ABSOLUTE (0.0 or 1.0), NOT delta.
    Fixes oscillation caused by sparse delta {-1,0,+1} being noisy at inference.
    Executor: suction_cmd = 1 if action[6] > 0.5 else 0 (no accumulation).
  - Per-frame phase-aligned language prompts (6 phases × 2 directions = 12 tasks).
    Each frame gets a phase-specific instruction based on gripper state and
    temporal position within the carry segment. No task_group_map randomisation.

Phase labeling (assigned per frame in trimmed episode):
  0 approach  : [start, pick_idx)  — gripper=0, robot moving toward object
  1 pick      : [pick_idx, pick_idx+PICK_SETTLE)  — gripper just closed (0→1)
  2 lift      : gripper=1, j1 not yet rotating (robot rising from pick position)
  3 transport : j1 rotating fast — base swinging between left↔right zones
  4 place     : j1 slowed, robot descending to place position
  5 release   : [open_idx, trail_end)  — gripper opens (1→0), vacuum off

  lift/transport/place boundary detection uses smoothed |dj1/dt|.
  j1 (base rotation) is the dominant axis for left↔right lateral motion in E6.
  If j1 barely moves (degenerate), falls back to proportional 30%/45%/25% split.

Action contract (7D):
  action[0:6] = joint velocity delta [Δj1..Δj6] (degrees/frame)
  action[6]   = gripper absolute: 1.0=vacuum ON (closed/holding), 0.0=vacuum OFF
  → No norm_stats patch needed (natural q01=0.0, q99=1.0 for binary absolute)

tasks.jsonl (12 entries, 2 direction × 6 phases):
  Left→right (indices 0-5):
    0: "approach the orange box on the left"
    1: "grasp the orange box"
    2: "lift the orange box"
    3: "carry the orange box to the right"
    4: "lower the orange box onto the right side"
    5: "release the orange box on the right"
  Right→left (indices 6-11):
    6: "approach the orange box on the right"
    7: "grasp the orange box"
    8: "lift the orange box"
    9: "carry the orange box to the left"
    10: "lower the orange box onto the left side"
    11: "release the orange box on the left"

Usage:
    uv run examples/e6/convert_e6_v16_to_lerobot.py \\
      --root "/media/billy/새 볼륨2/Dobot/2CAM-Orange-init"

    uv run examples/e6/convert_e6_v16_to_lerobot.py \\
      --root "/media/billy/새 볼륨2/Dobot/2CAM-Orange-init" \\
      --push-to-hub --hub-private
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
import tyro

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_REPO_ID = "Kyle-Riss/dobot_e6_pick_place_orange_v16"

JOINT_COLS = ("j1", "j2", "j3", "j4", "j5", "j6")
GRIPPER_COL = "gripper_tooldo1"
IMAGE_COL_HIK = "image_path_hik"
IMAGE_COL_ZED = "image_path_zed"

MOTION_THRESHOLD_DEG = 0.1
PLACE_SETTLE = 10    # trailing frames to keep after gripper 1→0 (place event)
PICK_SETTLE = 4      # frames AFTER close_idx still labeled "pick up"
ACTION_HORIZON = 16  # must match pi05_e6_v16_lora config action_horizon

# 10 unique task strings (indices 0-9).
# lift(2) and release(5) are direction-agnostic and shared between both directions.
# approach, pick up, transport, place are direction-specific.
TASKS: list[str] = [
    "approach the orange box on the left side",    # 0
    "pick up the orange box on the left side",     # 1
    "lift the orange box",                          # 2  ← shared
    "move the orange box to the right section",    # 3
    "place the orange box in the right section",   # 4
    "release the orange box",                       # 5  ← shared
    "approach the orange box on the right side",   # 6
    "pick up the orange box on the right side",    # 7
    "move the orange box to the left section",     # 8
    "place the orange box in the left section",    # 9
]

# (direction, phase) → task index
# phase: 0=approach, 1=pick up, 2=lift, 3=transport, 4=place, 5=release
TASK_INDEX: dict[tuple[str, int], int] = {
    ("left",  0): 0,  ("left",  1): 1,  ("left",  2): 2,
    ("left",  3): 3,  ("left",  4): 4,  ("left",  5): 5,
    ("right", 0): 6,  ("right", 1): 7,  ("right", 2): 2,
    ("right", 3): 8,  ("right", 4): 9,  ("right", 5): 5,
}

NUM_PHASES = 6  # approach, pick, lift, transport, place, release


# ---------------------------------------------------------------------------
# Trimming helpers
# ---------------------------------------------------------------------------

def find_first_motion(df: pd.DataFrame) -> int:
    joints = df[list(JOINT_COLS)].values.astype(np.float32)
    for i in range(1, len(joints)):
        if np.abs(joints[i] - joints[i - 1]).max() >= MOTION_THRESHOLD_DEG:
            return i
    return 0


def find_close_idx(gripper: np.ndarray) -> int | None:
    """First frame where gripper transitions 0→1 (vacuum ON, pick event)."""
    for i in range(1, len(gripper)):
        if gripper[i - 1] < 0.5 and gripper[i] > 0.5:
            return i
    return None


def find_open_idx(gripper: np.ndarray) -> int | None:
    """First frame where gripper transitions 1→0 (vacuum OFF, place event)."""
    for i in range(1, len(gripper)):
        if gripper[i - 1] > 0.5 and gripper[i] < 0.5:
            return i
    return None


# ---------------------------------------------------------------------------
# Phase assignment
# ---------------------------------------------------------------------------

SMOOTH_WINDOW = 5          # velocity smoothing window (frames)
TRANSPORT_PERCENTILE = 40  # motion_max percentile used as transport threshold


def _compute_carry_phases(
    joints: np.ndarray,
    carry_start: int,
    carry_end: int,
) -> tuple[int, int]:
    """Return (transport_start, transport_end) within [carry_start, carry_end).

    Uses smoothed motion_max = max(|Δj1|~|Δj6|) across all joints to find the
    contiguous block where the robot moves most (lateral transport between zones).
    Falls back to proportional 30%/45%/25% split if motion is uniformly low.
    """
    if carry_end <= carry_start:
        return carry_start, carry_start

    # motion_max: max absolute delta across all 6 joints per frame.
    delta = np.abs(np.diff(joints, axis=0, prepend=joints[:1]))  # (n, 6)
    motion_max = np.max(delta, axis=1)                            # (n,)

    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    motion_sm = np.convolve(motion_max, kernel, mode="same")

    seg = motion_sm[carry_start:carry_end]
    thresh = np.percentile(seg, TRANSPORT_PERCENTILE)

    # Degenerate: robot barely moves in carry segment. Use proportional fallback.
    if thresh < 0.05 or seg.max() < 0.10:
        span = carry_end - carry_start
        return carry_start + int(span * 0.30), carry_start + int(span * 0.75)

    # First frame above threshold → transport_start.
    transport_start = carry_end
    for i in range(carry_start, carry_end):
        if motion_sm[i] > thresh:
            transport_start = i
            break

    # Last frame above threshold → transport_end.
    transport_end = transport_start
    for i in range(carry_end - 1, transport_start - 1, -1):
        if motion_sm[i] > thresh:
            transport_end = i + 1
            break

    return transport_start, transport_end


def compute_phases(df: pd.DataFrame, pick_idx: int, open_idx: int) -> np.ndarray:
    """Return per-frame phase array (int, 0-5) for trimmed episode df.

    0=approach, 1=pick up, 2=lift, 3=transport, 4=place, 5=release

    Action-horizon-aware phase shifts:
      pick phase starts ACTION_HORIZON frames BEFORE close_idx so that
      action chunks in the approach tail are already labeled "pick up".

      release phase starts ACTION_HORIZON frames BEFORE open_idx (but no
      earlier than place_start) so action chunks carrying the gripper-open
      event are labeled "release", not "place".
    """
    n = len(df)
    phases = np.full(n, -1, dtype=np.int32)

    # Pick phase: ACTION_HORIZON before close through PICK_SETTLE after.
    pick_start = max(0, pick_idx - ACTION_HORIZON)
    pick_end   = min(pick_idx + PICK_SETTLE, open_idx)

    phases[:pick_start] = 0              # approach
    phases[pick_start:pick_end] = 1     # pick up

    # Carry segment for lift/transport/place: after pick, before release.
    carry_start = pick_end

    # Release phase: ACTION_HORIZON before open, but not before carry_start.
    release_start = max(carry_start, open_idx - ACTION_HORIZON)

    phases[release_start:] = 5          # release (covers open_idx + PLACE_SETTLE tail)

    # Lift/transport/place within [carry_start, release_start).
    if carry_start < release_start:
        joints = df[list(JOINT_COLS)].values.astype(np.float64)
        t_start, t_end = _compute_carry_phases(joints, carry_start, release_start)
        # Clamp t_end so place phase always has at least some frames.
        t_end = min(t_end, release_start)
        phases[carry_start:t_start] = 2  # lift
        phases[t_start:t_end]       = 3  # transport
        phases[t_end:release_start] = 4  # place

    phases[phases < 0] = 3              # guard: unassigned → transport
    return phases


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_image_hwc(path: Path, resize: int | None) -> np.ndarray:
    img = PIL.Image.open(path).convert("RGB")
    if resize is not None:
        img = img.resize((resize, resize), PIL.Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Episode helpers
# ---------------------------------------------------------------------------

def _read_csv(episode_dir: Path, csv_name: str) -> pd.DataFrame:
    p = episode_dir / csv_name
    if not p.is_file():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if len(df) < 20:
        raise ValueError(f"{p}: too few rows ({len(df)})")
    return df


def _read_meta(episode_dir: Path) -> dict:
    p = episode_dir / "episode_meta.json"
    if not p.is_file():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_episodes(
    episode_dir: list[Path] | None,
    root: Path | None,
    exclude: tuple[int, ...],
) -> list[Path]:
    if episode_dir:
        return [Path(p).resolve() for p in episode_dir]
    if root is None:
        raise ValueError("Provide --episode-dir or --root.")
    root = Path(root).resolve()
    excluded = set(exclude)
    candidates = [
        (int(c.name), c)
        for c in root.iterdir()
        if c.is_dir() and c.name.isdigit() and int(c.name) not in excluded
    ]
    candidates.sort(key=lambda x: x[0])
    if not candidates:
        raise ValueError(f"No numeric episode dirs under {root}")
    return [p for _, p in candidates]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    root: Path | None = None,
    episode_dir: list[Path] | None = None,
    exclude: tuple[int, ...] = (193,),
    csv_name: str = "robot_data.csv",
    images_subdir: str = "images",
    fps: int = 16,
    robot_type: str = "magician_e6",
    resize: int | None = None,
    clean: bool = True,
    push_to_hub: bool = False,
    hub_private: bool = False,
    image_writer_threads: int = 10,
    image_writer_processes: int = 4,
) -> None:
    episode_paths = _resolve_episodes(episode_dir, root, exclude)

    output_root = HF_LEROBOT_HOME / repo_id
    if clean and output_root.exists():
        shutil.rmtree(output_root)

    df0 = _read_csv(episode_paths[0], csv_name)
    hik0 = episode_paths[0] / images_subdir / str(df0[IMAGE_COL_HIK].iloc[0])
    zed0 = episode_paths[0] / images_subdir / str(df0[IMAGE_COL_ZED].iloc[0])
    sample_hik = _load_image_hwc(hik0, resize)
    sample_zed = _load_image_hwc(zed0, resize)
    h_hik, w_hik, _ = sample_hik.shape
    h_zed, w_zed, _ = sample_zed.shape

    features = {
        "exterior_image_1_left": {
            "dtype": "image",
            "shape": (h_hik, w_hik, 3),
            "names": ["height", "width", "channel"],
        },
        "exterior_image_2_left": {
            "dtype": "image",
            "shape": (h_zed, w_zed, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dj1", "dj2", "dj3", "dj4", "dj5", "dj6", "gripper_abs"],
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done":   {"dtype": "bool",    "shape": (1,), "names": None},
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=False,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    # Pre-register unique task strings (some direction-agnostic phases share the same text).
    for task_str in dict.fromkeys(TASKS):
        dataset.meta.add_task(task_str)

    total_frames = 0
    skipped_eps = 0
    direction_counts: dict[str, int] = {"left": 0, "right": 0}
    phase_counts = [0] * NUM_PHASES

    for ep_idx, ep in enumerate(episode_paths):
        try:
            df_raw = _read_csv(ep, csv_name)
            meta = _read_meta(ep)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  SKIP {ep.name}: {exc}")
            skipped_eps += 1
            continue

        missing = [c for c in [*JOINT_COLS, GRIPPER_COL, IMAGE_COL_HIK, IMAGE_COL_ZED]
                   if c not in df_raw.columns]
        if missing:
            print(f"  SKIP {ep.name}: missing columns {missing}")
            skipped_eps += 1
            continue

        source_zone = meta.get("source_zone", "").strip().lower()
        if source_zone not in ("left", "right"):
            print(f"  SKIP {ep.name}: unknown source_zone={source_zone!r}")
            skipped_eps += 1
            continue

        gripper_raw = df_raw[GRIPPER_COL].values.astype(np.float32)

        # Trailing trim uses place event (gripper 1→0) in raw data.
        raw_open_idx = find_open_idx(gripper_raw)
        if raw_open_idx is None:
            print(f"  SKIP {ep.name}: no gripper 1→0 transition found")
            skipped_eps += 1
            continue

        trail_end = min(raw_open_idx + PLACE_SETTLE, len(df_raw))
        df = df_raw.iloc[:trail_end].reset_index(drop=True)

        first_motion = find_first_motion(df)
        start = max(0, first_motion - 1)
        df = df.iloc[start:].reset_index(drop=True)

        if len(df) < 16:
            print(f"  SKIP {ep.name}: too short after trim ({len(df)} frames)")
            skipped_eps += 1
            continue

        # Re-find phase events in the trimmed frame of reference.
        gripper_trimmed = df[GRIPPER_COL].values.astype(np.float32)
        pick_idx = find_close_idx(gripper_trimmed)
        open_idx = find_open_idx(gripper_trimmed)

        if pick_idx is None:
            print(f"  SKIP {ep.name}: no gripper 0→1 (pick) transition in trimmed ep")
            skipped_eps += 1
            continue
        if open_idx is None:
            print(f"  SKIP {ep.name}: no gripper 1→0 (place) transition in trimmed ep")
            skipped_eps += 1
            continue
        if pick_idx >= open_idx:
            print(f"  SKIP {ep.name}: pick_idx({pick_idx}) >= open_idx({open_idx})")
            skipped_eps += 1
            continue

        # Compute per-frame phase labels using joint motion signal.
        phase_arr = compute_phases(df, pick_idx, open_idx)
        n_pairs = len(df) - 1

        for t in range(n_pairs):
            cur = df.iloc[t]
            nxt = df.iloc[t + 1]

            hik_path = ep / images_subdir / str(cur[IMAGE_COL_HIK])
            zed_path = ep / images_subdir / str(cur[IMAGE_COL_ZED])

            joints_cur = np.array([float(cur[c]) for c in JOINT_COLS], dtype=np.float32)
            joints_nxt = np.array([float(nxt[c]) for c in JOINT_COLS], dtype=np.float32)
            g_cur = np.float32(cur[GRIPPER_COL])
            g_nxt = np.float32(nxt[GRIPPER_COL])

            state  = np.concatenate([joints_cur, [g_cur]])
            # v16: gripper action = ABSOLUTE next-state (not delta)
            action = np.concatenate([joints_nxt - joints_cur, [g_nxt]])

            phase = int(phase_arr[t])
            task_index = TASK_INDEX[(source_zone, phase)]
            task_str = TASKS[task_index]

            phase_counts[phase] += 1
            is_last = (t == n_pairs - 1)

            dataset.add_frame({
                "exterior_image_1_left": _load_image_hwc(hik_path, resize),
                "exterior_image_2_left": _load_image_hwc(zed_path, resize),
                "state":  state,
                "action": action,
                "next.reward": np.array([1.0 if is_last else 0.0], dtype=np.float32),
                "next.done":   np.array([is_last], dtype=bool),
                "task": task_str,
            })

        dataset.save_episode()
        total_frames += n_pairs
        direction_counts[source_zone] += 1

        print(
            f"[{ep_idx + 1:3d}/{len(episode_paths)}] ep={ep.name:>4s}: "
            f"raw={len(df_raw)} → trimmed={len(df)} ({n_pairs} pairs) | "
            f"pick@{pick_idx} place@{open_idx} src={source_zone}"
        )

    phase_names = ["approach", "pick", "lift", "transport", "place", "release"]
    print(f"\nDone.")
    print(f"  Raw episodes  : {len(episode_paths)} ({skipped_eps} skipped)")
    print(f"  Episodes saved: {len(episode_paths) - skipped_eps} "
          f"(left={direction_counts['left']}, right={direction_counts['right']})")
    print(f"  Frame pairs   : {total_frames}")
    print(f"  Phase distribution:")
    for i, (name, count) in enumerate(zip(phase_names, phase_counts)):
        pct = 100.0 * count / max(1, total_frames)
        print(f"    {i} {name:12s}: {count:6d} ({pct:.1f}%)")
    print(f"  Local root    : {dataset.root}")
    print(f"\nNext steps:")
    print(f"  1. uv run scripts/compute_norm_stats.py --config-name pi05_e6_v16_lora")
    print(f"     (no gripper patch needed — absolute binary action naturally q01=0, q99=1)")
    print(f"  2. uv run scripts/train.py pi05_e6_v16_lora --exp-name e6_2cam_lora_v16")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["e6", "magician", "openpi", "v16", "velocity", "phase-prompt", "absolute-gripper"],
            private=hub_private,
            push_videos=False,
            license="apache-2.0",
        )
        print(f"\nPushed to HuggingFace: {repo_id}")


if __name__ == "__main__":
    tyro.cli(main)
