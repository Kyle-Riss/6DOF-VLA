"""
Convert Dobot Magician E6 v2 raw episode folders to LeRobot format for pi05_e6_v6_lora.

v6 changes vs v2 (velocity action + idle filtering + per-step phase prompt):

* ``action``: velocity delta ``state[t+1] - state[t]`` (7D degrees/frame), NOT absolute next-position.
  Gripper component is also delta (−1/0/+1).
* Idle frame filtering: consecutive runs of ≥5 frames where max joint |delta| < 0.05°/frame are removed.
  Last 15 frames of each episode (post-release holding) are also removed before filtering.
* Per-frame phase prompt: each frame receives one of five phase-specific prompts derived from
  z height, gripper state, and transitions — NOT a single episode-level string.
* HuggingFace repo: ``Kyle-Riss/dobot_e6_pick_place_orange_v6``

Phase classification (per frame, 0-indexed):
  approach  — gripper=0 (before grasp)
  grasp     — ±5 frames around gripper 0→1 transition
  transport — gripper=1, z>180mm
  place     — gripper=1, z≤200mm
  release   — ±5 frames around gripper 1→0 transition

Action semantics differ from v1/v2:
  v1/v2: action = absolute next joint position (degrees)
  v6:    action = velocity delta (degrees/frame); gripper delta is 0/±1

Inference code (run_e6_client.py) must integrate velocity to get absolute targets.
That change is deferred — do NOT use v6 checkpoint with v2-style inference.

Per-frame contract (matches ``LeRobotE6DataConfig`` + ``E6Inputs``):
  ``exterior_image_1_left``: HIK side camera, CHW uint8
  ``exterior_image_2_left``: ZED side camera, CHW uint8
  ``state``:  (7,) float32 — [j1..j6, gripper_tooldo1] at t (absolute position)
  ``action``: (7,) float32 — [Δj1..Δj6, Δgripper] = state[t+1]−state[t] (velocity)
  ``task``:   per-frame phase prompt string

Expected raw layout per episode directory::

    episode_dir/
      images/hik/frame_XXXXXX.jpg
      images/zed/frame_XXXXXX.jpg
      robot_data.csv     # j1..j6, gripper_tooldo1, z, image_path_hik, image_path_zed, ...
      episode_meta.json  # source_zone, target_zone (values: "left" / "right")

Usage::

    uv run examples/e6/convert_e6_v6_to_lerobot.py \\
      --root /home/billy/26kp/2CAM-Orange

    uv run examples/e6/convert_e6_v6_to_lerobot.py \\
      --root /home/billy/26kp/2CAM-Orange \\
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

DEFAULT_REPO_ID = "Kyle-Riss/dobot_e6_pick_place_orange_v6"

JOINT_COLS = ("j1", "j2", "j3", "j4", "j5", "j6")
GRIPPER_COL = "gripper_tooldo1"
Z_COL = "z"
IMAGE_COL_HIK = "image_path_hik"
IMAGE_COL_ZED = "image_path_zed"

# Phase prompt templates; {side} = source zone, {target_side} = target zone.
PHASE_PROMPTS: dict[str, str] = {
    "approach":  "move the arm down to approach the orange box on the {side}",
    "grasp":     "grasp the orange box on the {side}",
    "transport": "lift and carry the orange box to the {target_side}",
    "place":     "lower the orange box onto the {target_side}",
    "release":   "release the orange box on the {target_side}",
}

# Idle filtering parameters.
IDLE_THRESHOLD_DEG = 0.05   # max joint |delta| below which a frame is "idle"
MIN_IDLE_RUN = 5             # remove consecutive idle runs at least this long
LAST_FRAME_TRIM = 15         # drop last N frames per episode (post-release holding)

# Phase classification thresholds (mm).
Z_HIGH = 200.0   # z > Z_HIGH with gripper=0 → approach; gripper=1 → transport
Z_LIFT = 180.0   # gripper=1, z > Z_LIFT → transport
TRANSITION_WINDOW = 5  # frames around gripper transition labelled grasp/release


# ---------------------------------------------------------------------------
# Phase classification
# ---------------------------------------------------------------------------

def classify_phases(df: pd.DataFrame) -> list[str]:
    """Assign a phase label to every frame in the filtered episode DataFrame."""
    n = len(df)
    z = df[Z_COL].values.astype(float)
    gripper = df[GRIPPER_COL].values.astype(int)

    # Detect gripper transitions.
    grip_frames: list[int] = []    # 0→1
    release_frames: list[int] = []  # 1→0
    for i in range(1, n):
        if gripper[i - 1] == 0 and gripper[i] == 1:
            grip_frames.append(i)
        elif gripper[i - 1] == 1 and gripper[i] == 0:
            release_frames.append(i)

    # Base phase from gripper + z.
    phases: list[str] = []
    for i in range(n):
        if gripper[i] == 0:
            phases.append("approach")
        elif z[i] > Z_LIFT:
            phases.append("transport")
        else:
            phases.append("place")

    # Override transition windows.
    for gf in grip_frames:
        lo = max(0, gf - TRANSITION_WINDOW)
        hi = min(n, gf + TRANSITION_WINDOW + 1)
        for i in range(lo, hi):
            phases[i] = "grasp"

    for rf in release_frames:
        lo = max(0, rf - TRANSITION_WINDOW)
        hi = min(n, rf + TRANSITION_WINDOW + 1)
        for i in range(lo, hi):
            phases[i] = "release"

    return phases


# ---------------------------------------------------------------------------
# Idle filtering
# ---------------------------------------------------------------------------

def compute_idle_mask(df: pd.DataFrame) -> np.ndarray:
    """Return bool array (length n): True where joint delta < IDLE_THRESHOLD_DEG.

    Frame 0 is never marked idle (no previous frame to compare).
    """
    joints = df[list(JOINT_COLS)].values.astype(np.float32)
    n = len(joints)
    idle = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if np.abs(joints[i] - joints[i - 1]).max() < IDLE_THRESHOLD_DEG:
            idle[i] = True
    return idle


def build_keep_mask(idle: np.ndarray) -> np.ndarray:
    """Return bool array: True for frames to keep (remove consecutive idle runs ≥ MIN_IDLE_RUN)."""
    keep = np.ones(len(idle), dtype=bool)
    in_run = False
    run_start = 0
    for i, is_idle in enumerate(idle):
        if is_idle:
            if not in_run:
                in_run = True
                run_start = i
        else:
            if in_run:
                if i - run_start >= MIN_IDLE_RUN:
                    keep[run_start:i] = False
                in_run = False
    if in_run and len(idle) - run_start >= MIN_IDLE_RUN:
        keep[run_start:] = False
    return keep


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_image_chw(path: Path, resize: int | None) -> np.ndarray:
    img = PIL.Image.open(path).convert("RGB")
    if resize is not None:
        img = img.resize((resize, resize), PIL.Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB at {path}, got {arr.shape}")
    return np.transpose(arr, (2, 0, 1))


# ---------------------------------------------------------------------------
# Episode helpers
# ---------------------------------------------------------------------------

def _read_csv(episode_dir: Path, csv_name: str) -> pd.DataFrame:
    p = episode_dir / csv_name
    if not p.is_file():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if len(df) < LAST_FRAME_TRIM + MIN_IDLE_RUN + 2:
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
    candidates: list[tuple[int, Path]] = [
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
    exclude: tuple[int, ...] = (382,),
    csv_name: str = "robot_data.csv",
    images_subdir: str = "images",
    fps: int = 16,
    robot_type: str = "magician_e6",
    resize: int | None = None,
    clean: bool = True,
    push_to_hub: bool = False,
    hub_private: bool = False,
    image_writer_threads: int = 4,
    image_writer_processes: int = 0,
) -> None:
    """Convert v2 raw episodes into v6 LeRobot dataset (velocity action + idle filter + phase prompt)."""
    episode_paths = _resolve_episodes(episode_dir, root, exclude)

    output_root = HF_LEROBOT_HOME / repo_id
    if clean and output_root.exists():
        shutil.rmtree(output_root)

    # Probe image shapes from first episode.
    df0 = _read_csv(episode_paths[0], csv_name)
    hik0 = episode_paths[0] / images_subdir / str(df0[IMAGE_COL_HIK].iloc[0])
    zed0 = episode_paths[0] / images_subdir / str(df0[IMAGE_COL_ZED].iloc[0])
    sample_hik = _load_image_chw(hik0, resize)
    sample_zed = _load_image_chw(zed0, resize)
    _, h_hik, w_hik = sample_hik.shape
    _, h_zed, w_zed = sample_zed.shape

    features = {
        "exterior_image_1_left": {
            "dtype": "image",
            "shape": (3, h_hik, w_hik),
            "names": ["channel", "height", "width"],
        },
        "exterior_image_2_left": {
            "dtype": "image",
            "shape": (3, h_zed, w_zed),
            "names": ["channel", "height", "width"],
        },
        "state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dj1", "dj2", "dj3", "dj4", "dj5", "dj6", "dgripper"],
        },
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

    total_frames = 0
    total_raw = 0
    skipped = 0

    for ep_idx, ep in enumerate(episode_paths):
        try:
            df_raw = _read_csv(ep, csv_name)
            meta = _read_meta(ep)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  SKIP {ep.name}: {exc}")
            skipped += 1
            continue

        missing = [c for c in [*JOINT_COLS, GRIPPER_COL, Z_COL, IMAGE_COL_HIK, IMAGE_COL_ZED] if c not in df_raw.columns]
        if missing:
            print(f"  SKIP {ep.name}: missing CSV columns {missing}")
            skipped += 1
            continue

        source_zone = meta.get("source_zone", "left")
        target_zone = meta.get("target_zone", "right")

        # Step 1: trim last LAST_FRAME_TRIM frames.
        df = df_raw.iloc[:-LAST_FRAME_TRIM].reset_index(drop=True)

        # Step 2: idle filtering — remove consecutive idle runs ≥ MIN_IDLE_RUN.
        idle = compute_idle_mask(df)
        keep = build_keep_mask(idle)
        df_f = df[keep].reset_index(drop=True)

        raw_n = len(df_raw)
        filtered_n = len(df_f)
        total_raw += raw_n

        if filtered_n < 2:
            print(f"  SKIP {ep.name}: only {filtered_n} frames after filtering")
            skipped += 1
            continue

        # Step 3: classify phases on filtered frames.
        phases = classify_phases(df_f)

        # Step 4: build (t, t+1) velocity-action pairs.
        n_added = 0
        for t in range(filtered_n - 1):
            cur = df_f.iloc[t]
            nxt = df_f.iloc[t + 1]

            hik_path = ep / images_subdir / str(cur[IMAGE_COL_HIK])
            zed_path = ep / images_subdir / str(cur[IMAGE_COL_ZED])
            if not hik_path.is_file():
                raise FileNotFoundError(f"Missing HIK {hik_path}")
            if not zed_path.is_file():
                raise FileNotFoundError(f"Missing ZED {zed_path}")

            joints_cur = np.array([float(cur[c]) for c in JOINT_COLS], dtype=np.float32)
            joints_nxt = np.array([float(nxt[c]) for c in JOINT_COLS], dtype=np.float32)
            g_cur = np.float32(cur[GRIPPER_COL])
            g_nxt = np.float32(nxt[GRIPPER_COL])

            state = np.concatenate([joints_cur, [g_cur]])
            action = np.concatenate([joints_nxt - joints_cur, [g_nxt - g_cur]])  # velocity delta

            hik_chw = _load_image_chw(hik_path, resize)
            zed_chw = _load_image_chw(zed_path, resize)
            if hik_chw.shape != (3, h_hik, w_hik):
                raise ValueError(f"HIK shape mismatch at {hik_path}")
            if zed_chw.shape != (3, h_zed, w_zed):
                raise ValueError(f"ZED shape mismatch at {zed_path}")

            phase = phases[t]
            prompt = PHASE_PROMPTS[phase].format(side=source_zone, target_side=target_zone)

            dataset.add_frame({
                "exterior_image_1_left": hik_chw,
                "exterior_image_2_left": zed_chw,
                "state": state,
                "action": action,
                "task": prompt,
            })
            n_added += 1

        dataset.save_episode()
        total_frames += n_added
        removed = raw_n - filtered_n
        print(
            f"[{ep_idx + 1}/{len(episode_paths)}] ep={ep.name}: "
            f"raw={raw_n} → filtered={filtered_n} (−{removed}) → pairs={n_added} | "
            f"src={source_zone} tgt={target_zone}"
        )

    print(
        f"\nDone. {len(episode_paths) - skipped} episodes, {total_frames} frames "
        f"(skipped {skipped} episodes). Local root: {dataset.root}"
    )
    if total_raw > 0:
        print(f"Overall frame retention: {total_frames}/{total_raw} "
              f"= {100 * total_frames / total_raw:.1f}%")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["e6", "magician", "openpi", "v6", "velocity"],
            private=hub_private,
            push_videos=False,
            license="apache-2.0",
        )
        print(f"Pushed to HuggingFace: {repo_id}")


if __name__ == "__main__":
    tyro.cli(main)
