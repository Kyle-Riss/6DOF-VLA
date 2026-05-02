"""
Convert Dobot Magician E6 v8 raw episode folders to LeRobot format for pi05_e6_v8_lora.

v8 changes vs v6:
* Clean dataset (534 eps, no j1-inversion contamination).
* Phase classification fix: lift/place_descent are now distinguished via crossed_lift flag.
  v6 incorrectly labelled post-grasp ascent (lift) as "place" because both share
  gripper=1 + z≤Z_LIFT. Now tracked by whether z has crossed Z_LIFT after grasp.
* ep285 excluded (14Hz, low frame rate).
* New HF repo: Kyle-Riss/dobot_e6_pick_place_orange_v8

Phase classification (per frame):
  approach  — gripper=0, before grasp
  grasp     — ±5 frames around gripper 0→1 transition
  lift      — gripper=1, z≤Z_LIFT, before first crossing of Z_LIFT (post-grasp ascent)
  transport — gripper=1, z>Z_LIFT
  place     — gripper=1, z≤Z_LIFT, after crossing Z_LIFT (descent to place position)
  release   — ±5 frames around gripper 1→0 transition

Action semantics (same as v6):
  action = velocity delta (degrees/frame); gripper delta is 0/±1
  Inference server must integrate: target = current + action

Per-frame contract:
  exterior_image_1_left: HIK camera, CHW uint8
  exterior_image_2_left: ZED camera, CHW uint8
  state:  (7,) float32 — [j1..j6, gripper_tooldo1] absolute position
  action: (7,) float32 — [Δj1..Δj6, Δgripper] velocity delta
  task:   per-frame phase prompt string

Expected raw layout per episode directory:
    N/
      images/hik/frame_XXXXXX.jpg
      images/zed/frame_XXXXXX.jpg
      robot_data.csv
      episode_meta.json

Usage:
    uv run examples/e6/convert_e6_v8_to_lerobot.py \\
      --root "/media/billy/새 볼륨/Dobot/2CAM-Orange"

    uv run examples/e6/convert_e6_v8_to_lerobot.py \\
      --root "/media/billy/새 볼륨/Dobot/2CAM-Orange" \\
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

DEFAULT_REPO_ID = "Kyle-Riss/dobot_e6_pick_place_orange_v8"

JOINT_COLS = ("j1", "j2", "j3", "j4", "j5", "j6")
GRIPPER_COL = "gripper_tooldo1"
Z_COL = "z"
IMAGE_COL_HIK = "image_path_hik"
IMAGE_COL_ZED = "image_path_zed"

PHASE_PROMPTS: dict[str, str] = {
    "approach":  "move the arm down to approach the orange box on the {side}",
    "grasp":     "grasp the orange box on the {side}",
    "lift":      "lift the orange box from the {side}",
    "transport": "lift and carry the orange box to the {target_side}",
    "place":     "lower the orange box onto the {target_side}",
    "release":   "release the orange box on the {target_side}",
}

IDLE_THRESHOLD_DEG = 0.05
MIN_IDLE_RUN = 5
LAST_FRAME_TRIM = 15

Z_LIFT = 180.0        # z > Z_LIFT with gripper=1 → transport
TRANSITION_WINDOW = 5


# ---------------------------------------------------------------------------
# Phase classification
# ---------------------------------------------------------------------------

def classify_phases(df: pd.DataFrame) -> list[str]:
    """Assign phase label to every frame.

    Key fix vs v6: lift vs place are both gripper=1 + z≤Z_LIFT, but:
    - lift:  post-grasp ascent (z has NOT yet crossed Z_LIFT after grasp)
    - place: post-transport descent (z HAS crossed Z_LIFT at least once after grasp)

    Tracked via crossed_lift flag reset at each grasp event.
    """
    n = len(df)
    z = df[Z_COL].values.astype(float)
    gripper = df[GRIPPER_COL].values.astype(int)

    grip_frames: list[int] = []
    release_frames: list[int] = []
    for i in range(1, n):
        if gripper[i - 1] == 0 and gripper[i] == 1:
            grip_frames.append(i)
        elif gripper[i - 1] == 1 and gripper[i] == 0:
            release_frames.append(i)

    # Base phase with crossed_lift tracking.
    phases: list[str] = []
    grasp_happened = False
    crossed_lift = False

    for i in range(n):
        # Reset on each grasp event.
        if i in grip_frames:
            grasp_happened = True
            crossed_lift = False

        if gripper[i] == 0:
            phases.append("approach")
        elif z[i] > Z_LIFT:
            if grasp_happened:
                crossed_lift = True
            phases.append("transport")
        else:
            # gripper=1, z≤Z_LIFT
            if crossed_lift:
                phases.append("place")
            else:
                phases.append("lift")

    # Override transition windows for grasp/release.
    grip_set = set(grip_frames)
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
    joints = df[list(JOINT_COLS)].values.astype(np.float32)
    n = len(joints)
    idle = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if np.abs(joints[i] - joints[i - 1]).max() < IDLE_THRESHOLD_DEG:
            idle[i] = True
    return idle


def build_keep_mask(idle: np.ndarray) -> np.ndarray:
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
    exclude: tuple[int, ...] = (285,),  # ep285: 14Hz low frame rate
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
    """Convert v8 raw episodes into LeRobot dataset (velocity + idle filter + lift/place fix)."""
    episode_paths = _resolve_episodes(episode_dir, root, exclude)

    output_root = HF_LEROBOT_HOME / repo_id
    if clean and output_root.exists():
        shutil.rmtree(output_root)

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
    phase_counts: dict[str, int] = {p: 0 for p in PHASE_PROMPTS}

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
            print(f"  SKIP {ep.name}: missing columns {missing}")
            skipped += 1
            continue

        source_zone = meta.get("source_zone", "left")
        target_zone = meta.get("target_zone", "right")

        df = df_raw.iloc[:-LAST_FRAME_TRIM].reset_index(drop=True)
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

        phases = classify_phases(df_f)

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
            action = np.concatenate([joints_nxt - joints_cur, [g_nxt - g_cur]])

            hik_chw = _load_image_chw(hik_path, resize)
            zed_chw = _load_image_chw(zed_path, resize)

            phase = phases[t]
            prompt = PHASE_PROMPTS[phase].format(side=source_zone, target_side=target_zone)
            phase_counts[phase] += 1

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

    print(f"\nDone. {len(episode_paths) - skipped} episodes, {total_frames} frames "
          f"(skipped {skipped}). Local root: {dataset.root}")
    if total_raw > 0:
        print(f"Frame retention: {total_frames}/{total_raw} = {100*total_frames/total_raw:.1f}%")

    print("\nPhase distribution:")
    for phase, count in phase_counts.items():
        pct = 100 * count / total_frames if total_frames > 0 else 0
        print(f"  {phase:<12}: {count:6d} ({pct:.1f}%)")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["e6", "magician", "openpi", "v8", "velocity"],
            private=hub_private,
            push_videos=False,
            license="apache-2.0",
        )
        print(f"Pushed to HuggingFace: {repo_id}")


if __name__ == "__main__":
    tyro.cli(main)
