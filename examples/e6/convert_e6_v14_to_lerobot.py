"""
Convert Dobot MagicianE6 v14 raw episodes to LeRobot format for pi05_e6_v14_lora.

v14 design vs v13:
  - Full episode per recording (no sub-episode splitting).
    1 raw recording → 1 LeRobot episode.
  - Episode trimming at conversion time:
      * Leading idle  : drop frames before first joint motion (threshold 0.1 deg).
      * Trailing      : keep up to gripper 1→0 (place event) + PLACE_SETTLE frames.
                        Return-to-home trajectory is discarded.
  - close→open transition is always captured (it is the episode endpoint).
  - task_index stored as anchor (0=left, 3=right); per-step variant randomisation
    is done at training time via task_group_map={0:(0,1,2), 3:(3,4,5)}.
  - dtype=image  (frame-independent PNG, no inter-frame artefacts).
  - action = 7D velocity delta [Δj1..Δj6, Δgripper]; E6Inputs inserts dummy j7=0
    at training/inference time to align with pi05_base DROID 8D state format.

tasks.jsonl (6 entries, 2 anchor groups):
  0: "pick up the orange box from the left side and place it on the right side"
  1: "move the orange box from the left to the right"
  2: "grasp the orange box on the left and put it down on the right"
  3: "pick up the orange box from the right side and place it on the left side"
  4: "move the orange box from the right to the left"
  5: "grasp the orange box on the right and put it down on the left"

Usage:
    uv run examples/e6/convert_e6_v14_to_lerobot.py \\
      --root "/media/billy/새 볼륨2/Dobot/2CAM-Orange-init"

    uv run examples/e6/convert_e6_v14_to_lerobot.py \\
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

DEFAULT_REPO_ID = "Kyle-Riss/dobot_e6_pick_place_orange_v14"

JOINT_COLS = ("j1", "j2", "j3", "j4", "j5", "j6")
GRIPPER_COL = "gripper_tooldo1"
IMAGE_COL_HIK = "image_path_hik"
IMAGE_COL_ZED = "image_path_zed"

# Leading idle: drop frames whose max joint delta from previous frame < threshold.
MOTION_THRESHOLD_DEG = 0.1
# Trailing: keep this many frames after the gripper 1→0 (place) event.
PLACE_SETTLE = 10

# Left→right anchor=0, right→left anchor=3.
# task_group_map={0:(0,1,2), 3:(3,4,5)} applied at training time (config.py).
DIRECTION_ANCHOR: dict[str, int] = {"left": 0, "right": 3}
TASKS: list[str] = [
    "pick up the orange box from the left side and place it on the right side",
    "move the orange box from the left to the right",
    "grasp the orange box on the left and put it down on the right",
    "pick up the orange box from the right side and place it on the left side",
    "move the orange box from the right to the left",
    "grasp the orange box on the right and put it down on the left",
]


# ---------------------------------------------------------------------------
# Trimming helpers
# ---------------------------------------------------------------------------

def find_first_motion(df: pd.DataFrame) -> int:
    """Return index of the first frame where any joint moves >= MOTION_THRESHOLD_DEG."""
    joints = df[list(JOINT_COLS)].values.astype(np.float32)
    for i in range(1, len(joints)):
        if np.abs(joints[i] - joints[i - 1]).max() >= MOTION_THRESHOLD_DEG:
            return i
    return 0


def find_open_idx(gripper: np.ndarray) -> int | None:
    """Return first frame index where gripper transitions 1→0 (place event)."""
    for i in range(1, len(gripper)):
        if gripper[i - 1] > 0.5 and gripper[i] < 0.5:
            return i
    return None


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


def patch_gripper_norm_stats(norm_stats_path: Path, align_droid_state: bool = True) -> None:
    """Force gripper norm_stats to known safe values.

    With align_droid_state=True (v14+), gripper is at index 7 in the 8D
    DROID-aligned format.  With align_droid_state=False (v1-v13 legacy),
    gripper is at index 6 in the 7D format.
    """
    gripper_idx = 7 if align_droid_state else 6
    with norm_stats_path.open("r") as f:
        stats = json.load(f)
    # norm_stats may be wrapped under a 'norm_stats' key (openpi v2 format)
    # or flat (openpi v1 format).
    ns = stats.get("norm_stats", stats)
    acts_key = "actions" if "actions" in ns else "action"
    ns[acts_key]["q01"][gripper_idx] = -1.0
    ns[acts_key]["q99"][gripper_idx] = 1.0
    if "state" in ns:
        ns["state"]["q01"][gripper_idx] = 0.0
        ns["state"]["q99"][gripper_idx] = 1.0
    with norm_stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Gripper norm_stats patched (idx={gripper_idx}): {norm_stats_path}")


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
            "names": ["dj1", "dj2", "dj3", "dj4", "dj5", "dj6", "dgripper"],
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

    # Pre-register all 6 task strings so their indices match TASKS list positions (0-5).
    # task_group_map={0:(0,1,2), 3:(3,4,5)} at training time requires indices 0-5 to exist.
    for task_str in TASKS:
        dataset.meta.add_task(task_str)

    total_frames = 0
    skipped_eps = 0
    direction_counts: dict[str, int] = {"left": 0, "right": 0}

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
        if source_zone not in DIRECTION_ANCHOR:
            print(f"  SKIP {ep.name}: unknown source_zone={source_zone!r}")
            skipped_eps += 1
            continue

        gripper_vals = df_raw[GRIPPER_COL].values.astype(np.float32)

        # --- Trailing trim: find place event (gripper 1→0) ---
        open_idx = find_open_idx(gripper_vals)
        if open_idx is None:
            print(f"  SKIP {ep.name}: no gripper 1→0 transition found")
            skipped_eps += 1
            continue

        trail_end = min(open_idx + PLACE_SETTLE, len(df_raw))
        df = df_raw.iloc[:trail_end].reset_index(drop=True)

        # --- Leading trim: find first frame with actual motion ---
        first_motion = find_first_motion(df)
        start = max(0, first_motion - 1)  # keep 1 frame of context before motion
        df = df.iloc[start:].reset_index(drop=True)

        if len(df) < 16:
            print(f"  SKIP {ep.name}: episode too short after trim ({len(df)} frames)")
            skipped_eps += 1
            continue

        task_index = DIRECTION_ANCHOR[source_zone]
        task_str = TASKS[task_index]

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
            action = np.concatenate([joints_nxt - joints_cur, [g_nxt - g_cur]])

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
            f"place@{open_idx} src={source_zone}"
        )

    print(f"\nDone.")
    print(f"  Raw episodes  : {len(episode_paths)} ({skipped_eps} skipped)")
    print(f"  Episodes saved: {len(episode_paths) - skipped_eps} "
          f"(left={direction_counts['left']}, right={direction_counts['right']})")
    print(f"  Frame pairs   : {total_frames}")
    print(f"  Local root    : {dataset.root}")
    print(f"\nNext steps:")
    print(f"  1. uv run scripts/compute_norm_stats.py --config-name pi05_e6_v14_lora")
    print(f"  2. patch gripper norm_stats if needed (q01=-1.0, q99=1.0 at index 6)")
    print(f"  3. uv run scripts/train.py pi05_e6_v14_lora --exp-name e6_2cam_lora_v14")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["e6", "magician", "openpi", "v14", "velocity", "episode-level-prompt", "droid-aligned"],
            private=hub_private,
            push_videos=False,
            license="apache-2.0",
        )
        print(f"\nPushed to HuggingFace: {repo_id}")


if __name__ == "__main__":
    tyro.cli(main)
