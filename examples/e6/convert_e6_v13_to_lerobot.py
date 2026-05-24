"""
Convert Dobot Magician E6 v13 raw episode folders to LeRobot format for pi05_e6_v13_lora.

v13 changes vs v10:
* Episode-level prompts (3 variants × 2 directions) — matches DROID pretraining distribution.
* classify_phases() and PHASE_PROMPTS completely removed.
* Idle runs >= MIN_IDLE_RUN split the episode into separate sub-episodes (fixes action_horizon
  discontinuity that occurred when idle frames were silently removed mid-episode).
* task_index stored as GROUP ANCHOR (0=left, 3=right); per-step variant selection done at
  training time via PromptFromLeRobotTask(task_group_map={0:(0,1,2), 3:(3,4,5)}).
  This mirrors DROID's tf.random.shuffle([lang1,lang2,lang3])[0].
* use_videos=True — images stored as mp4 (videos/ folder), parquet holds metadata only.
  Matches droid_100 LeRobot format. Dramatically smaller dataset, faster training I/O.
* next.reward / next.done added — last frame of each sub-episode has reward=1.0, done=True.
  All other frames: reward=0.0, done=False. Matches droid_100 structure.
* Images in HWC uint8 format (height, width, channel) — required for video encoding.
* Source: /media/billy/새 볼륨2/Dobot/2CAM-Orange-init (199 raw, ep193 excluded)
* HF repo: Kyle-Riss/dobot_e6_pick_place_orange_v13

Episode-level contract:
  exterior_image_1_left: HIK camera, HWC uint8, stored as mp4
  exterior_image_2_left: ZED camera, HWC uint8, stored as mp4
  state:       (7,) float32 — [j1..j6, gripper_tooldo1] absolute position
  action:      (7,) float32 — [Δj1..Δj6, Δgripper] velocity delta
  next.reward: float32 — 1.0 on last frame, 0.0 otherwise
  next.done:   bool   — True on last frame, False otherwise
  task:        anchor instruction (task_index 0 or 3); variants selected at training time

tasks.jsonl (6 entries):
  0: "pick up the orange box from the left side and place it on the right side"  ← left anchor
  1: "move the orange box from the left to the right"
  2: "grasp the orange box on the left and put it down on the right"
  3: "pick up the orange box from the right side and place it on the left side"  ← right anchor
  4: "move the orange box from the right to the left"
  5: "grasp the orange box on the right and put it down on the left"

Usage:
    uv run examples/e6/convert_e6_v13_to_lerobot.py \\
      --root "/media/billy/새 볼륨2/Dobot/2CAM-Orange-init"

    uv run examples/e6/convert_e6_v13_to_lerobot.py \\
      --root "/media/billy/새 볼륨2/Dobot/2CAM-Orange-init" \\
      --push-to-hub --hub-private
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
import tyro

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_REPO_ID = "Kyle-Riss/dobot_e6_pick_place_orange_v13"

JOINT_COLS = ("j1", "j2", "j3", "j4", "j5", "j6")
GRIPPER_COL = "gripper_tooldo1"
IMAGE_COL_HIK = "image_path_hik"
IMAGE_COL_ZED = "image_path_zed"

# Episode-level instructions: 3 variants × 2 directions (2CAM-Orange-init has both).
#
# At conversion time, one variant is chosen randomly per sub-episode.
# This gives all 6 task_indices (0-5) in tasks.jsonl and natural per-episode
# language diversity — matching DROID's augmentation at the data level.
# No task_group_map needed at training time.
INSTRUCTIONS: list[str] = [
    # left→right variants (task_index 0, 1, 2):
    "pick up the orange box from the left side and place it on the right side",
    "move the orange box from the left to the right",
    "grasp the orange box on the left and put it down on the right",
    # right→left variants (task_index 3, 4, 5):
    "pick up the orange box from the right side and place it on the left side",
    "move the orange box from the right to the left",
    "grasp the orange box on the right and put it down on the left",
]

# Variant index ranges per direction; one is chosen randomly per sub-episode.
DIRECTION_VARIANTS: dict[str, list[int]] = {
    "left": [0, 1, 2],
    "right": [3, 4, 5],
}

IDLE_THRESHOLD_DEG = 0.05
MIN_IDLE_RUN = 5
LAST_FRAME_TRIM = 15


# ---------------------------------------------------------------------------
# Idle segmentation
# ---------------------------------------------------------------------------

def compute_idle_mask(df: pd.DataFrame) -> np.ndarray:
    joints = df[list(JOINT_COLS)].values.astype(np.float32)
    n = len(joints)
    idle = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if np.abs(joints[i] - joints[i - 1]).max() < IDLE_THRESHOLD_DEG:
            idle[i] = True
    return idle


def get_active_segments(idle: np.ndarray, gripper: np.ndarray) -> list[tuple[int, int]]:
    """
    Return half-open (start, end) ranges of contiguous active frames.

    Idle runs >= MIN_IDLE_RUN terminate the current sub-episode and start a new
    one after the idle block, so the action horizon never spans a removed gap.
    Short idle runs (< MIN_IDLE_RUN) are treated as noisy active frames and kept.

    If a gripper state transition (0→1 or 1→0) occurs within an idle run, the
    segment end is extended to include that transition frame pair. This ensures
    the gripper delta action (±1.0) appears in the training data instead of being
    silently dropped at the idle boundary (e.g. gripper release after placing).
    """
    n = len(idle)
    segments: list[tuple[int, int]] = []
    seg_start: int | None = None
    i = 0

    while i < n:
        if not idle[i]:
            if seg_start is None:
                seg_start = i
            i += 1
        else:
            run_end = i
            while run_end < n and idle[run_end]:
                run_end += 1
            run_len = run_end - i
            if run_len >= MIN_IDLE_RUN:
                if seg_start is not None:
                    # Scan the idle run for a gripper state transition.
                    # If found at frame k (gripper[k] != gripper[k+1]), extend
                    # the segment end to k+2 so the action pair (k, k+1) is
                    # included and delta_gripper (±1.0) is captured.
                    seg_end = i
                    for k in range(i, run_end - 1):
                        if gripper[k] != gripper[k + 1]:
                            seg_end = k + 2
                            break
                    segments.append((seg_start, seg_end))
                    seg_start = None
                i = run_end
            else:
                i = run_end

    if seg_start is not None:
        segments.append((seg_start, n))

    return segments


# ---------------------------------------------------------------------------
# Image loading — HWC uint8, required for LeRobot video encoding
# ---------------------------------------------------------------------------

def _load_image_hwc(path: Path, resize: int | None) -> np.ndarray:
    img = PIL.Image.open(path).convert("RGB")
    if resize is not None:
        img = img.resize((resize, resize), PIL.Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)  # (H, W, 3)


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


def patch_gripper_norm_stats(norm_stats_path: Path) -> None:
    """Force gripper (index 6) norm_stats to known safe values."""
    with norm_stats_path.open("r") as f:
        stats = json.load(f)
    stats["action"]["q01"][6] = -1.0
    stats["action"]["q99"][6] = 1.0
    if "state" in stats:
        stats["state"]["q01"][6] = 0.0
        stats["state"]["q99"][6] = 1.0
    with norm_stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Gripper norm_stats patched: {norm_stats_path}")


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
    """Convert v13 raw episodes: video storage, episode-level prompts, 3-variant language augmentation."""
    episode_paths = _resolve_episodes(episode_dir, root, exclude)

    output_root = HF_LEROBOT_HOME / repo_id
    if clean and output_root.exists():
        shutil.rmtree(output_root)

    # Sample first episode to get image dimensions (HWC)
    df0 = _read_csv(episode_paths[0], csv_name)
    hik0 = episode_paths[0] / images_subdir / str(df0[IMAGE_COL_HIK].iloc[0])
    zed0 = episode_paths[0] / images_subdir / str(df0[IMAGE_COL_ZED].iloc[0])
    sample_hik = _load_image_hwc(hik0, resize)
    sample_zed = _load_image_hwc(zed0, resize)
    h_hik, w_hik, _ = sample_hik.shape
    h_zed, w_zed, _ = sample_zed.shape

    features = {
        # HWC format, dtype="video" — LeRobot encodes to mp4 during save_episode()
        "exterior_image_1_left": {
            "dtype": "video",
            "shape": (h_hik, w_hik, 3),
            "names": ["height", "width", "channel"],
        },
        "exterior_image_2_left": {
            "dtype": "video",
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
        # Match droid_100 structure: last frame of each sub-episode = success
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=True,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    total_frames = 0
    total_subepisodes = 0
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

        missing = [
            c for c in [*JOINT_COLS, GRIPPER_COL, IMAGE_COL_HIK, IMAGE_COL_ZED]
            if c not in df_raw.columns
        ]
        if missing:
            print(f"  SKIP {ep.name}: missing columns {missing}")
            skipped_eps += 1
            continue

        source_zone = meta.get("source_zone", "").strip().lower()
        if source_zone not in DIRECTION_VARIANTS:
            print(f"  SKIP {ep.name}: unknown source_zone={source_zone!r}")
            skipped_eps += 1
            continue

        # Trim last frames (end-of-episode return jitter), then segment by idle
        df = df_raw.iloc[:-LAST_FRAME_TRIM].reset_index(drop=True)
        idle = compute_idle_mask(df)
        gripper_vals = df[GRIPPER_COL].values.astype(np.float32)
        segments = get_active_segments(idle, gripper_vals)

        raw_n = len(df_raw)
        ep_sub = 0
        ep_frames = 0

        for seg_start, seg_end in segments:
            df_seg = df.iloc[seg_start:seg_end].reset_index(drop=True)
            seg_len = len(df_seg)

            if seg_len < 16:
                continue

            # Randomly pick one of the 3 direction-specific variants per sub-episode.
            task_str = INSTRUCTIONS[random.choice(DIRECTION_VARIANTS[source_zone])]

            n_frames = seg_len - 1  # action pairs
            n_added = 0
            for t in range(n_frames):
                cur = df_seg.iloc[t]
                nxt = df_seg.iloc[t + 1]

                hik_path = ep / images_subdir / str(cur[IMAGE_COL_HIK])
                zed_path = ep / images_subdir / str(cur[IMAGE_COL_ZED])
                if not hik_path.is_file():
                    raise FileNotFoundError(f"Missing HIK: {hik_path}")
                if not zed_path.is_file():
                    raise FileNotFoundError(f"Missing ZED: {zed_path}")

                joints_cur = np.array([float(cur[c]) for c in JOINT_COLS], dtype=np.float32)
                joints_nxt = np.array([float(nxt[c]) for c in JOINT_COLS], dtype=np.float32)
                g_cur = np.float32(cur[GRIPPER_COL])
                g_nxt = np.float32(nxt[GRIPPER_COL])

                state = np.concatenate([joints_cur, [g_cur]])
                action = np.concatenate([joints_nxt - joints_cur, [g_nxt - g_cur]])

                is_last = (t == n_frames - 1)

                dataset.add_frame({
                    "exterior_image_1_left": _load_image_hwc(hik_path, resize),
                    "exterior_image_2_left": _load_image_hwc(zed_path, resize),
                    "state": state,
                    "action": action,
                    "next.reward": np.array([1.0 if is_last else 0.0], dtype=np.float32),
                    "next.done": np.array([is_last], dtype=bool),
                    "task": task_str,
                })
                n_added += 1

            dataset.save_episode()
            ep_sub += 1
            ep_frames += n_added

        total_frames += ep_frames
        total_subepisodes += ep_sub
        direction_counts[source_zone] += ep_sub

        seg_info = f"{ep_sub} sub-ep(s)" if ep_sub != 1 else "1 sub-ep"
        print(
            f"[{ep_idx + 1:3d}/{len(episode_paths)}] ep={ep.name:>4s}: "
            f"raw={raw_n} → {seg_info}, {ep_frames} pairs | src={source_zone}"
        )

    print(f"\nDone.")
    print(f"  Raw episodes : {len(episode_paths)} ({skipped_eps} skipped)")
    print(f"  Sub-episodes : {total_subepisodes} "
          f"(left→right={direction_counts['left']}, right→left={direction_counts['right']})")
    print(f"  Frame pairs  : {total_frames}")
    print(f"  Local root   : {dataset.root}")
    print(f"\nNext steps:")
    print(f"  1. Verify: cat {dataset.root}/meta/tasks.jsonl  (expect 6 entries)")
    print(f"  2. Verify: cat {dataset.root}/meta/info.json    (dtype=video, use_videos=true)")
    print(f"  3. Compute norm_stats:")
    print(f"     uv run scripts/compute_norm_stats.py --config-name pi05_e6_v13_lora")
    print(f"  4. Patch gripper norm_stats if q01/q99 ≠ -1/+1:")
    print(f"     python -c \"from examples.e6.convert_e6_v13_to_lerobot import patch_gripper_norm_stats; ...\"")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["e6", "magician", "openpi", "v13", "velocity", "episode-level-prompt"],
            private=hub_private,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"\nPushed to HuggingFace: {repo_id}")


if __name__ == "__main__":
    tyro.cli(main)
