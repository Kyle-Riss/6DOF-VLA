"""
Convert Dobot Magician E6 v2 raw episode folders to LeRobot format for pi05_e6_v2_lora.

v1 vs v2 differences (intentional separation, do NOT merge with v1 dataset):

* fps: 16Hz (v1: 18Hz)
* prompt: episode-level single string from ``episode_meta.json["prompt"]`` (v1: segment-level)
* images: already 224x224 (v1: original captured size, resized at convert time)
* default repo: ``kyle-riss/dobot_e6_pick_place_orange_v2`` (v1: ``billy/dobot_e6_pick_place_random_v1``)
* state/action: 7D ``[j1..j6, gripper_tooldo1]`` (same contract as v1 — vacuum gripper binary 0/1)

Per-frame contract (matches openpi ``LeRobotE6DataConfig`` + ``E6Inputs``):

* ``exterior_image_1_left``: HIK side camera (whole arm visible), CHW uint8
* ``exterior_image_2_left``: ZED side camera (object-focused), CHW uint8
* ``state``: (7,) float32 — [j1..j6, gripper_tooldo1] at t
* ``action``: (7,) float32 — [j1..j6, gripper_tooldo1] at t+1 (absolute next-position, NOT delta)
* ``task``: ``episode_meta.json["prompt"]`` (same string for every frame in one episode)

Last CSV row dropped (no t+1 target). Frames per episode = len(csv) - 1.

Expected raw layout per episode directory::

    episode_dir/
      images/hik/frame_XXXXXX.jpg
      images/zed/frame_XXXXXX.jpg
      robot_data.csv     # j1..j6, gripper_tooldo1, image_path_hik, image_path_zed, ...
      episode_meta.json  # prompt, source_zone, target_zone, ...

Usage (single episode, explicit task — for sanity checks)::

    uv run examples/e6/convert_e6_v2_to_lerobot.py \\
      --episode-dir "/media/billy/새 볼륨/Dobot/2CAM-Orange/1"

Bulk: scan a root and convert every numeric subfolder (skip ``--exclude`` IDs)::

    uv run examples/e6/convert_e6_v2_to_lerobot.py \\
      --root "/media/billy/새 볼륨/Dobot/2CAM-Orange" \\
      --exclude 382

Push to Hugging Face Hub::

    uv run examples/e6/convert_e6_v2_to_lerobot.py \\
      --root "/media/billy/새 볼륨/Dobot/2CAM-Orange" \\
      --exclude 382 \\
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

DEFAULT_REPO_ID = "kyle-riss/dobot_e6_pick_place_orange_v2"

JOINT_COLS_DEFAULT = ("j1", "j2", "j3", "j4", "j5", "j6")
GRIPPER_COL_DEFAULT = "gripper_tooldo1"
IMAGE_COL_HIK_DEFAULT = "image_path_hik"
IMAGE_COL_ZED_DEFAULT = "image_path_zed"


def _load_image_chw(path: Path, resize: int | None) -> np.ndarray:
    img = PIL.Image.open(path).convert("RGB")
    if resize is not None:
        img = img.resize((resize, resize), PIL.Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image at {path}, got {arr.shape}")
    return np.transpose(arr, (2, 0, 1))


def _read_episode_csv(episode_dir: Path, csv_name: str) -> pd.DataFrame:
    csv_path = episode_dir / csv_name
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}")
    df = pd.read_csv(csv_path)
    if len(df) < 2:
        raise ValueError(f"Need at least 2 rows in {csv_path} to build (t, t+1) pairs.")
    return df


def _read_episode_prompt(episode_dir: Path, meta_name: str = "episode_meta.json") -> str:
    meta_path = episode_dir / meta_name
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    prompt = meta.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{meta_path}: 'prompt' field missing or empty.")
    return prompt.strip()


def _resolve_episode_paths(
    episode_dir: list[Path] | None,
    root: Path | None,
    exclude: tuple[int, ...],
) -> list[Path]:
    if episode_dir:
        return [Path(p).resolve() for p in episode_dir]
    if root is None:
        raise ValueError("Provide either --episode-dir or --root.")
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    excluded = set(exclude)
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        ep_id = int(child.name)
        if ep_id in excluded:
            continue
        candidates.append((ep_id, child))
    candidates.sort(key=lambda x: x[0])
    if not candidates:
        raise ValueError(f"No numeric episode subdirs found in {root}")
    return [p for _, p in candidates]


def main(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    root: Path | None = None,
    episode_dir: list[Path] | None = None,
    exclude: tuple[int, ...] = (382,),
    csv_name: str = "robot_data.csv",
    images_subdir: str = "images",
    joint_cols: tuple[str, ...] = JOINT_COLS_DEFAULT,
    gripper_col: str = GRIPPER_COL_DEFAULT,
    image_col_hik: str = IMAGE_COL_HIK_DEFAULT,
    image_col_zed: str = IMAGE_COL_ZED_DEFAULT,
    fps: int = 16,
    robot_type: str = "magician_e6",
    resize: int | None = None,
    clean: bool = True,
    push_to_hub: bool = False,
    hub_private: bool = False,
    image_writer_threads: int = 4,
    image_writer_processes: int = 0,
) -> None:
    """Convert v2 raw episode directories into a single LeRobot dataset.

    ``resize=None`` keeps the on-disk 224x224 images unchanged (v2 stores already-resized).
    Pass an int to force a different size.
    """
    episode_paths = _resolve_episode_paths(episode_dir, root, exclude)

    output_root = HF_LEROBOT_HOME / repo_id
    if clean and output_root.exists():
        shutil.rmtree(output_root)

    df0 = _read_episode_csv(episode_paths[0], csv_name)
    first_hik = episode_paths[0] / images_subdir / str(df0[image_col_hik].iloc[0])
    first_zed = episode_paths[0] / images_subdir / str(df0[image_col_zed].iloc[0])
    sample_hik = _load_image_chw(first_hik, resize=resize)
    sample_zed = _load_image_chw(first_zed, resize=resize)
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
            "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper_cmd"],
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
    for ep_idx, ep in enumerate(episode_paths):
        df = _read_episode_csv(ep, csv_name)
        prompt = _read_episode_prompt(ep)

        missing = [c for c in [*joint_cols, gripper_col, image_col_hik, image_col_zed] if c not in df.columns]
        if missing:
            raise ValueError(f"{ep}: CSV missing columns {missing}. Found: {list(df.columns)}")

        n = len(df)
        n_added = 0
        for t in range(n - 1):
            cur = df.iloc[t]
            nxt = df.iloc[t + 1]
            hik_path = ep / images_subdir / str(cur[image_col_hik])
            zed_path = ep / images_subdir / str(cur[image_col_zed])
            if not hik_path.is_file():
                raise FileNotFoundError(f"Missing HIK image {hik_path}")
            if not zed_path.is_file():
                raise FileNotFoundError(f"Missing ZED image {zed_path}")

            joints_cur = np.array([float(cur[c]) for c in joint_cols], dtype=np.float32)
            joints_nxt = np.array([float(nxt[c]) for c in joint_cols], dtype=np.float32)
            g_cur = np.float32(cur[gripper_col])
            g_nxt = np.float32(nxt[gripper_col])
            state = np.concatenate([joints_cur, np.array([g_cur], dtype=np.float32)])
            action = np.concatenate([joints_nxt, np.array([g_nxt], dtype=np.float32)])

            hik_chw = _load_image_chw(hik_path, resize=resize)
            zed_chw = _load_image_chw(zed_path, resize=resize)
            if hik_chw.shape != (3, h_hik, w_hik):
                raise ValueError(f"HIK shape mismatch {hik_path}: {hik_chw.shape} vs (3,{h_hik},{w_hik})")
            if zed_chw.shape != (3, h_zed, w_zed):
                raise ValueError(f"ZED shape mismatch {zed_path}: {zed_chw.shape} vs (3,{h_zed},{w_zed})")

            frame = {
                "exterior_image_1_left": hik_chw,
                "exterior_image_2_left": zed_chw,
                "state": state,
                "action": action,
                "task": prompt,
            }
            dataset.add_frame(frame)
            n_added += 1

        dataset.save_episode()
        total_frames += n_added
        print(f"[{ep_idx + 1}/{len(episode_paths)}] {ep.name}: {n_added} frames | prompt={prompt!r}")

    print(f"Done. {len(episode_paths)} episodes, {total_frames} frames total. Local root: {dataset.root}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["e6", "magician", "openpi", "v2"],
            private=hub_private,
            push_videos=False,
            license="apache-2.0",
        )
        print(f"Pushed to HuggingFace: {repo_id}")


if __name__ == "__main__":
    tyro.cli(main)
