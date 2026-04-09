"""
Convert Dobot Magician E6 (or compatible) raw episode folders to LeRobot format for openpi E6 v1.

Expected layout per episode directory::

    episode_dir/
      images/              # frame images (e.g. frame_000000.jpg)
      robot_data.csv       # columns include j1..j6, gripper_tooldo1, image_path, timestamp, ...
      metadata.txt         # optional (not parsed by default)

Per-frame contract (matches openpi ``LeRobotE6DataConfig`` + ``E6Inputs``):

* ``exterior_image_1_left``: top camera, dtype image (CHW uint8 after load)
* ``state``: (7,) float32 — [j1..j6, gripper_tooldo1]
* ``action``: (7,) float32 — [Δj1..Δj6, gripper(t+1)] using rows t and t+1
* ``task``: natural-language string per frame, or one string for all frames when not using
  ``--primitive-v1`` (see below).

The last CSV row is dropped (no t+1 target). Frames = len(csv) - 1.

**Primitive v1 (recommended):** pass ``--primitive-v1`` and ``--segments-csv`` pointing to
``episode_primitive_segments.csv`` (from ``build_episode_primitive_segments.py``). Then each
training frame gets a canonical task from ``e6_v1_task_contract.py`` (approach / pick / move /
place only; boundary frames dropped). Requires episode folder names to be numeric (episode id).

Usage (from openpi repo root). Default ``--repo-id`` matches ``pi05_e6_v1`` in ``training/config.py``::

    uv run examples/e6/convert_e6_episode_to_lerobot.py \\
      --episode-dir /path/to/episode/1 \\
      --task \"approach red object\"

Override the dataset name if needed::

    uv run examples/e6/convert_e6_episode_to_lerobot.py \\
      --repo-id namespace/other_dataset \\
      --episode-dir /path/to/episode/1 \\
      --task \"approach red object\"

Multiple episodes in one run::

    uv run examples/e6/convert_e6_episode_to_lerobot.py \\
      --episode-dir /path/to/ep1 /path/to/ep2 \\
      --task \"align above red object\"

Push to Hugging Face Hub (same pattern as ``examples/libero/convert_libero_data_to_lerobot.py``)::

    uv run examples/e6/convert_e6_episode_to_lerobot.py ... --push-to-hub

The dataset is written under ``$HF_LEROBOT_HOME`` (see LeRobot docs), typically ``~/.cache/huggingface/lerobot/<repo_id>``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
import tyro

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

_E6_DIR = Path(__file__).resolve().parent
if str(_E6_DIR) not in sys.path:
    sys.path.insert(0, str(_E6_DIR))
import e6_v1_task_contract as _v1

# Must match ``TrainConfig(pi05_e6_v1).data.repo_id`` so training loads this LeRobot dataset.
DEFAULT_REPO_ID = "billy/dobot_e6_pick_place_random_v1"

# Default CSV columns (override with --joint-cols / --gripper-col / --image-col if needed)
JOINT_COLS_DEFAULT = ("j1", "j2", "j3", "j4", "j5", "j6")
GRIPPER_COL_DEFAULT = "gripper_tooldo1"
IMAGE_COL_DEFAULT = "image_path"


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
        raise ValueError(f"Need at least 2 rows in {csv_path} to build delta actions.")
    return df


def main(
    episode_dir: list[Path],
    *,
    repo_id: str = DEFAULT_REPO_ID,
    task: str | None = None,
    primitive_v1: bool = False,
    segments_csv: Path | None = None,
    object_phrase: str = _v1.OBJECT_PHRASE_DEFAULT,
    csv_name: str = "robot_data.csv",
    images_subdir: str = "images",
    joint_cols: tuple[str, ...] = JOINT_COLS_DEFAULT,
    gripper_col: str = GRIPPER_COL_DEFAULT,
    image_col: str = IMAGE_COL_DEFAULT,
    fps: int = 20,
    robot_type: str = "magician_e6",
    resize: int | None = 224,
    clean: bool = True,
    push_to_hub: bool = False,
    hub_private: bool = False,
    image_writer_threads: int = 4,
    image_writer_processes: int = 0,
) -> None:
    """Convert one or more episode directories into a single LeRobot dataset."""
    if primitive_v1:
        if segments_csv is None:
            raise ValueError("--segments-csv is required when --primitive-v1 is set.")
    elif not (task and task.strip()):
        raise ValueError("Provide --task for constant-task conversion, or use --primitive-v1 with --segments-csv.")

    episode_paths = [Path(p).resolve() for p in episode_dir]
    for p in episode_paths:
        if not p.is_dir():
            raise NotADirectoryError(p)
        if primitive_v1 and not p.name.isdigit():
            raise ValueError(f"--primitive-v1 requires numeric episode folder names, got: {p.name!r}")

    output_root = HF_LEROBOT_HOME / repo_id
    if clean and output_root.exists():
        shutil.rmtree(output_root)

    # Infer image shape from first frame of first episode
    df0 = _read_episode_csv(episode_paths[0], csv_name)
    first_img = episode_paths[0] / images_subdir / str(df0[image_col].iloc[0])
    sample = _load_image_chw(first_img, resize=resize)
    _, h, w = sample.shape

    features = {
        "exterior_image_1_left": {
            "dtype": "image",
            "shape": (3, h, w),
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
            "names": ["d_j1", "d_j2", "d_j3", "d_j4", "d_j5", "d_j6", "gripper_cmd"],
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

    for ep_idx, ep in enumerate(episode_paths):
        df = _read_episode_csv(ep, csv_name)
        missing = [c for c in [*joint_cols, gripper_col, image_col] if c not in df.columns]
        if missing:
            raise ValueError(f"{ep}: CSV missing columns {missing}. Found: {list(df.columns)}")

        n = len(df)
        frame_task_map: dict[int, str] | None = None
        if primitive_v1:
            assert segments_csv is not None
            raw_ranges, transport = _v1.load_v1_ranges_and_transport(segments_csv, int(ep.name))
            dropped = _v1.apply_v1_boundary_drops(raw_ranges, k=_v1.BOUNDARY_DROP_FRAMES_PER_SIDE)
            frame_task_map = _v1.frame_to_task_map(
                ranges_after_drop=dropped,
                transport_primitive=transport,
                object_phrase=object_phrase,
            )

        n_added = 0
        for t in range(n - 1):
            if primitive_v1:
                assert frame_task_map is not None
                if t not in frame_task_map:
                    continue
                task_str = frame_task_map[t]
            else:
                task_str = task  # type: ignore[assignment]

            cur = df.iloc[t]
            nxt = df.iloc[t + 1]
            img_path = ep / images_subdir / str(cur[image_col])
            if not img_path.is_file():
                raise FileNotFoundError(f"Missing image {img_path}")

            joints_cur = np.array([float(cur[c]) for c in joint_cols], dtype=np.float32)
            joints_nxt = np.array([float(nxt[c]) for c in joint_cols], dtype=np.float32)
            g_cur = np.float32(cur[gripper_col])
            g_nxt = np.float32(nxt[gripper_col])
            state = np.concatenate([joints_cur, np.array([g_cur], dtype=np.float32)])
            delta = joints_nxt - joints_cur
            action = np.concatenate([delta, np.array([g_nxt], dtype=np.float32)])

            img_chw = _load_image_chw(img_path, resize=resize)
            if img_chw.shape != (3, h, w):
                raise ValueError(
                    f"Image shape mismatch in {img_path}: {img_chw.shape} vs (3, {h}, {w}). "
                    "Use a fixed --resize or consistent camera resolution."
                )

            frame = {
                "exterior_image_1_left": img_chw,
                "state": state,
                "action": action,
                "task": task_str,
            }
            dataset.add_frame(frame)
            n_added += 1

        dataset.save_episode()
        if primitive_v1:
            print(f"Saved episode index {ep_idx} from {ep} ({n_added} frames after v1 primitive filter, csv rows n={n}).")
        else:
            print(f"Saved episode index {ep_idx} from {ep} ({n - 1} frames).")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["e6", "magician", "openpi"],
            private=hub_private,
            push_videos=False,
            license="apache-2.0",
        )

    print(f"Done. Local dataset root: {dataset.root}")


if __name__ == "__main__":
    tyro.cli(main)
