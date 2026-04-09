#!/usr/bin/env python3
"""Export consecutive frames from a LeRobot E6 dataset for primitive/task sync review.

Writes PNGs to a folder under ``preview_e6_primitive_sync/`` with filenames:
  ep{E}_fr{local:04d}_g{global:05d}_ti{ti}_{sanitized_task}.png

Example::

  ./.venv/bin/python scripts/preview_e6_consecutive_frames.py \\
    --repo-id billy/dobot_e6_pick_place_random_v1 \\
    --episode-index 0 --start-frame 0 --num-frames 48
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
import tyro
from PIL import Image

import lerobot.common.datasets.lerobot_dataset as ld


def _slug(s: str, max_len: int = 60) -> str:
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s).strip("_")
    return s[:max_len] or "task"


def main(
    repo_id: str = "billy/dobot_e6_pick_place_random_v1",
    *,
    episode_index: int = 0,
    start_frame: int = 0,
    num_frames: int = 48,
    out_dir: Path | None = None,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = out_dir or (repo_root / "preview_e6_primitive_sync" / f"consecutive_ep{episode_index:03d}_fr{start_frame:04d}_n{num_frames:03d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = ld.LeRobotDatasetMetadata(repo_id)
    ds = ld.LeRobotDataset(repo_id, delta_timestamps=None)

    # Only scan this episode's global index range (fast; do not iterate whole dataset)
    ep_from = int(ds.episode_data_index["from"][episode_index].item())
    ep_to = int(ds.episode_data_index["to"][episode_index].item())  # exclusive
    pairs: list[tuple[int, int, int]] = []
    for g in range(ep_from, ep_to):
        row = ds[g]
        fi = int(np.asarray(row["frame_index"]).squeeze())
        pairs.append((fi, g, int(np.asarray(row["task_index"]).squeeze())))
    pairs.sort(key=lambda x: x[0])

    want = set(range(start_frame, start_frame + num_frames))
    picked = [(fi, g, ti) for fi, g, ti in pairs if fi in want]
    if len(picked) < len(want):
        have = {fi for fi, _, _ in picked}
        missing = sorted(want - have)
        print(
            f"Warning: episode {episode_index} missing local frames {missing[:20]}"
            f"{'...' if len(missing) > 20 else ''} (v1 filter removed them). "
            f"Exporting {len(picked)} / {len(want)} requested.",
            flush=True,
        )

    for fi, g, ti in sorted(picked, key=lambda x: x[0]):
        row = ds[g]
        prompt = meta.tasks[ti]
        img = row.get("exterior_image_1_left")
        if img is None:
            obs = row.get("observation")
            if isinstance(obs, dict):
                for k in obs:
                    if "exterior" in k:
                        img = obs[k]
                        break
        if img is None:
            raise RuntimeError(f"No image at global index {g}")
        arr = img.cpu().numpy() if torch.is_tensor(img) else np.asarray(img)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        fn = f"ep{episode_index:03d}_fr{fi:04d}_g{g:05d}_ti{ti}_{_slug(prompt)}.png"
        Image.fromarray(arr).save(out_dir / fn)
        print(f"{fn}  {prompt!r}")

    print(f"\nWrote {len(picked)} images to: {out_dir}")


if __name__ == "__main__":
    tyro.cli(main)
