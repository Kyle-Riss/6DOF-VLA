#!/usr/bin/env python3
"""Load one LeRobot frame (same loader settings as openpi) and print shapes.

Use this before training to verify E6 contract vs your actual dataset keys.

Example:

  PYTHONPATH=src uv run python scripts/inspect_e6_lerobot_sample.py \\
    --repo-id your-org/e6_primitive_v1 --action-horizon 16 --index 0

Requires Hugging Face access to the dataset (token if private).
"""

from __future__ import annotations

import numpy as np
import tyro

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset


def _action_key(meta: lerobot_dataset.LeRobotDatasetMetadata) -> str:
    feats = getattr(meta, "features", None) or {}
    if "actions" in feats:
        return "actions"
    if "action" in feats:
        return "action"
    raise ValueError(
        "Could not find 'action' or 'actions' in dataset features. "
        f"Available keys: {sorted(feats.keys())}"
    )


def main(repo_id: str, action_horizon: int = 16, index: int = 0) -> None:
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    ak = _action_key(meta)
    ds = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            ak: [t / meta.fps for t in range(action_horizon)],
        },
    )
    print(f"repo_id={repo_id!r}")
    print(f"fps={meta.fps}  num_frames={len(ds)}  tasks={len(meta.tasks)}")
    print(f"delta_timestamps key: {ak!r} (must match LeRobotE6DataConfig.action_sequence_keys)")
    print(f"camera_keys (images/videos): {meta.camera_keys}")
    if index < 0 or index >= len(ds):
        raise SystemExit(f"index {index} out of range [0, {len(ds) - 1}]")

    row = ds[index]
    print("\n--- top-level keys ---")
    print(sorted(row.keys()))

    # Shapes we care about for E6 v1
    for key in ("observation.state", "action", "actions", "task_index", "timestamp"):
        if key in row:
            v = row[key]
            arr = np.asarray(v)
            print(f"{key}: shape={arr.shape} dtype={arr.dtype}")

    # Nested observation (LeRobot v2+)
    obs = row.get("observation")
    if isinstance(obs, dict):
        print("\n--- observation keys ---")
        for k, v in sorted(obs.items()):
            a = np.asarray(v)
            print(f"  observation.{k}: shape={a.shape} dtype={a.dtype}")

    # Prompt from task (same mapping as training)
    ti = row.get("task_index")
    if ti is not None:
        tid = int(np.asarray(ti).squeeze())
        prompt = meta.tasks.get(tid)
        print(f"\ntask_index={tid} -> prompt={prompt!r}")

    # Heuristic checks (do not fail hard; dataset may differ)
    print("\n--- E6 v1 heuristics (manual confirm) ---")
    st = row.get("observation.state")
    if st is not None:
        s = np.asarray(st).squeeze()
        print(f"observation.state numel={s.size} (expect 7 for E6 v1)")
    act = row.get(ak)
    if act is not None:
        a = np.asarray(act)
        print(f"{ak} chunk shape {a.shape} (expect ({action_horizon}, 7) for E6 v1)")


if __name__ == "__main__":
    tyro.cli(main)
