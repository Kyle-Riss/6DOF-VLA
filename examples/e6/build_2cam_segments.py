#!/usr/bin/env python3
"""
Build episode_primitive_segments.csv for 2CAM Pick-place dataset.

Reads robot_data.csv (columns: z, gripper_tooldo1) and episode_meta.json
(transport_direction) from each numeric episode folder under DATASET_ROOT.

Segmentation rules (same as build_episode_primitive_segments.py v1):
  - init_hold : frame 0 .. first motion frame - 1  (max INIT_HOLD_MAX)
  - approach  : init_hold_end+1 .. first low-Z run start - 1
  - pick      : first low-Z run  (z in [Z_MIN_VALID, Z_THR] for >= MIN_RUN frames)
  - move      : between first and second low-Z run
  - place     : second low-Z run
  - return    : after place .. end

Output: docs/gate2/episode_primitive_segments.csv
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

# --- tuning constants (keep in sync with team spec) ---
Z_THR: float = 112.0
Z_MIN_VALID: float = 1.0
MIN_RUN: int = 2
INIT_HOLD_MAX: int = 25
MOTION_MM: float = 3.0

SEGMENT_ORDER = ("init_hold", "approach", "pick", "move", "place")

TRANSPORT_MAP = {
    "move_left": "move_left",
    "move_right": "move_right",
    "move_to_middle": "move_to_middle",
}


def _low_z_runs(z: list[float]) -> list[tuple[int, int]]:
    valid = [Z_THR >= v >= Z_MIN_VALID for v in z]
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(z):
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < len(z) and valid[j]:
            j += 1
        if j - i >= MIN_RUN:
            runs.append((i, j - 1))
        i = j
    return runs


def _init_hold_end(rows: list[dict], *, max_frames: int = INIT_HOLD_MAX) -> int:
    if not rows:
        return 0
    x0, y0, z0 = float(rows[0]["x"]), float(rows[0]["y"]), float(rows[0]["z"])
    for i in range(1, len(rows)):
        if i >= max_frames:
            return max_frames - 1
        x, y, z = float(rows[i]["x"]), float(rows[i]["y"]), float(rows[i]["z"])
        if math.hypot(x - x0, y - y0) > MOTION_MM or abs(z - z0) > MOTION_MM:
            return i - 1
    return min(len(rows) - 1, max_frames - 1)


def _seg(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a >= 0 and b >= a else (-1, -1)


def process_episode(ep_dir: Path) -> list[dict] | None:
    csv_path = ep_dir / "robot_data.csv"
    meta_path = ep_dir / "episode_meta.json"
    if not csv_path.is_file():
        return None

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        return None

    transport_primitive = "move_unknown"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        td = meta.get("transport_direction", "")
        transport_primitive = TRANSPORT_MAP.get(td, "move_unknown")

    z = [float(r["z"]) for r in rows]
    n = len(rows)
    last = n - 1

    ih_end = _init_hold_end(rows)
    runs = _low_z_runs(z)

    if len(runs) >= 2:
        p0, p1 = runs[0]
        q0, q1 = runs[1]
        # Merge return into place: "place" covers lowering + arm retraction.
        segs = {
            "init_hold": _seg(0, ih_end),
            "approach":  _seg(ih_end + 1, p0 - 1),
            "pick":      _seg(p0, p1),
            "move":      _seg(p1 + 1, q0 - 1),
            "place":     _seg(q0, last),
        }
        status = "ok"
    elif len(runs) == 1:
        p0, p1 = runs[0]
        segs = {
            "init_hold": _seg(0, ih_end),
            "approach":  _seg(ih_end + 1, p0 - 1),
            "pick":      _seg(p0, p1),
            "move":      (-1, -1),
            "place":     (-1, -1),
            "return":    _seg(p1 + 1, last),
        }
        status = "single_z_run_only"
    else:
        segs = {
            "init_hold": _seg(0, ih_end),
            "approach":  _seg(ih_end + 1, last),
            "pick":      (-1, -1),
            "move":      (-1, -1),
            "place":     (-1, -1),
            "return":    (-1, -1),
        }
        status = "no_z_runs"

    folder = int(ep_dir.name)
    out_rows = []
    for seg in SEGMENT_ORDER:
        a, b = segs[seg]
        if a < 0:
            continue
        n_frames = b - a + 1
        out_rows.append({
            "episode_folder": folder,
            "segment": seg,
            "start_frame": a,
            "end_frame": b,
            "n_frames": n_frames,
            "start_image": f"hik/frame_{a:06d}.jpg",
            "end_image": f"hik/frame_{b:06d}.jpg",
            "direction_group": "",
            "transport_primitive": transport_primitive if seg in ("move", "place") else transport_primitive,
            "segment_status": status,
        })
    return out_rows


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = repo_root / "data" / "Pick-place-2CAM"
    out_path = repo_root / "docs" / "gate2" / "episode_primitive_segments.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eps = sorted(
        (d for d in dataset_root.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )

    fieldnames = [
        "episode_folder", "segment", "start_frame", "end_frame",
        "n_frames", "start_image", "end_image",
        "direction_group", "transport_primitive", "segment_status",
    ]

    total_eps = 0
    skipped = []
    status_counts: dict[str, int] = {}

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep_dir in eps:
            rows = process_episode(ep_dir)
            if rows is None:
                skipped.append(ep_dir.name)
                continue
            for row in rows:
                writer.writerow(row)
            total_eps += 1
            st = rows[0]["segment_status"] if rows else "empty"
            status_counts[st] = status_counts.get(st, 0) + 1

    print(f"Done. Episodes processed: {total_eps}, skipped: {skipped}")
    print(f"Status breakdown: {status_counts}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
