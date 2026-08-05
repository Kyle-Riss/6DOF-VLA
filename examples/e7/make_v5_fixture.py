"""Synthesise a schema-v5 episode so the conversion path can be exercised without a robot.

The point is not to produce plausible motion — it is to produce a file layout and a
metadata block that match what the collector now emits, so that the converter's v5
handling is run before anyone spends time teleoperating a probe episode. A bug found
here costs nothing; the same bug found after the probe costs another session at the
robot.

    uv run examples/e7/make_v5_fixture.py --out /tmp/e7_v5_fixture --episodes 3

Every episode it writes is marked ``schema_probe`` in ``book_id``. That marker is
what keeps synthetic frames out of a training set later, and the converter refuses
to convert them unless it is told to.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import tyro
from PIL import Image

PROBE_MARKER = "schema_probe"

# The rig, as the collector reports it. Taught TCP y per shelf waypoint: on this arm
# +y is left, so ascending y is right, centre, left.
SHELF_TCP_Y_MM = {"2": -377.76, "3": -173.36, "4": 9.60}
SHELF_SIDE = {"2": "right", "3": "center", "4": "left"}
SHELF_SIDE_KO = {"right": "오른쪽", "center": "가운데", "left": "왼쪽"}
CATEGORIES = ("science", "liberal_arts", "humanities")

# Written in the collector's order so a diff against a real file lines up.
CSV_COLS = (
    ["frame_id", "timestamp", "timestamp_monotonic",
     "image_path_hik", "image_path_zed", "image_path_label"]
    + [f"j{i}" for i in range(1, 7)]
    + ["x", "y", "z", "rx", "ry", "rz",
       "gripper_command", "gripper_trigger_raw", "teleop_enabled", "robot_mode",
       "error_latched", "error_code_first",
       "timestamp_hik", "timestamp_zed", "timestamp_robot", "joint_state_timestamp",
       "teleop_timestamp", "command_send_timestamp", "camera_skew_ms",
       "dt_from_prev", "frame_dropped_before", "n_commands_in_frame", "last_command_timestamp"]
    + [f"teleop_raw_{i}" for i in range(1, 7)]
    + [f"twist_sent_{i}" for i in range(1, 7)]
    + ["monotonic_t", "error_code", "warn_code", "error_timestamp", "limit_active", "limit_reason"]
    + [f"pre_limit_command_{i}" for i in range(1, 7)]
    + [f"post_limit_command_{i}" for i in range(1, 7)]
    + ["collection_active", "motion_source", "teleop_command_published",
       "active_sequence", "anchor_move_active", "wrist_toggle_active", "motion_source_age_ms"]
)


@dataclasses.dataclass
class Args:
    out: pathlib.Path
    episodes: int = 3
    frames: int = 220
    fps: float = 16.0
    shuffle_layout: bool = True
    """Rotate which category goes to which shelf per episode.

    Left off, every episode shares one layout and the converter's confound warning
    fires — which is itself worth seeing once.
    """


def _episode(root: pathlib.Path, idx: int, args: Args) -> None:
    rng = np.random.default_rng(1000 + idx)
    n, dt = args.frames, 1.0 / args.fps
    ep = root / str(idx)
    for cam in ("hik", "zed", "label"):
        (ep / "images" / cam).mkdir(parents=True, exist_ok=True)

    # Category -> shelf assignment. Rotating it is what makes the sign, rather than
    # the category token, the only thing that could predict the destination.
    rot = idx % 3 if args.shuffle_layout else 0
    wps = ["2", "3", "4"]
    layout = {c: SHELF_SIDE[wps[(i + rot) % 3]] for i, c in enumerate(CATEGORIES)}
    category = CATEGORIES[idx % 3]
    side = layout[category]
    wp = next(w for w, s in SHELF_SIDE.items() if s == side)

    # A slow drift plus noise. j5 is held near its mechanical constant so the
    # normalisation behaves the way the real arm's does.
    t = np.arange(n)
    joints = np.column_stack([
        90 + 12 * np.sin(t / 40) + rng.normal(0, .05, n),
        50 + 10 * np.sin(t / 55) + rng.normal(0, .05, n),
        60 + 18 * np.sin(t / 35) + rng.normal(0, .05, n),
        -22 + 12 * np.sin(t / 45) + rng.normal(0, .05, n),
        -88 + rng.normal(0, .02, n),
        176 + 12 * np.sin(t / 50) + rng.normal(0, .05, n),
    ]).astype(np.float32)

    grip = np.zeros(n, np.float32)
    close_at, open_at = int(n * .35), int(n * .80)
    grip[close_at:open_at] = 0.82                     # absolute aperture, a staircase
    teleop = np.ones(n, np.float32)
    teleop[:24] = 0.0                                  # scripted lead-in
    mode = np.where(teleop > 0, 5, 0).astype(np.int32)

    img = Image.fromarray(rng.integers(60, 190, (224, 224, 3), dtype=np.uint8))
    rows = []
    for k in range(n):
        for cam in ("hik", "zed", "label"):
            img.save(ep / "images" / cam / f"frame_{k:06d}.jpg", quality=88)
        rows.append({
            "frame_id": k, "timestamp": 1785500000.0 + k * dt, "timestamp_monotonic": k * dt,
            "image_path_hik": f"hik/frame_{k:06d}.jpg",
            "image_path_zed": f"zed/frame_{k:06d}.jpg",
            "image_path_label": f"label/frame_{k:06d}.jpg",
            **{f"j{i+1}": joints[k, i] for i in range(6)},
            "x": 300.0, "y": -180.0, "z": 200.0, "rx": 180.0, "ry": 0.0, "rz": 0.0,
            "gripper_command": grip[k], "gripper_trigger_raw": grip[k],
            "teleop_enabled": teleop[k], "robot_mode": mode[k],
            "error_latched": 0, "error_code_first": 0,
            "timestamp_hik": 1785500000.0 + k * dt, "timestamp_zed": 1785500000.0 + k * dt,
            "timestamp_robot": 1785500000.0 + k * dt,
            "joint_state_timestamp": 1785500000.0 + k * dt,
            "teleop_timestamp": 1785500000.0 + k * dt,
            "command_send_timestamp": 1785500000.0 + k * dt,
            "camera_skew_ms": 13.7, "dt_from_prev": 0.0 if k == 0 else dt,   # seconds — confirmed against the collector 2026-08-01
            "frame_dropped_before": 0, "n_commands_in_frame": 2 if teleop[k] else 0,
            "last_command_timestamp": 1785500000.0 + k * dt,
            **{f"teleop_raw_{i}": 0.0 for i in range(1, 7)},
            **{f"twist_sent_{i}": 0.0 for i in range(1, 7)},
            "monotonic_t": k * dt, "error_code": 0, "warn_code": 0, "error_timestamp": 0.0,
            "limit_active": 0, "limit_reason": "",
            **{f"pre_limit_command_{i}": 0.0 for i in range(1, 7)},
            **{f"post_limit_command_{i}": 0.0 for i in range(1, 7)},
            "collection_active": 1,
            "motion_source": "teleop" if teleop[k] else "waypoint_route",
            "teleop_command_published": int(teleop[k]),
            "active_sequence": "" if teleop[k] else "waypoint_route",
            "anchor_move_active": 0, "wrist_toggle_active": 0,
            "motion_source_age_ms": "" if k < 3 else 40.0,   # empty == None, never 0
        })

    import csv
    with (ep / "robot_data.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)

    (ep / "episode_events.csv").write_text(
        "event,frame_id,timestamp\n"
        f"grasp_close,{close_at},{1785500000.0 + close_at * dt}\n"
        f"grasp_open,{open_at},{1785500000.0 + open_at * dt}\n", encoding="utf-8")

    meta = {
        "schema_version": 5,
        "joint_unit": "degree", "control_mode": "cartesian_velocity",
        "command_semantics": "cartesian_twist",
        "fps": args.fps, "num_frames": n,
        "category": category, "category_source": "manual", "category_confirmed": True,
        # The converter refuses to invent an object name -- "approach the
        # science_001" would put a book id straight into the tokenizer -- so the
        # fixture has to carry the same field the collector writes.
        "prompt_object_name": "book",
        "labels_source": "manual",
        # v5 canonical destination. No `target_shelf_id`: that field is gone, and a
        # fixture that still emitted it would test the wrong thing.
        "resolved_target_shelf_id": wp,
        "resolved_target_side": SHELF_SIDE[wp],
        "resolved_target_side_ko": SHELF_SIDE_KO[SHELF_SIDE[wp]],
        "resolved_target_tcp_y_mm": SHELF_TCP_Y_MM[wp],
        "shelf_id_domain": ["2", "3", "4"],
        "target_shelf": SHELF_SIDE[wp],
        "actual_shelf": SHELF_SIDE[wp], "actual_shelf_distance_mm": 42.0,
        "shelf_layout": layout,
        "shelf_layout_physical": dict(SHELF_SIDE),
        "shelf_labels": {wps[(i + rot) % 3]: c for i, c in enumerate(CATEGORIES)},
        "shelf_waypoint_tcp_y_mm": SHELF_TCP_Y_MM,
        "waypoint_file_sha256": "e86b84e9" + "0" * 56,
        "motion_source_clock": "time.monotonic", "motion_source_status_hz": 5.0,
        "motion_source_age_normal_max_ms": 200.0,
        "motion_source_age_null_means": "no_wrist_status_received_yet",
        "motion_source_domain": ["teleop", "script", "anchor_auto", "error_recovery",
                                 "home_return", "waypoint_route", "wrist_toggle"],
        "active_sequence_domain": ["anchor_auto", "error_recovery", "home_return",
                                   "waypoint_route", "wrist_toggle"],
        "mcp_status": None, "mcp_book_text": None, "mcp_book_bbox_xyxy": None,
        "book_id": f"{PROBE_MARKER}_{idx:03d}",
        "object_id": f"book_{idx}", "start_region": "center",
    }
    (ep / "episode_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main(args: Args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(args.episodes):
        _episode(args.out, i, args)
        print(f"  wrote {args.out / str(i)}")
    print(f"\n{args.episodes} synthetic v5 episodes, {args.frames} frames each.")
    print(f"All marked {PROBE_MARKER!r} in book_id — not training data.")
    print(f"\n  uv run examples/e7/convert_e7_to_lerobot.py --root {args.out} \\")
    print("      --repo-id local/e7_v5_fixture --prompt-style category_only --allow-probe")


if __name__ == "__main__":
    main(tyro.cli(Args))
