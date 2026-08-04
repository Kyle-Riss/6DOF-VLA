"""Rebuild the destination label of the 08-04 pilot batch from the recorded release pose.

The batch was collected four episodes per shelf, but every episode carries
``resolved_target_side="right"``: the collector derives the destination from the
shelf label table, and with one category and a fixed arrangement that table
always resolves to the same shelf. The demonstrations vary; the label does not.
``actual_shelf`` is wrong on two episodes as well, disagreeing with the taught
shelf y by 180mm.

This writes a relabelled copy. The destination comes from the TCP y at release
and nothing else, so the label follows the demonstration rather than a table.
Images are symlinked, so the copy costs kilobytes.

    uv run examples/e7/relabel_pilot_from_release.py --src <collect dir> --out <work dir>

The copy is not collector output and must not be fed back as if it were: every
episode gets ``relabel_source`` and ``relabel_note`` recorded in its meta.

This does NOT make the batch a book-insertion dataset. The release poses sit
100-250mm in front of the shelf face in x -- the arm approached the right shelf
laterally and let go before entering it. Lateral targeting is real and separable;
insertion depth is absent. Use it to exercise the pipeline, not to claim the task.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil
import sys

import pandas as pd
import tyro

sys.path.insert(0, str(pathlib.Path(__file__).parent / "e7_prompt"))
from e7_prompt import CANONICAL_DESTINATIONS, RIG_LAYOUT  # noqa: E402

SCHEMA_VERSION = 5
GRIPPER_OPEN_DROP = 0.05   # a fall this size in the command column is a release
SIDE_TO_ID = {v: k for k, v in RIG_LAYOUT.items()}


@dataclasses.dataclass
class Args:
    src: pathlib.Path
    """Collector output directory holding numbered episode directories."""
    out: pathlib.Path
    """Where to write the relabelled copy. Overwritten if it exists."""
    episodes: str = "7-18"
    """Inclusive range, or a comma-separated list."""
    dry_run: bool = False


def parse_episodes(spec: str) -> list[str]:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return [str(i) for i in range(int(lo), int(hi) + 1)]
    return [s.strip() for s in spec.split(",") if s.strip()]


def release_index(g: pd.Series) -> int:
    """Last frame where the gripper command falls -- the release, not an adjustment."""
    v = g.fillna(0.0).to_numpy()
    for i in range(len(v) - 1, 0, -1):
        if v[i] < v[i - 1] - GRIPPER_OPEN_DROP:
            return i
    return len(v) - 1


def side_from_y(y: float, shelf_y: dict[str, float]) -> tuple[str, float]:
    """Nearest shelf by taught y. Lateral only: x is the insertion axis, and this
    batch never entered it, so including x would rank shelves by how far short the
    arm stopped rather than by which shelf it stopped in front of."""
    wp = min(shelf_y, key=lambda k: abs(shelf_y[k] - y))
    return RIG_LAYOUT[wp], abs(shelf_y[wp] - y)


def main(args: Args) -> None:
    eps = parse_episodes(args.episodes)
    if not args.dry_run:
        if args.out.exists():
            shutil.rmtree(args.out)
        args.out.mkdir(parents=True)

    print(f"  {'ep':5} {'y_mm':>9} {'side':>7} {'dy_mm':>7}  {'was':>7}  {'was actual':>10}")
    changed = 0
    for name in eps:
        sd = args.src / name
        meta = json.loads((sd / "episode_meta.json").read_text(encoding="utf-8"))
        df = pd.read_csv(sd / "robot_data.csv")

        shelf_y = {str(k): float(v) for k, v in (meta.get("shelf_waypoint_tcp_y_mm") or {}).items()}
        if not shelf_y:
            print(f"  {name:5} no shelf_waypoint_tcp_y_mm -- skipped")
            continue

        idx = release_index(df["gripper_command"])
        y = float(df["y"].iloc[idx])
        side, dy = side_from_y(y, shelf_y)
        assert side in CANONICAL_DESTINATIONS

        was, was_actual = meta.get("resolved_target_side"), meta.get("actual_shelf")
        mark = "" if (was == side and was_actual == side) else "  <- changed"
        if mark:
            changed += 1
        print(f"  {name:5} {y:9.1f} {side:>7} {dy:7.1f}  {str(was):>7}  {str(was_actual):>10}{mark}")

        if args.dry_run:
            continue

        meta.update({
            "schema_version": SCHEMA_VERSION,
            "resolved_target_side": side,
            "resolved_target_shelf_id": SIDE_TO_ID[side],
            "resolved_target_tcp_y_mm": shelf_y[SIDE_TO_ID[side]],
            "target_shelf": side,
            "actual_shelf": side,
            "actual_shelf_distance_mm": round(dy, 2),
            "target_resolution_status": "resolved",
            "target_resolution_reason": "release_tcp_y",
            # Provenance. Without these the copy is indistinguishable from collector
            # output, and the next person to read it would trust a label this script
            # inferred as though the arm had reported it.
            "relabel_source": "release_tcp_y",
            "relabel_note": (
                "Destination relabelled from the TCP y at release by "
                "examples/e7/relabel_pilot_from_release.py. The collector stamped "
                "'right' on every episode of this batch because one category and a "
                "fixed arrangement always resolve to the same shelf. Lateral "
                "targeting only -- the release poses are 100-250mm short of the "
                "shelf face in x, so this batch contains no insertion."
            ),
            "relabel_release_frame": int(idx),
            "relabel_release_y_mm": round(y, 2),
            "collector_resolved_target_side": was,
            "collector_actual_shelf": was_actual,
        })

        ed = args.out / name
        ed.mkdir(parents=True)
        (ed / "episode_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        for f in ("robot_data.csv", "episode_events.csv", "dataset.npy"):
            if (sd / f).is_file():
                shutil.copy2(sd / f, ed / f)
        if (sd / "images").is_dir():
            (ed / "images").symlink_to((sd / "images").resolve(), target_is_directory=True)

    print(f"\n  {changed}/{len(eps)} episode(s) relabelled"
          + ("  (dry run, nothing written)" if args.dry_run else f" -> {args.out}"))


if __name__ == "__main__":
    main(tyro.cli(Args))
