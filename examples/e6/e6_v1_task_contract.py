"""
E6 v1 — canonical LeRobot ``task`` strings and boundary-drop rules.

Team-fixed wording: do not change strings casually; extend only with explicit agreement.

* Training export uses four primitives only: approach, pick, move, place.
* init_hold / return frames are excluded from export (not v1 core).
* Segment boundaries: drop the last *k* frames of segment A and the first *k* frames of
  segment B at each A|B transition among the four (see BOUNDARY_DROP_FRAMES_PER_SIDE).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

# Fixed by team: same value must be used when regenerating datasets for comparability.
BOUNDARY_DROP_FRAMES_PER_SIDE: int = 2

OBJECT_PHRASE_DEFAULT: str = "red object"

V1_SEGMENTS: tuple[str, ...] = ("approach", "pick", "move", "place")


def task_approach(object_phrase: str = OBJECT_PHRASE_DEFAULT) -> str:
    return f"approach {object_phrase}"


def task_pick(object_phrase: str = OBJECT_PHRASE_DEFAULT) -> str:
    return f"pick {object_phrase}"


def task_move(direction: Literal["left", "right", "middle"]) -> str:
    return f"move object to {direction}"


def task_place(direction: Literal["left", "right", "middle"]) -> str:
    return f"place object to {direction}"


def place_direction_from_transport_primitive(transport_primitive: str) -> Literal["left", "right", "middle"]:
    """Maps segment CSV / build_episode_primitive_segments transport names to place wording."""
    if transport_primitive == "move_left":
        return "left"
    if transport_primitive == "move_right":
        return "right"
    if transport_primitive == "move_to_middle":
        return "middle"
    return "left"


def move_direction_from_transport_primitive(transport_primitive: str) -> Literal["left", "right", "middle"]:
    return place_direction_from_transport_primitive(transport_primitive)


def task_for_v1_segment(
    segment: str,
    *,
    transport_primitive: str,
    object_phrase: str = OBJECT_PHRASE_DEFAULT,
) -> str:
    """Return the exact LeRobot task line for a v1 primitive segment."""
    seg = segment.lower().strip()
    if seg == "approach":
        return task_approach(object_phrase)
    if seg == "pick":
        return task_pick(object_phrase)
    if seg == "move":
        d = move_direction_from_transport_primitive(transport_primitive)
        return task_move(d)
    if seg == "place":
        d = place_direction_from_transport_primitive(transport_primitive)
        return task_place(d)
    raise ValueError(f"Unsupported v1 segment: {segment!r}")


def shrink_pair_for_boundary(
    prev: tuple[int, int] | None, nxt: tuple[int, int] | None, k: int
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Drop *k* frames from the end of *prev* and *k* from the start of *nxt* (inclusive ranges)."""
    if prev is None or nxt is None or k <= 0:
        return prev, nxt
    a0, a1 = prev
    b0, b1 = nxt
    if a0 > a1 or b0 > b1:
        return prev, nxt
    new_prev = (a0, a1 - k) if a1 - k >= a0 else None
    new_nxt = (b0 + k, b1) if b0 + k <= b1 else None
    if new_prev is not None and new_prev[0] > new_prev[1]:
        new_prev = None
    if new_nxt is not None and new_nxt[0] > new_nxt[1]:
        new_nxt = None
    return new_prev, new_nxt


def apply_v1_boundary_drops(
    ranges: dict[str, tuple[int, int]],
    k: int = BOUNDARY_DROP_FRAMES_PER_SIDE,
) -> dict[str, tuple[int, int]]:
    """
    *ranges*: inclusive (start, end) per segment name; use only approach/pick/move/place.
    Empty or negative ranges should be represented as None or (0,-1); callers skip those.

    Applies sequential trimming at approach|pick, pick|move, move|place boundaries.
    """
    order = V1_SEGMENTS
    cur: dict[str, tuple[int, int] | None] = {}
    for name in order:
        t = ranges.get(name, (-1, -1))
        if t[0] < 0 or t[1] < 0 or t[0] > t[1]:
            cur[name] = None
        else:
            cur[name] = (t[0], t[1])

    for i in range(len(order) - 1):
        a_name, b_name = order[i], order[i + 1]
        prev, nxt = cur.get(a_name), cur.get(b_name)
        if prev is None or nxt is None:
            continue
        new_prev, new_nxt = shrink_pair_for_boundary(prev, nxt, k)
        cur[a_name], cur[b_name] = new_prev, new_nxt

    out: dict[str, tuple[int, int]] = {}
    for name in order:
        t = cur.get(name)
        if t is not None:
            out[name] = t
    return out


def load_v1_ranges_and_transport(
    segments_csv: Path,
    episode_folder: int,
) -> tuple[dict[str, tuple[int, int]], str]:
    """
    Read ``episode_primitive_segments.csv``-style rows for one episode.

    Returns:
        ranges: only ``approach``/``pick``/``move``/``place`` with inclusive indices.
        transport_primitive: from the ``move`` row (e.g. ``move_left``).
    """
    ranges: dict[str, tuple[int, int]] = {}
    transport = "move_unknown"
    with Path(segments_csv).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["episode_folder"]) != episode_folder:
                continue
            seg = row["segment"].strip().lower()
            if seg not in V1_SEGMENTS:
                continue
            a, b = int(row["start_frame"]), int(row["end_frame"])
            ranges[seg] = (a, b)
            if seg == "move":
                transport = row.get("transport_primitive", "move_unknown").strip() or "move_unknown"
    if "move" not in ranges:
        raise ValueError(f"No move segment for episode_folder={episode_folder} in {segments_csv}")
    return ranges, transport


def frame_to_task_map(
    *,
    ranges_after_drop: dict[str, tuple[int, int]],
    transport_primitive: str,
    object_phrase: str = OBJECT_PHRASE_DEFAULT,
) -> dict[int, str]:
    """Map absolute frame index (CSV row index, 0..n-1) -> task string for frames inside v1 ranges."""
    m: dict[int, str] = {}
    for seg in V1_SEGMENTS:
        if seg not in ranges_after_drop:
            continue
        a, b = ranges_after_drop[seg]
        if a > b:
            continue
        for fi in range(a, b + 1):
            m[fi] = task_for_v1_segment(seg, transport_primitive=transport_primitive, object_phrase=object_phrase)
    return m
