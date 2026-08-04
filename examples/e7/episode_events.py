"""Read ``episode_events.csv`` — the operator's own account of what happened.

Every other boundary in the pipeline is inferred: the grasp from a gripper
crossing, the demonstration window from a control flag, the destination from a
label table. Those inferences need thresholds, and a pilot batch collected to
validate a threshold cannot also be labelled by it. Teach-button presses carry
no threshold. The operator pressed 3, the arm went to shelf 3, and the event row
says so.

Two header layouts exist and both are read:

    event,frame_id,timestamp                 episodes 0-18, no argument column
    frame_id,event,value,timestamp           after 08-04

The older layout cannot express which shelf a press named, so ``value`` comes
back empty there and callers must fall back to inference. Guessing a shelf id
out of the older rows would manufacture the very declaration this module exists
to read.
"""

from __future__ import annotations

import csv
import dataclasses
import pathlib

# Buttons that name a shelf. Button 1 is the pick pose and has no shelf, so a
# `teach_arrived` for it must not be mistaken for reaching a destination.
SHELF_BUTTONS = ("2", "3", "4")

EV_REQUESTED = "teach_requested"
EV_ARRIVED = "teach_arrived"
EV_ABORTED = "teach_aborted"


@dataclasses.dataclass
class Event:
    frame: int
    name: str
    value: str = ""
    timestamp: float | None = None


@dataclasses.dataclass
class TeachTimeline:
    """Teach presses in order, and where the shelf move landed."""

    events: list[Event] = dataclasses.field(default_factory=list)
    legacy_header: bool = False

    @property
    def shelf_arrivals(self) -> list[Event]:
        return [e for e in self.events
                if e.name == EV_ARRIVED and e.value in SHELF_BUTTONS]

    @property
    def declared_shelf_id(self) -> str:
        """Last shelf the operator actually reached. "" when none did.

        The last press wins: a mis-press corrected before release means the
        operator changed their mind, and the correction is the intent. Aborted
        moves are not arrivals -- reading one as a destination would place the
        insertion boundary at a frame where the arm never went.
        """
        arrivals = self.shelf_arrivals
        return arrivals[-1].value if arrivals else ""

    @property
    def insertion_start_frame(self) -> int | None:
        """Frame the arm finished its move to the shelf — the insertion begins.

        This is the boundary that could not be derived before. Distance to the
        shelf cannot supply it, because how close is close enough is the question
        the pilot was collected to answer.
        """
        arrivals = self.shelf_arrivals
        return arrivals[-1].frame if arrivals else None

    def transit_spans(self) -> list[tuple[int, int]]:
        """Frame ranges the arm moved under a teach button, not under the operator.

        Scripted motion, so not demonstration: a policy trained on it would be
        asked to reproduce a trajectory whose trigger — a button press — never
        appears in its observations.
        """
        spans: list[tuple[int, int]] = []
        pending: Event | None = None
        for e in self.events:
            if e.name == EV_REQUESTED:
                pending = e
            elif e.name in (EV_ARRIVED, EV_ABORTED) and pending is not None:
                if e.frame > pending.frame:
                    spans.append((pending.frame, e.frame))
                pending = None
        return spans


def read_events(episode_dir: pathlib.Path, name: str = "episode_events.csv") -> TeachTimeline:
    """Parse the events file. Empty timeline when it is absent or unreadable."""
    path = episode_dir / name
    tl = TeachTimeline()
    if not path.is_file():
        return tl
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return tl

    header = [c.strip().lower() for c in rows[0]]
    if "frame_id" not in header or "event" not in header:
        return tl
    tl.legacy_header = "value" not in header
    idx = {c: i for i, c in enumerate(header)}

    for row in rows[1:]:
        if len(row) <= max(idx["frame_id"], idx["event"]):
            continue
        try:
            frame = int(float(row[idx["frame_id"]]))
        except ValueError:
            continue
        ts = None
        if (j := idx.get("timestamp")) is not None and j < len(row):
            try:
                ts = float(row[j])
            except ValueError:
                ts = None
        value = ""
        if (j := idx.get("value")) is not None and j < len(row):
            value = row[j].strip()
        tl.events.append(Event(frame, row[idx["event"]].strip(), value, ts))
    return tl
