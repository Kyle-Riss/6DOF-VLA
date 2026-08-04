"""Convert xArm 6 (E7) raw episodes to LeRobot format for pi05_e7_v1_lora.

Derived from ``examples/e6/convert_e6_v16_to_lerobot.py``. The action contract is
byte-identical to E6 v16+ so the E6→E7 cross-embodiment comparison holds:

    state  = [j1..j6 (t),   gripper (t)]           7D, degrees / gripper absolute
    action = [Δj1..Δj6,     gripper (t+1)]         7D, deg/frame / gripper absolute

Differences vs the E6 converter
-------------------------------
1. **6 tasks, unidirectional.** E6 was bidirectional (left↔right) and keyed prompts
   off ``episode_meta["source_zone"]``. The xArm6 task is "variable pick anchor →
   fixed target area", so there is no left/right axis: ``source_zone`` is neither
   required nor read, and the phase→task map is a plain 6-entry list.

2. **Gripper source is selectable.** The Jetson gripper goal is rate-gated
   (``min_send_interval_sec=0.4``), so ``gripper_command`` is a staircase quantised
   to ~6 frames at 16 Hz. ``gripper_trigger_raw`` (ungated, per frame) is the human
   intent signal. Both phase detection and the action label can be driven from
   either — see ``--gripper-source``. Ungated is preferred for phase boundaries;
   whichever is used for the label must match what the inference executor expects.

3. **Trim by ``robot_mode``, NOT by first motion.** This is the important one.
   An xArm6 episode contains scripted position-mode moves at both ends (home →
   pick anchor, and work → home) plus mid-episode mode flips. Measured on pilot
   episode 0: frame 0 and frame 678 have *identical* joint values, and the leading
   scripted move spans ~47 frames at |Δq| ≈ 1.3 deg/frame. E6's ``find_first_motion``
   returns frame 2 there, so ~30% of the converted episode would be the robot
   driving itself home — and the policy would learn to replay that.

   ``robot_mode == 5`` is xArm's Cartesian-velocity (teleop) mode; ``0`` is position
   mode (scripted). Only the contiguous mode-5 run containing both gripper events is
   kept. ``teleop_enabled`` is deliberately NOT used: on the pilot it agreed with
   actual motion only 59.3% of the time (robot_mode agreed 78.5%).

4. **Integrity guards.** E6 had none, so a dropped camera pair silently doubled the
   Δq at that index. Here ``dt_from_prev`` / ``frame_dropped_before`` /
   ``n_commands_in_frame`` are checked when present (all optional — older schemas
   just skip the corresponding guard). ``episode_meta.json`` is checked against the
   policy contract (units, rates) so a mis-configured collector fails loudly.

5. **Diagnostics report.** Per-axis action quantile ranges, gripper gating lag,
   twist clamping rate and robot_mode segment breakdown. The per-axis range check
   exists because E6's j5 was mechanically fixed: q99-q01 was 0.245 vs 3.0~4.3 for
   the other axes, and normalising by that range spent ~30% of the training gradient
   on sensor noise. Pilot episode 0 shows the same pattern on xArm's **j4**
   (range 0.224 vs median 1.96) — flagged automatically.

Action semantics
----------------
``--action-semantics sequential`` (default) writes ``a[t] = q[t+1] - q[t]``, matching
E6 v16+ so an E6 v23 checkpoint can be used as the init for a fair transfer run.

``--action-semantics current_relative`` writes the **absolute** next joint position
instead. The current-relative offset ``q[t+k+1] - q[t]`` is chunk-relative and so
cannot be baked per-frame; it is produced at load time by openpi's ``DeltaActions``
transform, which subtracts the current state from every action in the chunk.
⚠ Using this mode therefore ALSO requires adding to the data config::

    delta_action_mask = _transforms.make_bool_mask(6, -1)   # joints delta, gripper absolute
    data_transforms = _transforms.Group(
        inputs=[e7_policy.E7Inputs(...), _transforms.DeltaActions(delta_action_mask)],
        outputs=[e7_policy.E7Outputs(), _transforms.AbsoluteActions(delta_action_mask)],
    )

Usage:
    uv run examples/e7/convert_e7_to_lerobot.py --root ~/xarm_vla_episodes

    uv run examples/e7/convert_e7_to_lerobot.py --root ~/xarm_vla_episodes \\
      --gripper-source trigger_raw --gripper-binarize --on-gap warn
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import PIL.Image
import tyro

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_REPO_ID = "Kyle-Riss/xarm_e7_v1"

JOINT_COLS = ("j1", "j2", "j3", "j4", "j5", "j6")
GRIPPER_COL_GATED = "gripper_command"
GRIPPER_COL_RAW = "gripper_trigger_raw"
IMAGE_COL_HIK = "image_path_hik"
IMAGE_COL_ZED = "image_path_zed"

# Optional integrity columns — guards are skipped when a column is absent.
COL_DT = "dt_from_prev"
COL_DROPPED = "frame_dropped_before"
COL_NCMD = "n_commands_in_frame"
COL_MODE = "robot_mode"
COL_TELEOP = "teleop_enabled"

# xArm control modes. 5 = Cartesian velocity (what the teleop path drives), 0 =
# position mode. ``robot_mode`` alone is NOT a demonstration/scripted discriminator:
# the operator flips to position mode mid-episode to clear a collision, and the
# 99 Hz mode feedback is sampled at 16 Hz so transitions land late. Measured on the
# book batch, ``robot_mode == 5`` splits ep3's grasp (close@233 in run [148,311],
# open@314 outside it) and the whole episode is dropped.
#
# The demonstration gate is therefore the UNION ``robot_mode == 5 OR teleop_enabled``
# — teleop_enabled is the clutch, so it is 0 through scripted home/anchor moves even
# though the joystick axes (``teleop_raw_*``) still read non-zero there. Verified on
# the book batch: scripted lead-ins (ep0 f0-46, ep3 f0-46) have teleop_enabled = 0.0%
# and are excluded, while ep3's release+retract (f311-327, teleop_enabled 76.5%) is
# kept. ``teleop_raw`` is unusable as a gate: it is non-zero on 98-100% of frames.
TELEOP_MODE = 5
GATE_BRIDGE_FRAMES = 8  # bridge flag flicker shorter than this when forming runs

# Twist clamping diagnostic (raw teleop vs what the driver actually accepted).
TWIST_RAW_COLS = tuple(f"teleop_raw_{i}" for i in range(1, 7))
TWIST_SENT_COLS = tuple(f"twist_sent_{i}" for i in range(1, 7))

# episode_meta.json also carries `prompt_contract_hash` and `task_mapping_hash`
# from the collector. This converter deliberately does not read them, and they are
# deliberately absent from CONTEXT_KEYS.
#
# The two machines build their prompt packages independently, so the hash payloads
# are assembled differently and never agree -- measured, 33d2506b vs 72368ba0. An
# equality check on them would reject every episode forever while looking like a
# data problem. What has to agree is the rendered prompt string and the plaintext
# values (rig_layout, discrete_state_input), which is what the robot gates on.
# Leave the collector's hashes as a ledger; do not promote them to a check.

# episode_meta.json fields validated against the policy contract.
META_CONTRACT = {
    "joint_unit": "degree",
    "control_mode": "cartesian_velocity",
    "command_semantics": "cartesian_twist",
}

# Episodes below this are rejected outright rather than converted.
#
# v5 renamed the static `target_shelf_id` to `legacy_static_target_shelf_id` and
# introduced `resolved_target_*` as the only canonical destination. A v4 episode
# has no resolved field at all, and its static one describes the default rig
# arrangement rather than the one that episode was recorded under — promoting it
# would be right whenever the shelves happened to be in the default order and
# silently wrong the rest of the time, which is the worst of both. Reject instead.
#
# v3 and earlier were labelled under the previous taxonomy, where 교양책 mapped to
# `humanities`. Under the current one 교양 is `liberal_arts` and 인문학 is
# `humanities` — the same Korean word now denotes a different class. A v3 episode
# mixed into a v4+ batch therefore sends one whole category to the wrong shelf,
# and nothing downstream can detect it: the trajectories are valid, the prompts
# are well formed, and the loss looks normal. Relabel or discard; never convert.
MIN_SCHEMA_VERSION = 5

MOTION_THRESHOLD_DEG = 0.1
PLACE_SETTLE = 10    # trailing frames kept after gripper 1→0 (place event)
PICK_SETTLE = 4      # frames AFTER close_idx still labeled "pick up"
ACTION_HORIZON = 16  # must match pi05_e7_v1_lora action_horizon
SMOOTH_WINDOW = 5
TRANSPORT_PERCENTILE = 60
GATING_MATCH_WINDOW = 16  # frames; max |gated - raw| accepted when pairing crossings

# Episode-level context carried from episode_meta.json into the dataset. These are
# NOT model inputs — the rule reaches the policy through the prompt only. They exist
# so the counterfactual evaluation can group episodes into (same book, different
# rule) pairs after training.
CONTEXT_KEYS = (
    "pair_id", "rule_version", "context_mode", "category", "shelf_color", "object_id",
    # Classification provenance. Recorded so the on-device VLM can be scored after
    # the fact against the human-confirmed label, without collecting a second
    # dataset for it: `category` is the confirmed truth, `category_predicted`
    # is what the model said.
    "target_shelf", "shelf_label", "category_source", "category_predicted",
    "category_confidence", "category_confirmed", "vlm_model_version",
    "mcp_request_id", "semantic_frame_id", "cover_image_path", "ocr_text",
    # Where the book ACTUALLY ended up, from the TCP at release. Independent of
    # the label, so the two can be compared — see check_target_vs_actual.
    "actual_shelf", "actual_shelf_label", "actual_shelf_distance_mm",
    # Two different layouts, deliberately under different names. `shelf_layout` is
    # the spec form {category: side} and VARIES — that variation is the whole point,
    # because a batch where it never changes makes category and side perfectly
    # correlated and the sign can never be shown to matter. `shelf_layout_physical`
    # is the rig form {waypoint: side} and does NOT vary: the shelves stay put, only
    # the signs move. Carrying both keeps the two from being confused later.
    "shelf_layout", "shelf_layout_physical", "shelf_labels", "start_region", "book_id",
    # Taught TCP y per shelf waypoint. The only field here that does not
    # descend from a side constant, so the only one that can contradict it.
    "shelf_waypoint_tcp_y_mm", "waypoint_file_sha256",
    # v5. `resolved_*` is the destination this episode was actually recorded
    # against; the legacy field is carried only so a later audit can see what the
    # static table would have said, never to derive from.
    "resolved_target_shelf_id", "resolved_target_side", "resolved_target_side_ko",
    "resolved_target_tcp_y_mm", "shelf_id_domain", "legacy_static_target_shelf_id",
    "labels_source", "active_sequence", "motion_source_domain",
    "active_sequence_domain", "motion_source_clock", "motion_source_age_normal_max_ms",
    # The thresholds in force when the classifier response was judged. Without
    # these a later run with a different threshold is not comparable.
    "mcp_confidence_threshold", "book_bbox_max_clip_ratio", "mcp_schema_version",
    # The teach-button declaration, kept beside the value it overrides. The
    # collector used to write one destination field holding whichever of the two
    # it had; splitting them is what made the 08-04 mislabelling visible at all,
    # so both travel with the dataset and an audit can still see the disagreement.
    "intended_shelf_id", "intended_shelf_side", "intended_shelf_source",
    "intended_shelf_button_presses", "intended_matches_nearest",
    "target_resolution_status", "target_resolution_reason", "target_resolution_source",
    # What the shelf-label table would have said. Recorded, never derived from:
    # on a single-category batch it returns the same shelf every time.
    "label_derived_target_shelf_id", "label_derived_target_side",
    "label_derived_resolution_status", "label_derived_resolution_reason",
    # Where the book actually landed, assigned on lateral distance alone. Depth is
    # reported separately because it measures a different thing -- how far into the
    # shelf the arm went -- and folding it into the assignment ranks shelves by how
    # far short the release stopped rather than by which shelf it stopped at.
    "nearest_shelf_id", "nearest_shelf_lateral_mm", "nearest_shelf_depth_mm",
    "nearest_shelf_assignment_axis",
)

# episode_meta keys that may carry the observed landing shelf, in priority order.
ACTUAL_SHELF_KEYS = ("actual_shelf", "actual_shelf_label", "released_shelf", "nearest_shelf")
# ...and the release-to-waypoint distance, for judging whether the landing is even
# well defined. Lateral first: it is the axis the landing is assigned on, so it is
# the one the target-vs-actual comparison should report.
ACTUAL_DIST_KEYS = ("nearest_shelf_lateral_mm", "actual_shelf_distance_mm",
                    "release_shelf_distance_mm", "nearest_shelf_distance_mm")

# Two phase schemas, selected by episode_meta["task_id"].
#
# ``planar``    — flat pick-and-place (E6-equivalent). Boundaries come from the
#                 gripper transitions plus a motion-based lift/transport/place split.
# ``insertion`` — shelf insertion. The align→insert boundary produces neither a
#                 gripper transition nor a motion discontinuity, so it is derived
#                 from TCP geometry instead (see ``insertion_phases``).
#
# {obj} is filled from episode_meta["prompt_object_name"], {tgt} from
# ["target_shelf"] — a mixed collection therefore yields distinct prompts rather
# than mislabelling everything as one object/destination.
PLANAR_TEMPLATES: list[str] = [
    "approach the {obj}",                  # 0
    "pick up the {obj}",                   # 1
    "lift the {obj}",                      # 2
    "move the {obj} to the target area",   # 3
    "place the {obj} in the target area",  # 4
    "release the {obj}",                   # 5
]
PLANAR_NAMES = ["approach", "pick", "lift", "transport", "place", "release"]

INSERTION_TEMPLATES: list[str] = [
    "approach the {obj}",                      # 0
    "grasp the {obj}",                         # 1
    "lift the {obj}",                          # 2
    "carry the {obj} to the {tgt} shelf",      # 3
    "align the {obj} with the {tgt} shelf",    # 4
    "insert the {obj} into the {tgt} shelf",   # 5
    "release the {obj}",                       # 6
    "retract from the {tgt} shelf",            # 7
]
INSERTION_NAMES = ["approach", "grasp", "lift", "carry", "align", "insert", "release", "retract"]

SCHEMAS = {
    "planar": (PLANAR_TEMPLATES, PLANAR_NAMES),
    "insertion": (INSERTION_TEMPLATES, INSERTION_NAMES),
}
DEFAULT_OBJECT = "object"
DEFAULT_TARGET = "target"

# TCP-geometry knobs for the insertion schema.
#
# ⚠ These are millimetres of travel along the insertion axis, so the resulting
# phase LENGTHS depend on how fast the operator pushes. On pilot ep0 (a top-down
# can place, ~8 mm/frame descent) 15 mm buys only ~2 frames. A real shelf
# insertion is slower and longer, so retune against the first insertion batch:
# aim for insert ≈ 10-20% of the episode, and read the actual axis travel off the
# "Insertion axis" line in the conversion report.
INSERT_AXIS_WINDOW = 8      # frames before release used to estimate the insertion axis
INSERT_ENTER_MM = 15.0      # travel along the axis that counts as "inside the slot"
ALIGN_ENTER_MM = 60.0       # travel along the axis that counts as "lined up"
RETRACT_EXIT_MM = 10.0      # reverse travel after release that counts as retracting


def tasks_for(obj: str, tgt: str, schema: str) -> list[str]:
    templates, _ = SCHEMAS[schema]
    return [t.format(obj=obj, tgt=tgt) for t in templates]


# ---------------------------------------------------------------------------
# MCP prompt rendering
# ---------------------------------------------------------------------------
#
# The prompt template is EXPERIMENT DESIGN, so it is rendered here from the
# episode's enum fields rather than read back from a string the collector baked
# into episode_meta["prompt"]. That keeps the choice reversible: the same
# recorded episodes can be re-emitted under a different template by re-running
# the converter, and no collection has to be repeated to change it.
#
# The templates, the category map, the validators and the contract hash all live
# in the `e7_prompt` package, installed on BOTH this machine and the Jetson:
#
#     uv pip install ./examples/e7/e7_prompt
#
# There is deliberately no local copy. If training and inference each formatted
# their own string the two could drift apart silently -- the tokens would
# differ, the policy would degrade, and the degradation would be misattributed
# to the policy. See examples/e7/e7_prompt/README.md.
from e7_prompt import (  # noqa: E402
    CANONICAL_CATEGORIES,
    CANONICAL_DESTINATIONS,
    INCOMPLETE,
    PROMPT_TEMPLATES,
    TokenizerSpec,
    build_manifest,
    canonical_category,
    RIG_LAYOUT,
    compute_contract_hash,
    render_from_meta,
    validate_all,
)


def build_rule_tables(episode_paths: Sequence[Path]) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Aggregate ``rule_version -> {category: shelf_color}`` over every episode.

    A rule must be a FUNCTION: within one rule_version a category may map to
    exactly one colour. Violations are returned rather than raised so the report
    can list all of them at once.
    """
    tables: dict[str, dict[str, str]] = {}
    conflicts: list[str] = []
    for ep in episode_paths:
        try:
            meta = _read_meta(ep)
        except FileNotFoundError:
            continue
        ver = str(meta.get("rule_version") or "").strip()
        # Accept both key spellings: the collector schema is still settling and
        # a silent miss here drops the episode to the baked-in prompt fallback.
        cat = canonical_category(str(meta.get("category") or meta.get("object_category") or ""))
        tgt = resolved_target_side(meta) or str(meta.get("shelf_color") or "").strip()
        if not (ver and cat and tgt):
            continue
        prev = tables.setdefault(ver, {}).get(cat)
        if prev is not None and prev != tgt:
            conflicts.append(f"{ep.name}: {ver} maps {cat} -> both {prev!r} and {tgt!r}")
        tables[ver][cat] = tgt
    return tables, conflicts


def render_prompt(meta: dict, style: str, rule_tables: dict[str, dict[str, str]]) -> str | None:
    """Build the episode prompt from enum fields. ``None`` if fields are missing.

    Thin alias over the shared renderer so the converter and the Jetson client
    cannot diverge. Kept as a named function because the report and the guard
    both refer to it.
    """
    return render_from_meta(meta, style, rule_tables)


def _report_category_destination_matrix(counts: dict[tuple[str, str], int]) -> None:
    """Print how each category was distributed over destinations, and say what that permits.

    This is the measurement that decides whether the destination word in the
    prompt does any work. If a category always went to the same shelf, a policy
    can answer every episode from the category alone and never read the rest of
    the sentence -- the destination is then decoration, and prompt-sensitivity
    cannot be scored no matter how the eval is written. The batch has to break
    that correlation before the question is askable.

    Three destinations per category is the goal rather than two: with two, a
    held-out third combination has no training support, so a failure on it says
    nothing about whether the policy reads the destination or simply never saw
    that shelf named.
    """
    if not counts:
        return
    cats = sorted({c for c, _ in counts})
    dests = [d for d in CANONICAL_DESTINATIONS if any(d == x for _, x in counts)]
    print("\n  Category x destination (episodes):")
    print("    " + " " * 14 + "".join(f"{d:>10}" for d in dests) + "     distinct")
    for c in cats:
        row = [counts.get((c, d), 0) for d in dests]
        n = sum(1 for v in row if v)
        flag = "" if n >= 3 else ("  <- one destination only" if n < 2 else "  <- two")
        print(f"    {c:14}" + "".join(f"{v:>10}" for v in row) + f"{n:>13}{flag}")

    worst = min((sum(1 for d in dests if counts.get((c, d), 0)) for c in cats), default=0)
    if worst >= 3:
        print("    every category reaches all three shelves — the destination word carries")
        print("    information the category word cannot supply, so prompt-sensitivity is")
        print("    measurable: hold the observation fixed, vary only the destination.")
    elif worst == 2:
        print("    ⚠ some category reaches only two shelves. Better than one, but the third")
        print("      combination is untrained, so a miss there is unattributable.")
    else:
        print("    ⚠ some category reaches ONE shelf. The policy can satisfy every episode")
        print("      from the category alone; the destination word is unfalsifiable here.")
        print("      Move the shelf signs so the same category is demonstrated elsewhere.")


def _build_contract_manifest(
    config_name: str, prompt_style: str, rule_tables: dict[str, dict[str, str]]
) -> dict:
    """Stamp the prompt contract, reading tokenizer settings from the train config.

    ``discrete_state_input`` is the one that bites: it defaults to False, and
    flipping it makes the tokenizer wrap the text as
    ``"Task: ... , State: <7 values>;\\nAction: "`` -- ~35 extra tokens, no error,
    no visible symptom. Deriving it here instead of hardcoding means the manifest
    tracks whatever the config actually says.
    """
    try:
        from openpi.training import config as _openpi_config  # noqa: PLC0415

        model = _openpi_config.get_config(config_name).model
        return build_manifest(
            prompt_style,
            rule_tables,
            TokenizerSpec(
                max_token_len=int(model.max_token_len),
                discrete_state_input=bool(model.discrete_state_input),
            ),
            image_keys=tuple(model.image_keys),
            action_dim=int(model.action_dim),
            action_horizon=int(model.action_horizon),
        )
    except Exception as exc:  # noqa: BLE001 - never fail a conversion over the manifest
        print(f"  ⚠ could not read config {config_name!r} for the prompt contract: {exc}")
        print("    prompt_contract.json written WITHOUT tokenizer settings. The robot")
        print("    reads the plaintext values out of this file, so regenerate it with a")
        print("    valid config before deploying anything trained on this dataset.")
        return {
            "prompt_contract_hash": None,
            "prompt_style": prompt_style,
            "rule_tables": rule_tables,
            "config_name": config_name,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Trimming / event helpers
# ---------------------------------------------------------------------------

def find_first_motion(df: pd.DataFrame) -> int:
    joints = df[list(JOINT_COLS)].values.astype(np.float32)
    for i in range(1, len(joints)):
        if np.abs(joints[i] - joints[i - 1]).max() >= MOTION_THRESHOLD_DEG:
            return i
    return 0


def find_close_idx(gripper: np.ndarray, thresh: float) -> int | None:
    """First frame where the gripper command crosses UP through ``thresh`` (pick)."""
    for i in range(1, len(gripper)):
        if gripper[i - 1] < thresh <= gripper[i]:
            return i
    return None


def find_open_idx(gripper: np.ndarray, thresh: float, after: int = 0) -> int | None:
    """First frame at/after ``after`` where the command crosses DOWN through ``thresh``."""
    for i in range(max(1, after + 1), len(gripper)):
        if gripper[i - 1] >= thresh > gripper[i]:
            return i
    return None


def all_crossings(sig: np.ndarray, thresh: float, up: bool) -> list[int]:
    """Every index where ``sig`` crosses ``thresh`` in the given direction."""
    if up:
        return [i for i in range(1, len(sig)) if sig[i - 1] < thresh <= sig[i]]
    return [i for i in range(1, len(sig)) if sig[i - 1] >= thresh > sig[i]]


def mode_runs(mode: np.ndarray, target: int) -> list[tuple[int, int]]:
    """All contiguous [start, end) runs where ``mode == target``."""
    m = (mode == target).astype(np.int8)
    d = np.diff(np.r_[0, m, 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(), strict=True))


def bridged_runs(mask: np.ndarray, bridge: int = 0) -> list[tuple[int, int]]:
    """Contiguous inclusive [start, end] runs of True, bridging gaps <= ``bridge``."""
    m = np.asarray(mask, dtype=bool).copy()
    if bridge:
        idx = np.flatnonzero(m)
        for a, b in zip(idx[:-1], idx[1:], strict=False):
            if 1 < b - a <= bridge + 1:
                m[a:b] = True
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(m) - 1))
    return out


def demonstration_mask(df: pd.DataFrame, teleop_mode: int) -> np.ndarray:
    """Frames the operator was actually driving: ``robot_mode == 5 OR teleop_enabled``.

    Either column may be absent; when both are, every frame is considered active and
    the caller falls back to the first-motion trim.
    """
    n = len(df)
    active = np.zeros(n, dtype=bool)
    seen = False
    if COL_MODE in df.columns:
        active |= df[COL_MODE].values.astype(int) == teleop_mode
        seen = True
    if COL_TELEOP in df.columns:
        col = df[COL_TELEOP]
        active |= (col.astype(str).str.lower() == "true").values if col.dtype == object else col.values.astype(bool)
        seen = True
    return active if seen else np.ones(n, dtype=bool)


def grasp_pairs(gripper: np.ndarray, thresh: float) -> list[tuple[int, int]]:
    """Every (close, open) pair, each open being the first release after its close."""
    binary = (gripper >= thresh).astype(np.int8)
    d = np.diff(binary)
    closes = (np.flatnonzero(d == 1) + 1).tolist()
    opens = (np.flatnonzero(d == -1) + 1).tolist()
    pairs = []
    for c in closes:
        o = next((x for x in opens if x > c), None)
        if o is not None:
            pairs.append((c, o))
    return pairs


def select_grasp(pairs: list[tuple[int, int]]) -> tuple[int, int] | None:
    """The pair with the longest carry.

    Not the first: ep1 of the book batch has three grasps (56f, 14f, 77f) because the
    book slipped twice, and the first-crossing rule would convert a failed attempt.
    Not the last either — that would pick up a re-grip after the placement.
    """
    return max(pairs, key=lambda p: p[1] - p[0]) if pairs else None


def repair_joint_spikes(df: pd.DataFrame, joint_cols: Sequence[str]) -> int:
    """Replace frames where all six joints share one value; returns how many.

    The collector occasionally writes a scalar broadcast into the joint columns
    (``j1 == j2 == ... == j6``), e.g. book ep3 f324 = 6.990085179143219 on all six
    while the TCP columns are unchanged. Measured rate: 5/522 (ep1), 1/437 (ep3).
    Rare, but each one poisons TWO sequential-delta labels with a fabricated
    +-80..111 deg step, which then lands in the q99-q01 normalisation divisor
    (real per-axis range is 1.4-3.8 deg).

    A genuine pose with six identical joint angles is not reachable in practice, so
    the test has no false positives. Repair is a copy of the previous good frame,
    which makes the delta across the spike zero rather than fabricated.
    """
    joints = df[list(joint_cols)].values.astype(np.float64)
    bad = np.array([len(np.unique(np.round(row, 6))) == 1 for row in joints])
    if not bad.any():
        return 0
    good = np.flatnonzero(~bad)
    if good.size == 0:
        return 0
    for i in np.flatnonzero(bad):
        src = good[good < i]
        ref = src[-1] if src.size else good[0]
        df.loc[df.index[i], list(joint_cols)] = joints[ref]
    return int(bad.sum())


# ---------------------------------------------------------------------------
# Insertion phase segmentation — TCP geometry
#
# Neither the gripper signal nor joint motion marks the align→insert boundary:
# the arm keeps moving smoothly straight through it. What DOES change is the
# direction of travel — the last stretch before release is a straight push along
# the slot axis. So estimate that axis from the frames just before release, then
# project the whole trajectory onto it and read the boundaries off the projection.
#
# Verified on pilot ep0: place_x/y in episode_meta matches TCP at release exactly,
# and the 8-frame pre-release displacement gives a clean unit axis.
# ---------------------------------------------------------------------------

def insertion_phases(
    tcp: np.ndarray, close_idx: int, open_idx: int, n_frames: int
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Per-frame phase array (0-7) for the insertion schema.

    Returns (phases, insert_axis, degenerate). ``degenerate`` is True when the
    pre-release displacement is too small to define an axis, in which case the
    align/insert split falls back to a proportional cut and is reported.
    """
    phases = np.full(n_frames, -1, dtype=np.int32)

    # 0 approach / 1 grasp — same convention as the planar schema: the pick label
    # starts ACTION_HORIZON frames early so a chunk spanning the grasp is labelled.
    pick_start = max(0, close_idx - ACTION_HORIZON)
    pick_end = min(close_idx + PICK_SETTLE, open_idx)
    phases[:pick_start] = 0
    phases[pick_start:pick_end] = 1

    # 6 release starts AT the gripper event — unlike the planar schema it is NOT
    # shifted ACTION_HORIZON frames early. Insert is the skill worth labelling, and
    # a 16-frame early release would swallow it whole whenever the push into the
    # slot is shorter than one chunk (which it was on the pilot: insert=0 frames).
    release_start = open_idx

    # Insertion axis: direction of travel over the frames just before release.
    lo = max(pick_end, open_idx - INSERT_AXIS_WINDOW)
    disp = tcp[open_idx] - tcp[lo]
    norm = float(np.linalg.norm(disp))
    degenerate = norm < 1.0
    axis = disp / norm if not degenerate else np.array([0.0, 0.0, -1.0])

    # Projection of each frame onto the axis, measured from the release point.
    # Negative = still short of the slot, 0 = at the release pose.
    proj = (tcp - tcp[open_idx]) @ axis

    if degenerate:
        span = release_start - pick_end
        a_start = pick_end + int(span * 0.60)
        i_start = pick_end + int(span * 0.85)
    else:
        # Walk back from release: insert starts where the arm was still
        # INSERT_ENTER_MM short of the slot, align where it was ALIGN_ENTER_MM short.
        i_start = release_start
        for i in range(release_start - 1, pick_end - 1, -1):
            if proj[i] < -INSERT_ENTER_MM:
                i_start = i + 1
                break
        a_start = i_start
        for i in range(i_start - 1, pick_end - 1, -1):
            if proj[i] < -ALIGN_ENTER_MM:
                a_start = i + 1
                break

    # 2 lift / 3 carry inside [pick_end, a_start): lift is the first third.
    carry_span = a_start - pick_end
    lift_end = pick_end + max(1, int(carry_span * 0.35)) if carry_span > 1 else a_start
    phases[pick_end:lift_end] = 2
    phases[lift_end:a_start] = 3
    phases[a_start:i_start] = 4
    phases[i_start:release_start] = 5
    phases[release_start:] = 6

    # 7 retract — after release, once the TCP has backed off along the axis.
    for i in range(open_idx, n_frames):
        if proj[i] > RETRACT_EXIT_MM:
            phases[i:] = 7
            break

    phases[phases < 0] = 3
    return phases, axis, degenerate


# ---------------------------------------------------------------------------
# Planar phase segmentation (identical to E6 — motion signal is max over ALL
# joints, so it does not assume any particular axis carries the transport motion)
# ---------------------------------------------------------------------------

def _compute_carry_phases(joints: np.ndarray, carry_start: int, carry_end: int) -> tuple[int, int, bool]:
    """Return (transport_start, transport_end, used_fallback)."""
    if carry_end <= carry_start:
        return carry_start, carry_start, False

    delta = np.abs(np.diff(joints, axis=0, prepend=joints[:1]))   # (n, 6)
    motion_max = np.max(delta, axis=1)
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    motion_sm = np.convolve(motion_max, kernel, mode="same")

    seg = motion_sm[carry_start:carry_end]
    thresh = np.percentile(seg, TRANSPORT_PERCENTILE)

    # Degenerate: barely any motion in the carry segment → proportional fallback.
    # NOTE thresholds are absolute degrees, tuned on E6. If xArm6 Δq lives at a
    # different scale this fires spuriously — the report counts how often it does.
    if thresh < 0.05 or seg.max() < 0.10:
        span = carry_end - carry_start
        return carry_start + int(span * 0.30), carry_start + int(span * 0.75), True

    transport_start = carry_end
    for i in range(carry_start, carry_end):
        if motion_sm[i] > thresh:
            transport_start = i
            break
    transport_end = transport_start
    for i in range(carry_end - 1, transport_start - 1, -1):
        if motion_sm[i] > thresh:
            transport_end = i + 1
            break
    return transport_start, transport_end, False


def compute_phases(df: pd.DataFrame, pick_idx: int, open_idx: int) -> tuple[np.ndarray, bool]:
    """Per-frame phase array (0-5) plus a flag for whether the fallback split fired.

    Action-horizon-aware shifts (same as E6): the pick phase starts ACTION_HORIZON
    frames BEFORE close_idx and the release phase ACTION_HORIZON before open_idx, so
    an action chunk that carries the grasp/release event is already labeled with it.
    """
    n = len(df)
    phases = np.full(n, -1, dtype=np.int32)

    pick_start = max(0, pick_idx - ACTION_HORIZON)
    pick_end = min(pick_idx + PICK_SETTLE, open_idx)
    phases[:pick_start] = 0
    phases[pick_start:pick_end] = 1

    carry_start = pick_end
    release_start = max(carry_start, open_idx - ACTION_HORIZON)
    phases[release_start:] = 5

    used_fallback = False
    if carry_start < release_start:
        joints = df[list(JOINT_COLS)].values.astype(np.float64)
        t_start, t_end, used_fallback = _compute_carry_phases(joints, carry_start, release_start)
        t_end = min(t_end, release_start)
        phases[carry_start:t_start] = 2
        phases[t_start:t_end] = 3
        phases[t_end:release_start] = 4

    phases[phases < 0] = 3
    return phases, used_fallback


# ---------------------------------------------------------------------------
# Integrity guards
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GuardReport:
    dt_outliers: int = 0
    dt_unit: str = ""   # detected, not assumed — see check_integrity
    dropped_flags: int = 0
    ncmd_values: dict[int, int] = dataclasses.field(default_factory=dict)
    missing_cols: set[str] = dataclasses.field(default_factory=set)
    mode_kept: int = 0       # frames inside the selected teleop run
    mode_dropped: int = 0    # frames outside it (scripted moves, idle)
    # Other teleop runs in the episode — real human motion (often a manual
    # retract after release) that is NOT spliced in, because a position-mode
    # move usually sits between the runs and concatenating would fabricate a
    # jump in the joint trajectory. Reported so the loss is visible.
    dropped_teleop_runs: list[str] = dataclasses.field(default_factory=list)
    joint_spikes: int = 0    # frames where all six joints held one broadcast scalar
    rig_layout_mismatch: list[str] = dataclasses.field(default_factory=list)
    rig_geometry_mismatch: list[str] = dataclasses.field(default_factory=list)
    rig_geometry_unusable: list[str] = dataclasses.field(default_factory=list)
    missing_prompt_eps: list[str] = dataclasses.field(default_factory=list)
    # Episodes whose enum fields were too incomplete to render a prompt, so the
    # collector's baked string was used instead. Those episodes are NOT under the
    # requested template and must not be pooled into a template comparison.
    unrendered_eps: list[str] = dataclasses.field(default_factory=list)
    rule_conflicts: list[str] = dataclasses.field(default_factory=list)
    # Stretches where the arm was commanded but did not move (velocity limit,
    # joint limit, collision). State barely changes and the delta action is
    # ~0, so training on them teaches "hold still here" when the truth is
    # "blocked here" -- a direct cause of a policy that freezes on hardware.
    stalls: list[str] = dataclasses.field(default_factory=list)
    stall_frames: int = 0
    # Frames where a joint crossed a +-360 wrap. The adjacent difference there
    # is a ~-360 deg/frame artefact, and the spike repair would silently turn
    # it into a fabricated value instead of flagging it.
    joint_wraps: list[str] = dataclasses.field(default_factory=list)
    wrap_fixed_frames: int = 0   # deltas rewritten to the shortest path
    # Fraction of frames whose joint vector is bit-identical to the previous
    # one. With feedback at F Hz sampled at 16 Hz this sits near max(0, 1-F/16),
    # so a high value means the driver is publishing slower than it claims.
    zero_dq_frac: dict[str, float] = dataclasses.field(default_factory=dict)
    unconfirmed_category_eps: list[str] = dataclasses.field(default_factory=list)
    # Episodes whose prompt names one shelf while the arm released at another.
    # The single most damaging label defect: the policy is shown text that
    # contradicts the trajectory, so it learns the text is noise.
    target_actual_mismatch: list[str] = dataclasses.field(default_factory=list)
    # Landing not recorded at all — usually no gripper release. Tracked apart from
    # a wrong-shelf landing so the two failures are not conflated in triage.
    no_landing_eps: list[str] = dataclasses.field(default_factory=list)
    release_distances: list[float] = dataclasses.field(default_factory=list)
    layouts_seen: dict[str, int] = dataclasses.field(default_factory=dict)


# "Not moving" has to be judged against how fast this episode moves at all. A
# careful insertion can travel well under 1 mm/frame, so a fixed millimetre bar
# flags the whole approach; a stall is TCP travel collapsing to sensor noise
# while the command stays up. Hence a fraction of the episode's own median step,
# floored at the noise level.
STALL_TCP_FLOOR_MM = 0.15   # below this is measurement noise regardless of scale
STALL_TCP_FRAC = 0.10       # ...or a tenth of this episode's median step, whichever is larger
STALL_CMD_MIN = 0.05        # commanded twist magnitude above which the operator is "pushing"
STALL_MIN_FRAMES = 8        # 0.5 s at 16 Hz — shorter is just a pause between motions
WRAP_DEG = 180.0            # adjacent joint jump above this is a revolution wrap, not motion
ZERO_DQ_WARN = 0.20         # >20% repeated joint vectors => feedback slower than ~13 Hz


def detect_stalls(df: pd.DataFrame, ep_name: str, rep: GuardReport) -> np.ndarray:
    """Mark frames where the operator was commanding but the arm did not move.

    A stall is not a clamping artefact and not a pause: the twist is non-zero
    and the TCP is not travelling, which means the controller is refusing the
    motion. Those frames pair a near-zero action with a scene that looks like
    mid-reach, so they teach the policy to stop exactly where it should push on.
    """
    n = len(df)
    stalled = np.zeros(n, dtype=bool)
    if not ({"x", "y", "z"} <= set(df.columns)):
        rep.missing_cols.update({"x", "y", "z"})
        return stalled
    if not all(c in df.columns for c in TWIST_RAW_COLS):
        return stalled

    tcp = df[["x", "y", "z"]].to_numpy(float)
    step_mm = np.r_[0.0, np.linalg.norm(np.diff(tcp, axis=0), axis=1)]
    cmd = np.abs(df[list(TWIST_RAW_COLS)].to_numpy(float)).max(axis=1)
    thresh = max(STALL_TCP_FLOOR_MM, STALL_TCP_FRAC * float(np.median(step_mm[1:])))
    candidate = (step_mm < thresh) & (cmd > STALL_CMD_MIN)

    # Keep only runs long enough to be a real block rather than a turnaround.
    start = None
    for i in range(n + 1):
        if i < n and candidate[i]:
            start = i if start is None else start
            continue
        if start is not None and i - start >= STALL_MIN_FRAMES:
            stalled[start:i] = True
            rep.stalls.append(
                f"ep{ep_name} f{start}-{i - 1} ({i - start}f={(i - start) / 16.0:.1f}s) "
                f"cmd~{cmd[start:i].mean():.2f} tcp~{step_mm[start:i].mean():.3f}mm/f "
                f"(episode median {np.median(step_mm[1:]):.3f}, bar {thresh:.3f})"
            )
        start = None
    rep.stall_frames += int(stalled.sum())
    return stalled


def check_joint_wrap(df: pd.DataFrame, ep_name: str, rep: GuardReport) -> None:
    """Flag +-360 revolution wraps before the spike repair can hide them.

    J4 reaching +360 was what locked the controller into mode 0, so a wrap here
    is a hardware-side event to fix at the source, not a number to smooth over.
    """
    for c in JOINT_COLS:
        if c not in df.columns:
            continue
        d = np.diff(df[c].to_numpy(float))
        idx = np.flatnonzero(np.abs(d) > WRAP_DEG)
        for i in idx:
            rep.joint_wraps.append(
                f"ep{ep_name} {c} f{i}->{i + 1}: {df[c].iloc[i]:+.1f} -> "
                f"{df[c].iloc[i + 1]:+.1f} ({d[i]:+.1f} deg)"
            )


# Shelves nearer than this make the left/right ordering a coin flip rather than a
# measurement, so the check abstains instead of guessing. The rig measures 183 and
# 204 mm between neighbours, so this is a wide margin.
GEOM_MIN_SEPARATION_MM = 50.0


def _sides_from_waypoint_y(ys: dict) -> tuple[dict[str, str] | None, str | None]:
    """Derive {waypoint: side} from taught TCP y, independent of any side constant."""
    try:
        y = {str(k).strip(): float(v) for k, v in ys.items()}
    except (TypeError, ValueError):
        return None, f"shelf_waypoint_tcp_y_mm is not numeric: {ys!r}"
    if len(y) != len(RIG_LAYOUT):
        return None, f"expected {len(RIG_LAYOUT)} shelves, got {sorted(y)}"
    order = sorted(y, key=lambda k: y[k])          # +y is left; ascending is right->left
    gaps = [y[b] - y[a] for a, b in zip(order, order[1:])]
    if min(gaps) < GEOM_MIN_SEPARATION_MM:
        return None, (f"shelves only {min(gaps):.0f} mm apart in y "
                      f"(need {GEOM_MIN_SEPARATION_MM:.0f}); ordering is not decidable")
    return dict(zip(order, ("right", "center", "left"))), None


# The destination a demonstration was actually recorded against.
#
# `resolved_target_side` is the operator's declaration, taken from which teach
# button was pressed. It is the ONLY key read here, and that is deliberate.
#
# The collector used to derive the destination from the episode's shelf labels
# and the book's category. On the 08-04 batch that derivation returned "right"
# for all twelve episodes -- one category and a fixed arrangement always resolve
# to the same shelf -- while the operator had actually filled three shelves, four
# episodes each. The collector now writes the derived value under
# `label_derived_target_side` and leaves `resolved_target_side` null when no
# button was pressed, so an episode with no declaration is dropped rather than
# labelled by a table.
#
# `target_shelf` was in this list and had to come out. It still carries the
# derived value, so keeping it as a fallback reinstated exactly the mislabelling
# the collector had just been changed to prevent: null resolved_target_side would
# fall through and return "right" again, silently. A fallback that reaches a
# lower-authority field is not a safety net here, it is the bug.
#
# `legacy_static_target_shelf_id` is the pre-v5 static field and is likewise not
# consulted: it names a waypoint under the default arrangement, so reading it on
# a shuffled batch points at the wrong shelf.
RESOLVED_SIDE_KEYS = ("resolved_target_side",)
LEGACY_TARGET_KEYS = ("legacy_static_target_shelf_id", "target_shelf_id")
DERIVED_TARGET_KEYS = ("label_derived_target_side", "target_shelf")


def resolved_target_side(meta: dict) -> str:
    """Canonical destination side, or "" when the episode does not carry one."""
    for k in RESOLVED_SIDE_KEYS:
        v = str(meta.get(k) or "").strip().lower()
        if v:
            return v
    return ""


def derived_only_target(meta: dict) -> str | None:
    """A label-derived destination present while the declared one is absent.

    Reported rather than used. The distinction matters when reading a skip: an
    episode that names no destination at all was recorded without a teach press,
    while one carrying only a derived value had the press lost or never wired.
    """
    if resolved_target_side(meta):
        return None
    for k in DERIVED_TARGET_KEYS:
        if str(meta.get(k) or "").strip():
            return k
    return None


def legacy_only_target(meta: dict) -> str | None:
    """The legacy field, when it is the ONLY thing naming a destination."""
    if resolved_target_side(meta):
        return None
    for k in LEGACY_TARGET_KEYS:
        if meta.get(k) not in (None, ""):
            return k
    return None


def check_target_vs_actual(meta: dict, ep_name: str, rep: GuardReport) -> bool:
    """Does the shelf named in the prompt match the one the arm actually released at?

    The collector derives the landing shelf from the TCP at release, independently
    of any label. When the two disagree the episode shows the policy a prompt that
    contradicts the demonstration, which teaches it that the prompt is noise —
    strictly worse than dropping the episode. Returns False when it must be dropped.
    """
    target = resolved_target_side(meta)
    actual = ""
    for k in ACTUAL_SHELF_KEYS:
        if meta.get(k):
            actual = str(meta[k]).strip().lower()
            break
    for k in ACTUAL_DIST_KEYS:
        if meta.get(k) is not None:
            try:
                rep.release_distances.append(float(meta[k]))
            except (TypeError, ValueError):
                pass
            break

    # Two checks against RIG_LAYOUT, and they are not the same check.
    #
    # `shelf_layout_physical` is derived on the collector from its own side
    # constant, so comparing it here compares two constants. That is worth doing —
    # it catches the two machines running different versions of the shared prompt
    # package, which is the likeliest way these drift apart — but it cannot catch a
    # shelf being physically moved, and if the side constant is itself wrong both
    # sides are wrong together and agree.
    #
    # `shelf_waypoint_tcp_y_mm` is the taught TCP position of each shelf. It comes
    # from the waypoint file, not from any side constant, so it is the one field
    # that can contradict them. On this arm +y points left, so sorting the shelves
    # by y ascending gives right, centre, left. If that convention is wrong the
    # derived order is reversed and this fires on the first episode, which is the
    # failure we want.
    phys = meta.get("shelf_layout_physical")
    if isinstance(phys, dict) and phys:
        got = {str(k).strip(): str(v).strip().lower() for k, v in phys.items()}
        if got != dict(RIG_LAYOUT):
            rep.rig_layout_mismatch.append(
                f"ep{ep_name}: collector constant {got} vs contract {dict(RIG_LAYOUT)}")

    ys = meta.get("shelf_waypoint_tcp_y_mm")
    if isinstance(ys, dict) and ys:
        derived, why = _sides_from_waypoint_y(ys)
        if why:
            rep.rig_geometry_unusable.append(f"ep{ep_name}: {why}")
        elif derived != dict(RIG_LAYOUT):
            rep.rig_geometry_mismatch.append(
                f"ep{ep_name}: taught positions say {derived}, contract says {dict(RIG_LAYOUT)}")

    layout = meta.get("shelf_layout")
    if layout:
        key = json.dumps(layout, sort_keys=True, ensure_ascii=False) if isinstance(layout, dict) else str(layout)
        rep.layouts_seen[key] = rep.layouts_seen.get(key, 0) + 1

    if target and not actual:
        # No landing recorded — almost always "the gripper never released", which
        # is a different defect from "released at the wrong shelf" and gets caught
        # by the place-crossing check with a message that names the real cause.
        # Counting it as a mismatch here would misdirect the diagnosis.
        rep.no_landing_eps.append(ep_name)
        return True
    if not (target and actual):
        return True  # collector did not record either; nothing to compare
    if target != actual:
        rep.target_actual_mismatch.append(f"ep{ep_name}: prompt says {target!r}, released at {actual!r}")
        return False
    return True


def check_feedback_rate(df: pd.DataFrame, ep_name: str, rep: GuardReport) -> None:
    """Repeated joint vectors betray a driver publishing below the record rate.

    joint_states dropping to 5 Hz while recording at 16 Hz leaves ~69% of frames
    holding the previous values, and the resulting zero actions look like
    deliberate stillness to the loss.
    """
    cols = [c for c in JOINT_COLS if c in df.columns]
    if len(cols) < len(JOINT_COLS) or len(df) < 2:
        return
    q = df[cols].to_numpy(float)
    frac = float((np.abs(np.diff(q, axis=0)).max(axis=1) == 0.0).mean())
    rep.zero_dq_frac[ep_name] = frac


def check_integrity(df: pd.DataFrame, fps: float, dt_tol: float, rep: GuardReport) -> np.ndarray:
    """Return a per-frame bool mask: True = this frame's *outgoing* pair is usable.

    A pair (t, t+1) is unusable when the interval between them is off-nominal or a
    camera pair was dropped — the adjacent-difference action at that index would be
    the accumulation of more than one frame period.
    """
    n = len(df)
    ok = np.ones(n, dtype=bool)
    nominal = 1.0 / fps

    if COL_DT in df.columns:
        dt = df[COL_DT].values.astype(np.float64)
        # The collector's unit table calls this milliseconds; earlier data read as
        # seconds. Guessing wrong does not warn — it rejects every pair in the batch
        # with a timing complaint, which reads like a hardware fault. The two
        # candidates are three orders of magnitude apart, so detect instead: at 16 Hz
        # the median is either ~0.0625 or ~62.5, and nothing sits between them.
        finite = dt[np.isfinite(dt) & (dt > 0)]
        if finite.size and np.median(finite) > nominal * 100:
            dt = dt / 1000.0
            rep.dt_unit = "ms"
        else:
            rep.dt_unit = "s"
        # dt[t] describes the gap BEFORE frame t, so it invalidates pair (t-1, t).
        bad = np.abs(dt - nominal) > dt_tol * nominal
        bad[0] = False  # first frame has no predecessor
        rep.dt_outliers += int(bad.sum())
        ok[np.clip(np.flatnonzero(bad) - 1, 0, n - 1)] = False
    else:
        rep.missing_cols.add(COL_DT)

    if COL_DROPPED in df.columns:
        dropped = df[COL_DROPPED].values.astype(bool)
        rep.dropped_flags += int(dropped.sum())
        ok[np.clip(np.flatnonzero(dropped) - 1, 0, n - 1)] = False
    else:
        rep.missing_cols.add(COL_DROPPED)

    if COL_NCMD in df.columns:
        for v, c in zip(*np.unique(df[COL_NCMD].values.astype(int), return_counts=True)):
            rep.ncmd_values[int(v)] = rep.ncmd_values.get(int(v), 0) + int(c)
    else:
        rep.missing_cols.add(COL_NCMD)

    return ok


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_image_hwc(path: Path, resize: int | None) -> np.ndarray:
    img = PIL.Image.open(path).convert("RGB")
    if resize is not None:
        img = img.resize((resize, resize), PIL.Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _read_csv(episode_dir: Path, csv_name: str) -> pd.DataFrame:
    p = episode_dir / csv_name
    if not p.is_file():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if len(df) < 20:
        raise ValueError(f"{p}: too few rows ({len(df)})")
    return df


def _read_meta(episode_dir: Path) -> dict:
    p = episode_dir / "episode_meta.json"
    if not p.is_file():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    # vlm_result.json holds what the on-device classifier SAID. episode_meta holds
    # what the operator CONFIRMED. They are merged under distinct keys and the
    # prediction never overwrites the confirmed label -- the prompt must be built
    # from the truth the demonstrator acted on, or the policy is taught that the
    # prompt is noise.
    v = episode_dir / "vlm_result.json"
    if v.is_file():
        with v.open("r", encoding="utf-8") as f:
            vlm = json.load(f)
        for src, dst in (
            ("category", "category_predicted"),
            ("confidence", "category_confidence"),
            ("model_version", "vlm_model_version"),
            ("request_id", "mcp_request_id"),
        ):
            if src in vlm and dst not in meta:
                meta[dst] = vlm[src]
        meta.setdefault("category_source", "external_vlm")
    return meta


def _resolve_episodes(episode_dir: list[Path] | None, root: Path | None,
                      exclude: tuple[int, ...]) -> list[Path]:
    if episode_dir:
        return [Path(p).resolve() for p in episode_dir]
    if root is None:
        raise ValueError("Provide --episode-dir or --root.")
    root = Path(root).expanduser().resolve()
    excluded = set(exclude)
    candidates = [(int(c.name), c) for c in root.iterdir()
                  if c.is_dir() and c.name.isdigit() and int(c.name) not in excluded]
    candidates.sort(key=lambda x: x[0])
    if not candidates:
        raise ValueError(f"No numeric episode dirs under {root}")
    return [p for _, p in candidates]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    root: Path | None = None,
    episode_dir: list[Path] | None = None,
    exclude: tuple[int, ...] = (),
    csv_name: str = "robot_data.csv",
    images_subdir: str = "images",
    fps: int = 16,
    robot_type: str = "xarm6",
    resize: int | None = None,
    # --- gripper handling -------------------------------------------------
    gripper_source: Literal["command", "trigger_raw"] = "command",
    gripper_threshold: float = 0.5,
    gripper_binarize: bool = False,
    # --- phase schema ------------------------------------------------------
    schema: Literal["auto", "planar", "insertion"] = "auto",
    # --- prompt ------------------------------------------------------------
    prompt_source: Literal["rendered", "meta", "phase"] = "rendered",
    prompt_style: Literal["resolved", "category_only", "single_rule", "rule_table"] = "resolved",
    # Config the checkpoint will be trained under. Only read to stamp the prompt
    # contract manifest -- the tokenizer settings that change the token sequence
    # (max_token_len, discrete_state_input) live there, not here.
    config_name: str = "pi05_e7_v1_lora",
    # Declared rule set, ``{rule_version: {category: destination}}``. Required in
    # practice for --prompt-style rule_table: a table derived from the episodes
    # only contains the categories that were collected, which silently degrades
    # the prompt to single_rule. Same artefact the MCP server publishes.
    rule_table_file: Path | None = None,
    # Lower only to inspect archived data; never for a batch that will be trained on.
    min_schema_version: int = MIN_SCHEMA_VERSION,
    # A probe episode exercises the file format without a real scene: no shelf signs,
    # synthetic motion. It looks entirely normal to every other check here, which is
    # exactly why it needs its own switch — off by default.
    allow_probe: bool = False,
    # --- action contract ---------------------------------------------------
    action_semantics: Literal["sequential", "current_relative"] = "sequential",
    # --- segmentation ------------------------------------------------------
    use_robot_mode: bool = True,
    teleop_mode: int = TELEOP_MODE,
    # --- integrity --------------------------------------------------------
    on_gap: Literal["skip", "warn"] = "skip",
    dt_tol: float = 0.5,
    check_meta: bool = True,
    # --- output -----------------------------------------------------------
    clean: bool = True,
    push_to_hub: bool = False,
    hub_private: bool = False,
    image_writer_threads: int = 10,
    image_writer_processes: int = 4,
) -> None:
    """Convert xArm6 raw episodes to a LeRobot dataset.

    Args:
        gripper_source: which column drives BOTH phase detection and the action
            label. ``command`` = rate-gated goal (what the robot received);
            ``trigger_raw`` = ungated human trigger (no 0.4 s quantisation).
        gripper_threshold: level used for the 0↔1 crossing that marks pick/place.
        gripper_binarize: emit the gripper action as {0.0, 1.0} (E6-identical
            contract) instead of passing the continuous value through.
        schema: phase schema. ``auto`` picks ``insertion`` when
            episode_meta["task_id"] contains "insert", else ``planar``.
        prompt_style: rule clause used when ``prompt_source="rendered"``.
            ``single_rule`` gives only the episode's own rule (the answer is right
            of the arrow), ``rule_table`` gives all three rows so the category must
            be matched, ``resolved`` names the destination and states no rule.
            Changing this and re-running re-labels an existing batch — no
            re-collection is needed, which is why the template is decided here
            and not in episode_meta.
        prompt_source: what goes into the LeRobot ``task`` field, i.e. the string the
            policy is conditioned on. ``meta`` (default) uses episode_meta["prompt"]
            unchanged for every frame — required for the MCP track, because that is
            the only string carrying the rule. ``phase`` restores the E6 behaviour of
            a per-phase task string; it names the destination directly, so use it
            only for the E6-comparable ablation. Phases are segmented either way and
            still drive the report.
        action_semantics: ``sequential`` writes q[t+1]-q[t] (E6-compatible, default).
            ``current_relative`` writes absolute q[t+1] and REQUIRES ``DeltaActions``
            in the data config — see the module docstring.
        use_robot_mode: trim to the ``robot_mode == teleop_mode`` run containing the
            grasp, dropping scripted home/anchor moves. Disable only for datasets
            with no ``robot_mode`` column (then E6's first-motion trim is used, which
            will include scripted moves).
        teleop_mode: xArm control mode that means "teleoperated" (5 = Cartesian velocity).
        check_meta: validate episode_meta.json against the policy contract
            (joint_unit / control_mode / command_semantics).
        on_gap: ``skip`` drops an episode that contains any dt/drop violation;
            ``warn`` converts it anyway and only reports. Use ``warn`` to inspect
            pilot data, ``skip`` for production.
        dt_tol: relative tolerance on the frame interval (0.5 = ±50% of 1/fps).
    """
    episode_paths = _resolve_episodes(episode_dir, root, exclude)
    gcol = GRIPPER_COL_GATED if gripper_source == "command" else GRIPPER_COL_RAW

    # Pre-pass: the rule_table template needs every row of a rule_version, which
    # only exists across episodes, so the tables are built before conversion.
    #
    # A DERIVED table is only as complete as what happened to be collected. With
    # one category recorded, `rules: humanities->right` is what comes out — which
    # is `single_rule` wearing a rule_table label, and the whole manipulation is
    # gone. The rule set is a DECLARED artefact (the same table the MCP server
    # publishes as a Resource), so pass it in with --rule-table-file and let the
    # derived one serve as a cross-check on the episode labels.
    rule_tables, rule_conflicts = build_rule_tables(episode_paths)
    if rule_table_file is not None:
        declared = json.loads(Path(rule_table_file).read_text(encoding="utf-8"))
        declared = {str(v): {canonical_category(c): str(t) for c, t in tbl.items()}
                    for v, tbl in declared.items()}
        for ver, dtbl in sorted(declared.items()):
            for cat, got in sorted(rule_tables.get(ver, {}).items()):
                want = dtbl.get(cat)
                if want is not None and want != got:
                    rule_conflicts.append(
                        f"declared {ver}:{cat}->{want!r} but episodes show {got!r} "
                        f"— either the table or the episode labels are wrong"
                    )
        unknown = sorted(set(rule_tables) - set(declared))
        if unknown:
            rule_conflicts.append(f"episodes use rule_version(s) absent from the table: {unknown}")
        rule_tables = declared

    output_root = HF_LEROBOT_HOME / repo_id
    if clean and output_root.exists():
        shutil.rmtree(output_root)

    df0 = _read_csv(episode_paths[0], csv_name)
    h_hik, w_hik, _ = _load_image_hwc(episode_paths[0] / images_subdir / str(df0[IMAGE_COL_HIK].iloc[0]), resize).shape
    h_zed, w_zed, _ = _load_image_hwc(episode_paths[0] / images_subdir / str(df0[IMAGE_COL_ZED].iloc[0]), resize).shape

    features = {
        "exterior_image_1_left": {"dtype": "image", "shape": (h_hik, w_hik, 3),
                                  "names": ["height", "width", "channel"]},
        "exterior_image_2_left": {"dtype": "image", "shape": (h_zed, w_zed, 3),
                                  "names": ["height", "width", "channel"]},
        "state":  {"dtype": "float32", "shape": (7,),
                   "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]},
        "action": {"dtype": "float32", "shape": (7,),
                   "names": ["dj1", "dj2", "dj3", "dj4", "dj5", "dj6", "gripper_abs"]},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done":   {"dtype": "bool",    "shape": (1,), "names": None},
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, robot_type=robot_type, features=features,
        use_videos=False,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )
    registered_objects: set[str] = set()
    object_counts: dict[str, int] = {}
    schema_counts: dict[str, int] = {}
    insert_axes: list[np.ndarray] = []
    prompt_counts: dict[str, int] = {}
    context_counts: dict[str, dict[str, int]] = {}
    cat_dest_counts: dict[tuple[str, str], int] = {}
    episode_context: dict[int, dict[str, str]] = {}

    total_frames = 0
    skipped_eps = 0
    fallback_eps = 0
    phase_counts: dict[tuple[str, int], int] = {}
    guard = GuardReport()
    guard.rule_conflicts = rule_conflicts
    all_actions: list[np.ndarray] = []
    grip_values: list[np.ndarray] = []
    gating_lags: list[int] = []
    clamp_mag: list[np.ndarray] = []

    for ep_idx, ep in enumerate(episode_paths):
        try:
            df_raw = _read_csv(ep, csv_name)
            meta = _read_meta(ep)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  SKIP {ep.name}: {exc}")
            skipped_eps += 1
            continue

        missing = [c for c in [*JOINT_COLS, gcol, IMAGE_COL_HIK, IMAGE_COL_ZED]
                   if c not in df_raw.columns]
        if missing:
            print(f"  SKIP {ep.name}: missing columns {missing}")
            skipped_eps += 1
            continue

        # An unconfirmed category means nobody checked the classifier before the
        # demonstration. If it was wrong the prompt names one class while the
        # operator placed another, which trains the policy to ignore the prompt —
        # the exact opposite of what this dataset is for. Absent field = legacy
        # episode, warn only; present-and-false = drop.
        sv = meta.get("schema_version")
        if sv is None or int(sv) < min_schema_version:
            print(f"  SKIP {ep.name}: schema_version={sv!r} < {min_schema_version}.")
            if sv is not None and int(sv) >= 4:
                print(f"       v4 has no resolved_target_* — only the static field, which "
                      f"names a waypoint under the DEFAULT arrangement. On a shuffled batch "
                      f"that points at the wrong shelf, so it is not promoted. Re-record.")
            else:
                print(f"       Pre-v4 uses the old taxonomy (교양책→humanities); mixing it "
                      f"silently sends a whole category to the wrong shelf. Relabel or discard.")
            skipped_eps += 1
            continue

        # Belt and braces: a v5-stamped episode that still only carries the legacy
        # field means the collector stamped the version without emitting the field.
        if (legacy_key := legacy_only_target(meta)) is not None:
            print(f"  SKIP {ep.name}: schema says v{sv} but the only destination field is "
                  f"{legacy_key!r}. That one is the static default arrangement, not what "
                  f"this episode was recorded against, and promoting it is exactly the "
                  f"migration that must not happen.")
            skipped_eps += 1
            continue

        if not allow_probe and "schema_probe" in str(meta.get("book_id") or ""):
            print(f"  SKIP {ep.name}: book_id marks this a schema probe. It has no shelf "
                  f"signs and its motion is synthetic, so it would train the policy on "
                  f"a scene that does not exist. Pass --allow-probe to convert anyway.")
            skipped_eps += 1
            continue

        # The classifier fields are required only when the classifier was used.
        # A manual pilot correctly leaves them null, and null means "never called" —
        # which is a different fact from "called and could not decide".
        if str(meta.get("category_source") or "").strip().lower() == "mcp":
            missing_mcp = [k for k in ("mcp_status", "category_confidence", "vlm_model_version")
                           if meta.get(k) in (None, "")]
            if missing_mcp:
                print(f"  SKIP {ep.name}: category_source=mcp but {missing_mcp} are absent. "
                      f"A classifier-sourced label with no record of what the classifier "
                      f"said cannot be audited against it later.")
                skipped_eps += 1
                continue

        if "category_confirmed" in meta and not meta["category_confirmed"]:
            print(f"  SKIP {ep.name}: category_confirmed=false (operator did not "
                  f"verify the classifier before demonstrating)")
            skipped_eps += 1
            continue
        if "category_confirmed" not in meta and meta.get("category_source") == "external_vlm":
            guard.unconfirmed_category_eps.append(ep.name)

        cat_norm = canonical_category(str(meta.get("category") or meta.get("object_category") or ""))
        tgt_norm = resolved_target_side(meta)
        if cat_norm and cat_norm not in CANONICAL_CATEGORIES:
            print(f"  SKIP {ep.name}: category {cat_norm!r} not in {CANONICAL_CATEGORIES}. "
                  f"Unmapped values pass through canonicalisation unchanged, so a typo "
                  f"would render as 'category={cat_norm}.' and reach the tokenizer.")
            skipped_eps += 1
            continue
        if tgt_norm and tgt_norm not in CANONICAL_DESTINATIONS:
            print(f"  SKIP {ep.name}: resolved target side {tgt_norm!r} not in {CANONICAL_DESTINATIONS}")
            skipped_eps += 1
            continue

        if not check_target_vs_actual(meta, ep.name, guard):
            print(f"  SKIP {ep.name}: prompt names {resolved_target_side(meta)!r} but the arm "
                  f"released at {[meta.get(k) for k in ACTUAL_SHELF_KEYS if meta.get(k)][0]!r}")
            skipped_eps += 1
            continue

        # -- repair scalar-broadcast joint frames before anything reads them ---
        n_spikes = repair_joint_spikes(df_raw, JOINT_COLS)
        if n_spikes:
            guard.joint_spikes += n_spikes
            print(f"  FIX  {ep.name}: repaired {n_spikes} joint spike frame(s) "
                  f"(all six joints equal — collector wrote a scalar)")

        # -- gating lag diagnostic (needs both gripper columns) -------------
        # Pair each gated crossing with the NEAREST raw crossing, not first-with-first:
        # the raw trigger contains sub-second transients (pilot ep0 has a 3-frame
        # accidental squeeze at f365-367) and naive pairing reports a bogus 180-frame
        # lag. Unmatched crossings beyond the window are ignored.
        if GRIPPER_COL_GATED in df_raw.columns and GRIPPER_COL_RAW in df_raw.columns:
            g_gated = df_raw[GRIPPER_COL_GATED].values.astype(np.float32)
            g_raw = df_raw[GRIPPER_COL_RAW].values.astype(np.float32)
            for up in (True, False):
                gc = all_crossings(g_gated, gripper_threshold, up)
                rc = all_crossings(g_raw, gripper_threshold, up)
                for b in gc:
                    if not rc:
                        continue
                    a = min(rc, key=lambda x: abs(x - b))
                    if abs(b - a) <= GATING_MATCH_WINDOW:
                        gating_lags.append(b - a)

        gripper_raw_series = df_raw[gcol].values.astype(np.float32)
        grip_values.append(gripper_raw_series.copy())

        # -- contract check --------------------------------------------------
        if check_meta:
            bad = {k: meta.get(k) for k, v in META_CONTRACT.items()
                   if k in meta and meta[k] != v}
            if bad:
                print(f"  SKIP {ep.name}: episode_meta violates contract {bad} "
                      f"(expected {META_CONTRACT}); rerun with --no-check-meta to override")
                skipped_eps += 1
                continue
            if abs(float(meta.get("record_rate_hz", fps)) - fps) > 0.5:
                print(f"  WARN {ep.name}: record_rate_hz={meta.get('record_rate_hz')} vs fps={fps}")

        # -- locate the grasp ------------------------------------------------
        # Take the LONGEST close→open pair, not the first: a book that slips is
        # re-gripped, and the first crossing then points at the failed attempt.
        pairs = grasp_pairs(gripper_raw_series, gripper_threshold)
        grasp = select_grasp(pairs)
        if grasp is None:
            print(f"  SKIP {ep.name}: gripper {gripper_threshold} crossing missing in '{gcol}' "
                  f"(range {gripper_raw_series.min():.3f}~{gripper_raw_series.max():.3f})")
            skipped_eps += 1
            continue
        close_idx, open_idx = grasp
        if len(pairs) > 1:
            shown = ", ".join(f"{c}→{o}({o - c}f)" for c, o in pairs)
            print(f"  NOTE {ep.name}: {len(pairs)} grasps [{shown}] — kept the longest carry "
                  f"{close_idx}→{open_idx}")

        # -- trim ------------------------------------------------------------
        # The demonstration is the run of operator-driven frames containing the
        # grasp; everything outside it is a scripted home/anchor move or idle time.
        lo = 0
        if use_robot_mode and (COL_MODE in df_raw.columns or COL_TELEOP in df_raw.columns):
            active = demonstration_mask(df_raw, teleop_mode)
            runs = bridged_runs(active, GATE_BRIDGE_FRAMES)
            run = next((r for r in runs if r[0] <= close_idx and r[1] >= open_idx), None)
            if run is None:
                print(f"  SKIP {ep.name}: grasp (close@{close_idx}, open@{open_idx}) spans a break in "
                      f"operator control; runs={runs[:6]}{'...' if len(runs) > 6 else ''}")
                skipped_eps += 1
                continue
            lo, hi_run = run[0], run[1] + 1
            # Start after any earlier abandoned grasp inside the run. Without this,
            # ep1's window would open at f224 and carry the failed 288→302 attempt
            # into the "approach" phase — teaching the policy to grip and let go.
            prior = [o for _, o in pairs if o <= close_idx]
            if prior:
                lo = max(lo, max(prior) + 1)
            guard.mode_kept += hi_run - lo
            guard.mode_dropped += len(df_raw) - (hi_run - lo)
            hi = min(open_idx + PLACE_SETTLE, hi_run)

            # Any OTHER active run is teleoperated too — most often a manual retract
            # after the release. Dropping it silently would throw away real
            # demonstration, so report every one with its TCP displacement. They are
            # not spliced in: an inactive stretch sits between the runs, so
            # concatenating would fabricate a jump in the joint trajectory.
            for a, b in runs:
                if (a, b) == run or b - a < 2:
                    continue
                seg = df_raw.iloc[a:b + 1]
                dz = ez = ""
                if {"x", "y", "z"} <= set(df_raw.columns):
                    t0, t1 = seg[["x", "y", "z"]].values[0], seg[["x", "y", "z"]].values[-1]
                    dz, ez = f"  TCP Δz {t1[2] - t0[2]:+7.1f}mm", f" Δxy {np.linalg.norm(t1[:2] - t0[:2]):6.1f}mm"
                guard.dropped_teleop_runs.append(
                    f"ep{ep.name} f{a}-{b} ({b - a + 1}f={(b - a + 1) / fps:.1f}s){dz}{ez}"
                    + ("  [after release]" if a >= open_idx else "")
                )
        else:
            if use_robot_mode:
                guard.missing_cols.add(COL_MODE)
            hi = min(open_idx + PLACE_SETTLE, len(df_raw))

        df = df_raw.iloc[lo:hi].reset_index(drop=True)
        # Leading trim inside the teleop run: drop the still frames before motion.
        start = max(0, find_first_motion(df) - 1)
        df = df.iloc[start:].reset_index(drop=True)
        trim_lo = lo + start

        if len(df) < ACTION_HORIZON:
            print(f"  SKIP {ep.name}: too short after trim ({len(df)} frames)")
            skipped_eps += 1
            continue

        # -- data-quality guards ----------------------------------------------
        # All three run on the TRIMMED frame so the numbers describe what
        # actually reaches training, not the scripted approach and idle tail.
        detect_stalls(df, ep.name, guard)
        check_joint_wrap(df, ep.name, guard)
        check_feedback_rate(df, ep.name, guard)

        # -- twist clamping diagnostic ---------------------------------------
        if all(c in df.columns for c in TWIST_RAW_COLS + TWIST_SENT_COLS):
            rawt = df[list(TWIST_RAW_COLS)].values.astype(np.float64)
            sentt = df[list(TWIST_SENT_COLS)].values.astype(np.float64)
            clamp_mag.append(np.abs(rawt - sentt).max(axis=1))

        # -- integrity ------------------------------------------------------
        pair_ok = check_integrity(df, fps, dt_tol, guard)
        n_bad = int((~pair_ok[:-1]).sum())
        if n_bad and on_gap == "skip":
            print(f"  SKIP {ep.name}: {n_bad} pair(s) violate dt/drop guard "
                  f"(rerun with --on-gap warn to convert anyway)")
            skipped_eps += 1
            continue
        if n_bad:
            print(f"  WARN {ep.name}: {n_bad} pair(s) violate dt/drop guard — converted anyway")

        # -- phase events in the trimmed frame of reference -------------------
        # Re-select with the SAME longest-carry rule, not the first crossing: the
        # trimmed window can still contain an earlier failed grasp (ep1 keeps
        # 288→302 alongside the real 340→417), and re-running find_close_idx here
        # would silently segment the phases around the failed one.
        gt = df[gcol].values.astype(np.float32)
        trimmed = select_grasp(grasp_pairs(gt, gripper_threshold))
        pick_idx, open_idx = trimmed if trimmed else (None, None)
        if pick_idx is None:
            print(f"  SKIP {ep.name}: no gripper {gripper_threshold}↑ (pick) crossing after trim")
            skipped_eps += 1
            continue
        if open_idx is None:
            print(f"  SKIP {ep.name}: no gripper {gripper_threshold}↓ (place) crossing after trim")
            skipped_eps += 1
            continue
        if pick_idx >= open_idx:
            print(f"  SKIP {ep.name}: pick_idx({pick_idx}) >= open_idx({open_idx})")
            skipped_eps += 1
            continue

        # -- phase segmentation ------------------------------------------------
        ep_schema = schema if schema != "auto" else (
            "insertion" if "insert" in str(meta.get("task_id", "")).lower() else "planar"
        )
        if ep_schema == "insertion":
            if not {"x", "y", "z"} <= set(df.columns):
                print(f"  SKIP {ep.name}: insertion schema needs TCP x/y/z columns")
                skipped_eps += 1
                continue
            tcp = df[["x", "y", "z"]].values.astype(np.float64)
            phase_arr, axis, used_fallback = insertion_phases(tcp, pick_idx, open_idx, len(df))
            insert_axes.append(axis)
        else:
            phase_arr, used_fallback = compute_phases(df, pick_idx, open_idx)
        fallback_eps += int(used_fallback)
        schema_counts[ep_schema] = schema_counts.get(ep_schema, 0) + 1

        # -- prompt -----------------------------------------------------------
        obj = str(meta.get("prompt_object_name") or meta.get("object_label") or DEFAULT_OBJECT)
        tgt = resolved_target_side(meta) or str(meta.get("shelf_color") or meta.get("target_color")
                  or DEFAULT_TARGET)
        phase_tasks = tasks_for(obj, tgt, ep_schema)

        # ``meta``  — episode_meta["prompt"], constant for the whole episode. This is
        #   the one that carries the MCP context ("category=physics. rule:
        #   physics->blue. ..."), so it is the only source under which a policy CAN
        #   learn to read the rule instead of memorising the destination. It also
        #   makes inference trivial: the executor sets one string per episode and
        #   never has to estimate which phase it is in.
        # ``phase`` — the E6-style per-phase string ("align the book with the blue
        #   shelf"). Names the destination outright, so the rule becomes redundant
        #   and the counterfactual test cannot separate reading from memorising.
        #   Kept for the E6-comparable ablation, not for the MCP track.
        # ``rendered`` — built here from the enum fields (category / shelf_color /
        #   rule_version) via ``render_prompt``. The template is experiment design
        #   and belongs to the training side, so this is the default: re-running
        #   the converter with a different --prompt-style re-labels an already
        #   collected batch, no re-collection.
        ep_prompt = None
        if prompt_source == "rendered":
            ep_prompt = render_prompt(meta, prompt_style, rule_tables)
            if ep_prompt is None and meta.get("prompt"):
                ep_prompt = str(meta["prompt"])   # fall back to the collector string
                guard.unrendered_eps.append(ep.name)
        elif prompt_source == "meta" and meta.get("prompt"):
            ep_prompt = str(meta["prompt"])

        if ep_prompt is not None:
            ep_tasks = [ep_prompt] * len(phase_tasks)
        else:
            if prompt_source in ("meta", "rendered"):
                guard.missing_prompt_eps.append(ep.name)
            ep_tasks = phase_tasks
        for task_str in dict.fromkeys(ep_tasks):
            if task_str not in registered_objects:
                dataset.meta.add_task(task_str)
                registered_objects.add(task_str)
        prompt_counts[ep_tasks[0]] = prompt_counts.get(ep_tasks[0], 0) + 1

        label = obj if ep_schema == "planar" else f"{obj}→{tgt}"
        object_counts[label] = object_counts.get(label, 0) + 1

        # Episode context — not a model input, only for grouping at evaluation.
        ep_context = {k: str(meta.get(k) or "") for k in CONTEXT_KEYS}
        for k in CONTEXT_KEYS:
            context_counts.setdefault(k, {})
            context_counts[k][ep_context[k]] = context_counts[k].get(ep_context[k], 0) + 1
        # Counted as a pair, not two independent tallies: the question is whether
        # category and destination vary TOGETHER, and marginal counts cannot say.
        # Twelve episodes split 4/4/4 over destinations look balanced in the
        # margin while every one of them carries the same category.
        if (_c := canonical_category(str(meta.get("category") or ""))) and (
                _d := resolved_target_side(meta)):
            cat_dest_counts[(_c, _d)] = cat_dest_counts.get((_c, _d), 0) + 1
        episode_context[len(episode_context)] = {
            "episode_dir": ep.name, "prompt": ep_tasks[0], "schema": ep_schema, **ep_context,
        }

        n_pairs = len(df) - 1
        ep_actions = np.empty((n_pairs, 7), dtype=np.float32)

        for t in range(n_pairs):
            cur, nxt = df.iloc[t], df.iloc[t + 1]

            joints_cur = np.array([float(cur[c]) for c in JOINT_COLS], dtype=np.float32)
            joints_nxt = np.array([float(nxt[c]) for c in JOINT_COLS], dtype=np.float32)
            g_cur, g_nxt = np.float32(cur[gcol]), np.float32(nxt[gcol])
            if gripper_binarize:
                g_cur = np.float32(g_cur >= gripper_threshold)
                g_nxt = np.float32(g_nxt >= gripper_threshold)

            state = np.concatenate([joints_cur, [g_cur]])
            # sequential      : Δq per frame — self-contained, E6-compatible.
            # current_relative: absolute q[t+1]; DeltaActions subtracts the chunk's
            #                   own state at load time to give q[t+k+1] - q[t].
            if action_semantics == "sequential":
                # Shortest-path delta. On a revolute joint, -4.6 deg -> +355.4 deg
                # is a 0.0 deg move, not a +360 one, and the naive difference is a
                # pure representation artefact. It also survives normalisation:
                # q01/q99 are percentiles, so one outlier in ~540 frames leaves the
                # divisor untouched and that frame normalises to ~4900 sigma, which
                # detonates the run (measured: dj4 std 15.5 vs 0.06-0.45 elsewhere,
                # loss 0.13 -> 2386). Wrapping into (-180, 180] is a no-op for every
                # real motion, since 180 deg/frame at 16 Hz is far past the hardware.
                raw_delta = joints_nxt - joints_cur
                wrapped = np.abs(raw_delta) > WRAP_DEG
                # Subtract a full turn only where one actually happened, so every
                # normal delta stays bit-exact (a modulo round-trip would perturb
                # all of them at the 1e-7 level and make the counter meaningless).
                joint_action = np.where(wrapped, raw_delta - np.sign(raw_delta) * 360.0, raw_delta)
                if wrapped.any():
                    guard.wrap_fixed_frames += 1
            else:
                joint_action = joints_nxt
            action = np.concatenate([joint_action, [g_nxt]])
            ep_actions[t] = action

            phase = int(phase_arr[t])
            phase_counts[(ep_schema, phase)] = phase_counts.get((ep_schema, phase), 0) + 1
            is_last = t == n_pairs - 1

            dataset.add_frame({
                "exterior_image_1_left": _load_image_hwc(ep / images_subdir / str(cur[IMAGE_COL_HIK]), resize),
                "exterior_image_2_left": _load_image_hwc(ep / images_subdir / str(cur[IMAGE_COL_ZED]), resize),
                "state": state,
                "action": action,
                "next.reward": np.array([1.0 if is_last else 0.0], dtype=np.float32),
                "next.done": np.array([is_last], dtype=bool),
                "task": ep_tasks[phase],
            })

        dataset.save_episode()
        all_actions.append(ep_actions)
        total_frames += n_pairs
        print(f"[{ep_idx + 1:3d}/{len(episode_paths)}] ep={ep.name:>4s}: "
              f"raw={len(df_raw)} → teleop[{trim_lo}:{trim_lo + len(df)}] "
              f"→ {n_pairs} pairs | pick@{pick_idx} place@{open_idx}"
              + ("  [phase fallback]" if used_fallback else ""))

    # Episode context sidecar. Deliberately NOT parquet columns: the rule reaches the
    # policy through the prompt only, and extra per-frame columns would just be
    # dropped by openpi's loader. Evaluation reads this to build (same book,
    # different rule) pairs from ``pair_id`` / ``rule_version``.
    ctx_path = HF_LEROBOT_HOME / repo_id / "meta" / "e7_context.json"
    ctx_path.parent.mkdir(parents=True, exist_ok=True)
    ctx_path.write_text(json.dumps(
        {"prompt_source": prompt_source, "prompt_style": prompt_style,
         "rule_tables": rule_tables, "episodes": episode_context},
        indent=2, ensure_ascii=False
    ), encoding="utf-8")

    # Prompt contract manifest. Ships next to the checkpoint; the Jetson client
    # recomputes the hash at startup and refuses to serve on a mismatch. Written
    # here rather than at train time because the rule tables are only knowable
    # once the episodes have been read.
    manifest = _build_contract_manifest(config_name, prompt_style, rule_tables)
    (ctx_path.parent / "prompt_contract.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _report(dataset, episode_paths, skipped_eps, total_frames, phase_counts,
            guard, all_actions, grip_values, gating_lags, fallback_eps,
            gcol, gripper_binarize, repo_id, action_semantics, clamp_mag, object_counts,
            schema_counts, insert_axes, prompt_source, prompt_counts, context_counts,
            prompt_style, rule_tables)

    if push_to_hub:
        dataset.push_to_hub(tags=["e7", "xarm6", "openpi", "velocity", "phase-prompt"],
                            private=hub_private, push_videos=False, license="apache-2.0")
        print(f"\nPushed to HuggingFace: {repo_id}")


def _report(dataset, episode_paths, skipped_eps, total_frames, phase_counts, guard,
            all_actions, grip_values, gating_lags, fallback_eps,
            gcol, gripper_binarize, repo_id, action_semantics, clamp_mag, object_counts,
            schema_counts, insert_axes, prompt_source, prompt_counts, context_counts,
            prompt_style, rule_tables) -> None:
    print("\n" + "=" * 72)
    print("CONVERSION REPORT")
    print("=" * 72)
    print(f"  Raw episodes   : {len(episode_paths)}  ({skipped_eps} skipped)")
    print(f"  Episodes saved : {len(episode_paths) - skipped_eps}")
    print(f"  Frame pairs    : {total_frames}")
    print(f"  Gripper source : {gcol}  (binarize={gripper_binarize})")
    print(f"  Action semantics: {action_semantics}"
          + ("   ⚠ requires DeltaActions in the data config"
             if action_semantics == "current_relative" else ""))

    if guard.mode_kept or guard.mode_dropped:
        tot = guard.mode_kept + guard.mode_dropped
        print(f"\n  Operator-control trim: kept {guard.mode_kept}/{tot} frames "
              f"({100.0 * guard.mode_kept / max(1, tot):.1f}%), "
              f"dropped {guard.mode_dropped} (scripted home/anchor moves + idle)")

    if guard.joint_spikes:
        print(f"  ⚠ joint spikes repaired: {guard.joint_spikes} frame(s) had all six joints "
              f"equal (collector wrote a scalar).")
        print("     Each one would otherwise fabricate two ±80-110 deg delta labels.")
        print("     Fix at the source: only record when len(joint_state.position) == 6.")

    if object_counts:
        print(f"  Objects        : {dict(sorted(object_counts.items()))}")

    # -- prompt: the only channel carrying the MCP rule ----------------------
    note = {
        "rendered": f"   (rendered here from enum fields, style={prompt_style})",
        "meta": "   (episode_meta['prompt'] verbatim — template owned by the collector)",
        "phase": "   ⚠ per-phase strings — names the destination, no counterfactual possible",
    }[prompt_source]
    print(f"\n  Prompt source  : {prompt_source}{note}")
    if prompt_source == "rendered" and prompt_style in ("resolved", "single_rule"):
        print(f"    ⚠ {prompt_style} names the destination in the prompt, so the policy")
        print("      can reach the shelf by copying that token. This measures 'is the")
        print("      instruction read', not 'is a rule applied' — no rule is varied, so")
        print("      there is no counterfactual to run. Inference must therefore supply")
        print("      target_shelf itself; it comes from rule_tables in the manifest.")
    if prompt_source == "rendered" and prompt_style == "category_only":
        print("    ⚠ category_only puts no destination in the prompt: the category→shelf")
        print("      association lives only in the weights. Inference needs nothing but")
        print("      the classifier's category, but the association cannot be changed")
        print("      without retraining, and a shelf reshuffle invalidates the checkpoint.")
    if rule_tables:
        # Same validators the Jetson MCP client aborts on, imported from the
        # shared package — a table this converter accepts is one the robot will
        # accept, and vice versa.
        problems = validate_all(rule_tables)
        print("    Rule tables (derived across episodes):")
        for ver in sorted(rule_tables):
            rows = ", ".join(f"{c}->{t}" for c, t in sorted(rule_tables[ver].items()))
            flags = problems.get(ver, [])
            print(f"      {ver:6s} {rows}" + (f"   ⚠ {'; '.join(flags)}" if flags else ""))
        for msg in problems.get("*", []):
            print(f"    ⚠ {msg}")
            print("      Under rule_table the answer is guessable from which destination")
            print("      tokens appear. Make every version a permutation of the same set.")
        if prompt_style == "rule_table" and any(
            INCOMPLETE in p for ps in problems.values() for p in ps
        ):
            print("    🔴 rule_table SILENTLY DEGRADED to single_rule for those versions.")
            print("       A one-row table puts the answer right of the only arrow, so the")
            print("       policy never has to match its category against anything. Pass the")
            print("       declared table with --rule-table-file, or collect all three")
            print("       categories under each rule_version. This run measures nothing")
            print("       that --prompt-style single_rule would not have measured.")
    for p, c in sorted(prompt_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {c:3d} ep  {p!r}")
    if guard.rule_conflicts:
        print("  ⚠ rule is not a function — same version maps one category to two colours:")
        for c in guard.rule_conflicts:
            print(f"      {c}")
    if guard.unrendered_eps:
        print(f"  ⚠ could not render from enums in {guard.unrendered_eps} — used the collector "
              f"string. Those episodes are NOT under style={prompt_style}; exclude them from "
              f"any template comparison.")
    if guard.missing_prompt_eps:
        print(f"  ⚠ no prompt at all in {guard.missing_prompt_eps} — fell back to "
              f"phase strings, so their rule is NOT in the dataset.")

    if context_counts:
        print("\n  Episode context (evaluation grouping, not a model input):")
        for k in CONTEXT_KEYS:
            vals = context_counts.get(k) or {}
            if not vals or set(vals) == {""}:
                print(f"    {k:14s} ⚠ absent")
                continue
            print(f"    {k:14s} {dict(sorted(vals.items()))}")
        _report_category_destination_matrix(cat_dest_counts)

    if schema_counts:
        print(f"  Phase schema   : {dict(sorted(schema_counts.items()))}")

    empty = []
    for sch in sorted(schema_counts):
        names = SCHEMAS[sch][1]
        tot = sum(c for (s, _), c in phase_counts.items() if s == sch)
        print(f"\n  Phase distribution [{sch}]  ({tot} frames):")
        for i, name in enumerate(names):
            count = phase_counts.get((sch, i), 0)
            flag = ""
            if count == 0:
                flag = "   ⚠ EMPTY — this task string never appears"
                empty.append(f"{sch}/{name}")
            print(f"    {i} {name:10s}: {count:6d} ({100.0 * count / max(1, tot):5.1f}%){flag}")
    if empty:
        print(f"\n  ⚠ phases with zero frames: {empty}")
        print("     A phase collapsed — the operator moved continuously through it so the")
        print("     boundary detector found no segment. For 'planar' tune TRANSPORT_PERCENTILE;")
        print("     for 'insertion' tune INSERT_ENTER_MM / ALIGN_ENTER_MM against the axis")
        print("     travel reported below, or record explicit events (episode_events.csv).")
    if fallback_eps:
        print(f"  ⚠  phase fallback (proportional split) fired in {fallback_eps} episode(s)")
        print("     → planar: E6-tuned degree thresholds may not fit xArm6 Δq scale")
        print("     → insertion: pre-release displacement <1mm, insertion axis undefined")

    if insert_axes:
        ax = np.stack(insert_axes)
        mean_ax = ax.mean(axis=0)
        mean_ax = mean_ax / max(float(np.linalg.norm(mean_ax)), 1e-9)
        spread = float(np.mean([np.linalg.norm(a - mean_ax) for a in ax]))
        print(f"\n  Insertion axis : mean [{mean_ax[0]:+.2f} {mean_ax[1]:+.2f} {mean_ax[2]:+.2f}] "
              f"(x,y,z)  spread {spread:.3f}  n={len(insert_axes)}")
        if abs(mean_ax[2]) > 0.8:
            print("     ⚠ axis is near-vertical — this looks like a top-down place, not a")
            print("       horizontal shelf insertion. Check the schema selection.")
        if spread > 0.5:
            print("     ⚠ axis varies a lot across episodes — shelves at different angles,")
            print("       or the pre-release window is catching approach motion.")

    if guard.dropped_teleop_runs:
        print(f"\n  ⚠ Dropped teleop runs ({len(guard.dropped_teleop_runs)}) — real human motion "
              "outside the selected run:")
        for line in guard.dropped_teleop_runs[:10]:
            print(f"     {line}")
        if len(guard.dropped_teleop_runs) > 10:
            print(f"     ... +{len(guard.dropped_teleop_runs) - 10} more")
        print("     A '[after release]' run is usually the manual retract. To keep it, the")
        print("     collector must hold mode 5 continuously from grasp through retract.")

    print("\n  Integrity guards:")
    if guard.missing_cols:
        print(f"    ⚠  columns absent, guard skipped: {sorted(guard.missing_cols)}")
    print(f"    dt outliers        : {guard.dt_outliers}"
          f"{f'  (dt_from_prev read as {guard.dt_unit})' if guard.dt_unit else ''}")
    print(f"    frame_dropped flags: {guard.dropped_flags}")

    if guard.zero_dq_frac:
        worst = max(guard.zero_dq_frac.items(), key=lambda kv: kv[1])
        print(f"    repeated joint vectors: worst ep{worst[0]} {100 * worst[1]:.1f}%  "
              f"(mean {100 * np.mean(list(guard.zero_dq_frac.values())):.1f}%)")
        bad = {k: v for k, v in guard.zero_dq_frac.items() if v > ZERO_DQ_WARN}
        if bad:
            print(f"    🔴 joint feedback below the record rate in {sorted(bad)}")
            print(f"       >{100 * ZERO_DQ_WARN:.0f}% of frames repeat the previous joint vector.")
            print("       At 16 Hz recording, a fraction f implies feedback near 16*(1-f) Hz —")
            print("       e.g. 69% means the driver was publishing at 5 Hz. Those repeats become")
            print("       zero actions and read as deliberate stillness. Fix joint_states_rate")
            print("       at the driver and re-collect; this is not repairable downstream.")

    if guard.stalls:
        print(f"    🔴 stalls: {len(guard.stalls)} run(s), {guard.stall_frames} frames "
              f"({100 * guard.stall_frames / max(total_frames, 1):.1f}% of kept frames)")
        for s in guard.stalls[:8]:
            print(f"       {s}")
        if len(guard.stalls) > 8:
            print(f"       … and {len(guard.stalls) - 8} more")
        print("       Commanded but not moving (velocity/joint limit or collision). The")
        print("       action there is ~0 while the scene looks mid-reach, so training on it")
        print("       teaches the policy to freeze exactly where it should push through.")
        print("       Fix the cause on the robot; excluding them only hides the gap.")

    if guard.joint_wraps:
        print(f"    🔴 revolution wraps: {len(guard.joint_wraps)}")
        for w in guard.joint_wraps[:8]:
            print(f"       {w}")
        print(f"       {guard.wrap_fixed_frames} action frame(s) rewritten to the shortest-path")
        print("       delta, so the dataset is usable. But the wrap means J4 reached the")
        print("       revolution boundary — the same boundary that locked the controller")
        print("       into mode 0 — so add the hardware guard at the source. Left raw, one")
        print("       such frame normalises to ~4900 sigma and takes the whole run with it:")
        print("       q01/q99 are percentiles, so a lone outlier never widens the divisor.")

    if guard.target_actual_mismatch:
        print(f"    🔴 prompt/landing mismatch, dropped: {len(guard.target_actual_mismatch)}")
        for m in guard.target_actual_mismatch[:8]:
            print(f"       {m}")
        print("       The prompt named one shelf and the arm released at another. Keeping")
        print("       these teaches the policy that the prompt is noise — the exact")
        print("       opposite of what this dataset is for. Check the label, or whether")
        print("       CATEGORY_TO_SHELF and the physical waypoints have drifted apart.")

    if guard.rig_layout_mismatch:
        print(f"    🔴 collector's side constant disagrees with the contract on "
              f"{len(guard.rig_layout_mismatch)} episode(s):")
        for m in guard.rig_layout_mismatch[:4]:
            print(f"       {m}")
        print("       Both sides are module constants, so the overwhelmingly likely")
        print("       cause is the two machines running different versions of the")
        print("       shared prompt package. Check that first; a moved shelf would")
        print("       normally show up in the geometry check below instead.")
    if guard.rig_geometry_mismatch:
        print(f"    🔴 taught shelf positions contradict the contract on "
              f"{len(guard.rig_geometry_mismatch)} episode(s):")
        for m in guard.rig_geometry_mismatch[:4]:
            print(f"       {m}")
        print("       This one does not descend from any side constant, so it is the")
        print("       check that can actually catch a physically rearranged rig — or")
        print("       a side constant that is wrong on BOTH machines at once. Resolve")
        print("       it against the waypoint file before converting anything.")
    if guard.rig_geometry_unusable:
        print(f"    ⚠  shelf geometry not decidable on {len(guard.rig_geometry_unusable)} episode(s):")
        for m in guard.rig_geometry_unusable[:3]:
            print(f"       {m}")

    if guard.no_landing_eps:
        print(f"    ⚠  no landing recorded (actual_shelf null): {guard.no_landing_eps}")
        print("       Distinct from a wrong-shelf landing — the arm never released, so")
        print("       there was no placement to judge. Look at the gripper trace, not the label.")

    if guard.release_distances:
        d = np.array(guard.release_distances)
        print(f"    release→shelf distance: median {np.median(d):.0f}mm  "
              f"p95 {np.percentile(d, 95):.0f}mm  max {d.max():.0f}mm")
        if np.percentile(d, 95) > 150:
            print("       ⚠ releases land far from any waypoint, so 'which shelf' is a weak")
            print("         judgement. Either the waypoints are stale or the insertions are")
            print("         not reaching the shelf.")

    if guard.layouts_seen:
        print(f"    shelf layouts seen: {len(guard.layouts_seen)}")
        for lay, n in sorted(guard.layouts_seen.items(), key=lambda kv: -kv[1]):
            print(f"       {n:3d} ep  {lay}")
        if len(guard.layouts_seen) == 1:
            print("       ⚠ one layout only — label and position are perfectly correlated,")
            print("         so nothing here can separate 'reads the sign' from 'went to the")
            print("         same place'. Vary the sign placement to keep that question open.")

    if guard.unconfirmed_category_eps:
        print(f"    ⚠  classifier label never confirmed by an operator: "
              f"{guard.unconfirmed_category_eps}")
        print("       category_source=external_vlm with no category_confirmed field. If the")
        print("       classifier was wrong, the prompt names one class and the demonstration")
        print("       shows another, which trains the policy to ignore the prompt.")
    if guard.ncmd_values:
        print(f"    n_commands_in_frame: {dict(sorted(guard.ncmd_values.items()))}")
        if set(guard.ncmd_values) != {2}:
            print("       ⚠  not uniformly 2 → twist publish is not an exact 2:1 multiple of "
                  "the record rate (32 Hz not applied, or mixed batches)")

    if grip_values:
        g = np.concatenate(grip_values)
        print("\n  Gripper signal:")
        print(f"    range {g.min():.3f} ~ {g.max():.3f} | unique levels {len(np.unique(np.round(g, 3)))}")
        print(f"    frac<0.5 {float((g < 0.5).mean()):.3f}   frac>=0.5 {float((g >= 0.5).mean()):.3f}")
    if gating_lags:
        lag = np.array(gating_lags)
        print(f"    gating lag (gated - raw, frames): mean {lag.mean():.2f}  "
              f"p95 {np.percentile(lag, 95):.1f}  max {lag.max()}")
        print("       → 0.4 s gate at 16 Hz = up to ~6 frames; large values shift phase boundaries")

    if clamp_mag:
        c = np.concatenate(clamp_mag)
        frac = float((c > 1e-6).mean())
        print("\n  Twist clamping (teleop_raw vs twist_sent):")
        print(f"    |raw-sent|max  mean {c.mean():.4f}  p95 {np.percentile(c, 95):.4f}  max {c.max():.4f}")
        print(f"    clamped frames {100 * frac:.1f}%")
        if frac > 0.10:
            print("       ⚠ >10% — teleop gain too high. The operator is fighting the ±1 limit,")
            print("         so the recorded trajectory is a clipped version of the intent.")

    if all_actions:
        a = np.concatenate(all_actions, axis=0)
        q01, q99 = np.percentile(a, 1, axis=0), np.percentile(a, 99, axis=0)
        rng = q99 - q01
        joint_rng = rng[:6]
        med = float(np.median(joint_rng))
        print("\n  Per-axis action quantile range (q99-q01) — normalisation divisor:")
        for i in range(7):
            name = f"dj{i + 1}" if i < 6 else "grip"
            flag = ""
            if i < 6 and med > 0 and joint_rng[i] < med / 5.0:
                flag = "   ⚠ DEGENERATE (<1/5 of median) — E6 j5 pathology"
            print(f"    {name:5s} q01 {q01[i]:8.4f}  q99 {q99[i]:8.4f}  range {rng[i]:8.4f}{flag}")
        print(f"    median joint range: {med:.4f}")
        amax = np.abs(a[:, :6]).max(axis=1)
        print(f"\n  |Δq| tail (singularity spikes): p99 {np.percentile(amax, 99):.4f}  "
              f"max {amax.max():.4f}  (deg/frame)")

    print(f"\n  Local root: {dataset.root}")
    print("\nNext steps:")
    print("  1. config.py 의 pi05_e7_v1_lora 에서 repo_id / asset_id 를 "
          f"'{repo_id}' 로 교체")
    print("  2. uv run scripts/compute_norm_stats.py --config-name pi05_e7_v1_lora")
    print("  3. uv run scripts/train.py pi05_e7_v1_lora --exp-name e7_2cam_lora_v1")
    print("  ⚠ 파일럿 데이터로 만든 norm_stats 는 스모크 전용 — 본 수집 후 재계산할 것")


if __name__ == "__main__":
    tyro.cli(main)
