"""Compare a recorded episode against what the converter expects, before converting it.

The converter already refuses bad episodes, but it refuses them one reason at a time
and in the middle of a long run. This reads a single episode and reports every
discrepancy at once, so a probe episode can be checked the moment it exists rather
than after a conversion has half-finished.

    uv run examples/e7/check_schema.py --episode ~/xarm_vla_episodes/0

It compares against the same constants the converter uses, plus the fixture in
make_v5_fixture.py, so the three cannot drift apart silently.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

import pandas as pd
import tyro

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from convert_e7_to_lerobot import (  # noqa: E402
    COL_DROPPED, COL_DT, COL_MODE, COL_NCMD, COL_TELEOP, CONTEXT_KEYS,
    GRIPPER_COL_GATED, GRIPPER_COL_RAW, IMAGE_COL_HIK, IMAGE_COL_ZED,
    JOINT_COLS, META_CONTRACT, MIN_SCHEMA_VERSION, TWIST_RAW_COLS, TWIST_SENT_COLS,
    _sides_from_waypoint_y, derived_only_target, legacy_only_target, resolved_target_side,
)
from make_v5_fixture import CSV_COLS as FIXTURE_COLS  # noqa: E402

from e7_prompt import CANONICAL_CATEGORIES, CANONICAL_DESTINATIONS, RIG_LAYOUT, canonical_category  # noqa: E402

REQUIRED_COLS = (*JOINT_COLS, GRIPPER_COL_GATED, IMAGE_COL_HIK, IMAGE_COL_ZED)
GATE_COLS = (COL_MODE, COL_TELEOP)
DIAGNOSTIC_COLS = (GRIPPER_COL_RAW, COL_DT, COL_DROPPED, COL_NCMD, *TWIST_RAW_COLS, *TWIST_SENT_COLS)

COL_MOTION_SRC = "motion_source"
COL_ACTIVE_SEQ = "active_sequence"

# Scripted motion splits in two, and the split is whether the observation can
# account for it.
#
# A waypoint route to the declared shelf is goal-directed and the goal is in the
# prompt: given "carry the book to the center shelf" and a grasped book, the
# motion follows. Cloning a planner is ordinary imitation learning, and excluding
# it would leave the policy unable to travel at all -- something would have to
# drive the arm to the shelf at inference too.
#
# A home return or an error recovery has no such account. Nothing in the image or
# the prompt says why the arm left for the home pose, so a policy trained across
# one learns to go home at unpredictable moments.
LEARNABLE_SCRIPTED = ("waypoint_route",)
UNPREDICTABLE_SCRIPTED = ("home_return", "error_recovery", "anchor_auto", "script")


@dataclasses.dataclass
class Args:
    episode: pathlib.Path
    """One episode directory: robot_data.csv + episode_meta.json + images/."""
    fps: float = 16.0


class Report:
    def __init__(self) -> None:
        self.blocking: list[str] = []
        self.degraded: list[str] = []
        self.notes: list[str] = []

    def block(self, m: str) -> None:
        self.blocking.append(m)

    def degrade(self, m: str) -> None:
        self.degraded.append(m)

    def note(self, m: str) -> None:
        self.notes.append(m)


def check_csv(ep: pathlib.Path, fps: float, r: Report) -> pd.DataFrame | None:
    csv = ep / "robot_data.csv"
    if not csv.is_file():
        r.block(f"no robot_data.csv in {ep}")
        return None
    df = pd.read_csv(csv)
    cols = set(df.columns)

    if missing := [c for c in REQUIRED_COLS if c not in cols]:
        r.block(f"required columns absent: {missing}")
    if missing := [c for c in GATE_COLS if c not in cols]:
        r.block(f"demonstration gate needs {missing} — without them every frame reads "
                f"as operator-controlled and scripted moves get trained on")
    if missing := [c for c in DIAGNOSTIC_COLS if c not in cols]:
        r.degrade(f"{len(missing)} diagnostic column(s) absent, those guards skip: {missing[:6]}")

    if extra := sorted(cols - set(FIXTURE_COLS)):
        r.note(f"{len(extra)} column(s) the fixture does not model (harmless, but the "
               f"fixture is now out of date): {extra[:8]}")
    if absent := [c for c in FIXTURE_COLS if c not in cols]:
        r.note(f"{len(absent)} column(s) the fixture has that this episode lacks: {absent[:8]}")

    # The unit that cost a whole batch once. Report what the converter will infer.
    if COL_DT in cols:
        med = float(pd.to_numeric(df[COL_DT], errors="coerce").dropna().median())
        unit = "ms" if med > (1.0 / fps) * 100 else "s"
        r.note(f"dt_from_prev median {med:g} -> read as {unit} "
               f"(nominal {1.0 / fps:g} s at {fps:g} Hz)")

    # A staircase, not an event stream. A column of deltas looks the same at a glance.
    if GRIPPER_COL_GATED in cols:
        g = pd.to_numeric(df[GRIPPER_COL_GATED], errors="coerce").dropna()
        lo, hi = float(g.min()), float(g.max())
        if lo < -0.01 or hi > 1.01:
            r.block(f"{GRIPPER_COL_GATED} ranges {lo:.3f}..{hi:.3f}; the contract is an "
                    f"absolute aperture in [0,1]. A signed or unbounded column here means "
                    f"the value is a delta, and action[6] would become a delta of a delta")
        elif hi - lo < 1e-6:
            r.degrade(f"{GRIPPER_COL_GATED} never changes ({lo:.3f}) — no grasp in this episode")
        else:
            r.note(f"{GRIPPER_COL_GATED} {lo:.3f}..{hi:.3f}, {g.nunique()} distinct level(s)")
    return df


def check_scripted_motion(df: pd.DataFrame | None, r: Report) -> None:
    """Scripted motion between the grasp and the release breaks the end-to-end contract.

    This is the check the 08-04 batch needed and did not have. Every frame in it
    passed inspection -- rate, skew, exposure, gating all clean -- because the
    defect was not in any frame. It was in who moved the arm: the operator picked
    the book, a teach button drove it to the shelf, the operator placed it. Eleven
    of twelve episodes were unusable and nothing in the frame-level report said so.
    """
    if df is None or GRIPPER_COL_GATED not in df.columns:
        return
    if COL_MOTION_SRC not in df.columns:
        r.degrade(f"no {COL_MOTION_SRC} — cannot tell operator-driven frames from scripted ones")
        return

    g = pd.to_numeric(df[GRIPPER_COL_GATED], errors="coerce").fillna(0.0).to_numpy()
    close = next((i for i in range(1, len(g)) if g[i - 1] < 0.5 <= g[i]), None)
    release = next((i for i in range(len(g) - 1, 0, -1) if g[i] < g[i - 1] - 0.05), None)
    if close is None or release is None or release <= close:
        r.note("no complete grasp (close then release) — scripted-motion check skipped")
        return

    src = df[COL_MOTION_SRC].astype(str).str.strip().str.lower()
    carry = src.iloc[close:release + 1]
    hits = carry[carry.isin(UNPREDICTABLE_SCRIPTED)]
    routed = carry[carry.isin(LEARNABLE_SCRIPTED)]
    if hits.empty:
        if routed.empty:
            r.note(f"carry frames {close}..{release} are operator-driven throughout")
        else:
            r.note(f"{len(routed)} waypoint-route frame(s) ({100.0 * len(routed) / max(1, release - close + 1):.1f}% "
                   f"of the carry) — learnable: the destination they head for is named in the prompt")
        return

    counts = hits.value_counts().to_dict()
    pct = 100.0 * len(hits) / max(1, release - close + 1)
    r.block(f"{len(hits)} unpredictable scripted frame(s) ({pct:.1f}% of the carry) between "
            f"grasp@{close} and release@{release}: {counts}. Nothing in the observation or the "
            f"prompt accounts for these, so a policy trained across them learns to make the "
            f"same move at moments it cannot anticipate")
    if COL_ACTIVE_SEQ in df.columns:
        seqs = sorted({s for s in df[COL_ACTIVE_SEQ].astype(str).str.strip().iloc[close:release + 1]
                       if s and s.lower() not in ("nan", "none")})
        if seqs:
            r.note(f"active_sequence during the carry: {seqs} — note this cannot say WHICH "
                   f"shelf, only that a route ran; the destination comes from the declaration")


def check_meta(ep: pathlib.Path, r: Report) -> dict:
    f = ep / "episode_meta.json"
    if not f.is_file():
        r.block(f"no episode_meta.json in {ep}")
        return {}
    meta = json.loads(f.read_text(encoding="utf-8"))

    sv = meta.get("schema_version")
    if sv is None or int(sv) < MIN_SCHEMA_VERSION:
        r.block(f"schema_version={sv!r}, converter requires >= {MIN_SCHEMA_VERSION}")
    if bad := {k: meta.get(k) for k, v in META_CONTRACT.items() if meta.get(k) != v}:
        r.block(f"contract fields disagree: {bad} (expected {META_CONTRACT})")

    side = resolved_target_side(meta)
    if not side:
        r.block("no resolved target side — the operator pressed no teach button, so "
                "nothing declares where this episode was meant to go")
    elif side not in CANONICAL_DESTINATIONS:
        r.block(f"resolved target side {side!r} not in {CANONICAL_DESTINATIONS}")
    else:
        r.note(f"resolved target side: {side!r}")
    if (k := derived_only_target(meta)) is not None:
        r.block(f"the only destination field is {k!r}, which is derived from the shelf "
                f"labels and the category rather than declared by the operator. On a "
                f"single-category batch that derivation returns the same shelf for every "
                f"episode, so it is reported and not used")
    if (k := legacy_only_target(meta)) is not None:
        r.block(f"the only destination field is {k!r}; that names the DEFAULT arrangement, "
                f"not this episode's, and promoting it is the migration that must not happen")

    cat = canonical_category(str(meta.get("category") or ""))
    if not cat:
        r.block("no category")
    elif cat not in CANONICAL_CATEGORIES:
        r.block(f"category {cat!r} not in {CANONICAL_CATEGORIES} — unmapped values pass "
                f"through canonicalisation and reach the tokenizer verbatim")
    if meta.get("category_confirmed") is False:
        r.block("category_confirmed=false")
    if str(meta.get("category_source") or "").lower() == "mcp":
        if absent := [k for k in ("mcp_status", "category_confidence", "vlm_model_version")
                      if meta.get(k) in (None, "")]:
            r.block(f"category_source=mcp but {absent} absent")
    else:
        r.note(f"category_source={meta.get('category_source')!r} (classifier fields may be null)")

    # The rig, checked two ways: against the collector's constant, and against geometry.
    phys = meta.get("shelf_layout_physical")
    if isinstance(phys, dict) and phys:
        got = {str(k).strip(): str(v).strip().lower() for k, v in phys.items()}
        if got != dict(RIG_LAYOUT):
            r.block(f"shelf_layout_physical {got} != contract {dict(RIG_LAYOUT)} — most "
                    f"likely the two machines run different prompt-package versions")
    else:
        r.degrade("no shelf_layout_physical")

    ys = meta.get("shelf_waypoint_tcp_y_mm")
    if isinstance(ys, dict) and ys:
        derived, why = _sides_from_waypoint_y(ys)
        if why:
            r.degrade(f"shelf geometry not decidable: {why}")
        elif derived != dict(RIG_LAYOUT):
            r.block(f"taught positions say {derived}, contract says {dict(RIG_LAYOUT)} — this "
                    f"one is independent of any side constant, so it can catch a rig that "
                    f"was physically rearranged")
        else:
            r.note("taught shelf positions agree with the contract")
    else:
        r.degrade("no shelf_waypoint_tcp_y_mm — the only independent check on left/right "
                  "is unavailable, leaving two module constants comparing to each other")

    if absent := [k for k in CONTEXT_KEYS if k not in meta]:
        r.note(f"{len(absent)}/{len(CONTEXT_KEYS)} context key(s) absent (carried if present, "
               f"not required): {absent[:8]}")
    if "schema_probe" in str(meta.get("book_id") or ""):
        r.note("book_id marks a schema probe — conversion needs --allow-probe, and this "
               "episode must not reach a training set")
    return meta


def check_images(ep: pathlib.Path, df: pd.DataFrame | None, r: Report) -> None:
    if df is None or IMAGE_COL_HIK not in df.columns:
        return
    first = str(df[IMAGE_COL_HIK].iloc[0])
    if (ep / "images" / first).is_file():
        r.note(f"image_path_hik is relative to images/ ({first!r}) — as the converter expects")
    elif (ep / first).is_file():
        r.block(f"image_path_hik {first!r} resolves from the episode root, but the converter "
                f"joins it under images/. Either drop the images/ prefix from the column or "
                f"pass --images-subdir ''")
    else:
        r.block(f"image_path_hik {first!r} resolves from neither {ep} nor {ep / 'images'}")


def main(args: Args) -> None:
    ep = args.episode
    r = Report()
    print(f"checking {ep}\n")
    df = check_csv(ep, args.fps, r)
    check_scripted_motion(df, r)
    check_meta(ep, r)
    check_images(ep, df, r)

    if df is not None:
        print(f"  {len(df)} frames, {len(df.columns)} columns\n")
    for label, items, mark in (("BLOCKING", r.blocking, "🔴"),
                               ("DEGRADED", r.degraded, "⚠ "),
                               ("NOTES", r.notes, "  ")):
        if not items:
            continue
        print(f"{label}")
        for m in items:
            print(f"  {mark} {m}")
        print()

    if r.blocking:
        print(f"{len(r.blocking)} blocking issue(s) — this episode will not convert.")
        raise SystemExit(1)
    print("No blocking issues. This episode will convert."
          + (f" {len(r.degraded)} guard(s) will be skipped." if r.degraded else ""))


if __name__ == "__main__":
    main(tyro.cli(Args))
