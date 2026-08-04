"""Prompt templates and category canonicalisation.

Editing anything in this module changes ``compute_contract_hash`` and therefore
invalidates every checkpoint trained before the edit. That is deliberate: the
prompt string IS the contract between training and inference, and a silent
template change is the failure mode this package exists to prevent.
"""

from __future__ import annotations

# The three conditions differ ONLY in the rule clause, so the clause is the
# manipulated variable and everything else is held constant.
#
# ``single_rule`` states the answer to the right of the arrow, so a policy can
# reach the destination by copying that token without ever applying the rule --
# it is informationally equivalent to ``resolved`` and only tests whether the
# instruction is read at all. ``rule_table`` presents every rule_version with
# the SAME multiset of destination tokens, so token presence carries no signal
# and the category must actually be matched against a row.
#
# The two are NOT a difficulty ladder. A failure on ``single_rule`` does not
# imply ``rule_table`` will fail: the full table makes the mapping structure
# explicit and may well be easier. Treat them as independent ablations.
PROMPT_TEMPLATES: dict[str, str] = {
    # No destination anywhere. The category->shelf association lives only in the
    # weights, learned from demonstrations. Inference needs nothing but the
    # category the classifier returned, which is exactly what the MacBook tool
    # provides -- no lookup table has to be kept in sync on the robot.
    "category_only": "place the {cat} book in the appropriate shelf",
    # Destination and nothing else. For the split where scene understanding runs
    # off-device: the classifier reads the book, reads the shelf signs, resolves
    # which shelf this book belongs in, and the policy is told only where to put
    # it. The book is already grasped by then, so its category cannot inform the
    # motion and carrying it in the prompt only creates a shortcut -- within a
    # batch the category and the shelf move together, so a policy can answer from
    # the category word and never read the destination. Leaving it out makes the
    # destination word the only thing that can be read, and the only thing that
    # can be varied to test whether it was.
    "destination_only": "insert the book into the {tgt} shelf",
    # Names the destination and the category. Useful as an ablation against
    # destination_only -- if the two score the same, the category word is being
    # ignored, which is what it should be once the book is in the gripper.
    "resolved":    "category={cat}. insert the {cat} book into the {tgt} shelf",
    "single_rule": "category={cat}. rule: {cat}->{tgt}. "
                   "insert the {cat} book into the correct shelf",
    "rule_table":  "category={cat}. rules: {table}. "
                   "insert the {cat} book into the correct shelf",
}

PROMPT_STYLES: tuple[str, ...] = ("category_only", "destination_only", "resolved",
                                 "single_rule", "rule_table")

# Styles whose prompt text contains the destination. Everything else has to
# recover it from the weights, which means inference needs no shelf table.
STYLES_NEEDING_TARGET: frozenset[str] = frozenset(
    {"destination_only", "resolved", "single_rule", "rule_table"})

# The three book categories, in the canonical English form that reaches the
# tokenizer. The frozen Gemma 2B saw overwhelmingly English text and E6 used
# English prompts throughout.
#
# ⚠ Changed 2026-07-30 with the MacBook-classifier design. The previous taxonomy
# was (engineering, humanities, other) for 공학책/교양책/이외의 책. Under the new
# one 교양 means `liberal_arts`, where it used to map to `humanities` -- the same
# Korean word now denotes a DIFFERENT class. Episodes recorded under the old
# taxonomy cannot be reinterpreted, only relabelled.
CANONICAL_CATEGORIES: tuple[str, ...] = ("science", "liberal_arts", "humanities")

# The three destinations. Position rather than colour: `white` has no hue and
# collides with the xArm's white arm and clipped pixels in feature space.
CANONICAL_DESTINATIONS: tuple[str, ...] = ("left", "center", "right")

# Which physical side each shelf waypoint is on. This is a geometric fact about
# the rig, not a variable: the category->shelf assignment gets shuffled between
# episodes, but waypoint 2 is where it is.
#
# It belongs in the contract hash precisely because getting it wrong is silent.
# Flip left and right here and everything still runs — episodes convert, prompts
# render, loss falls — while every label points at the opposite shelf. Nothing
# downstream can detect that; only a mismatch against the checkpoint can.
RIG_LAYOUT: dict[str, str] = {"2": "right", "3": "center", "4": "left"}


def side_for_waypoint(waypoint: str | int) -> str | None:
    """Physical side of a shelf waypoint, or ``None`` if it is not a shelf."""
    return RIG_LAYOUT.get(str(waypoint).strip())

# episode_meta may carry Korean labels. Unmapped values pass through unchanged
# so an unexpected label shows up in the report instead of being swallowed.
CATEGORY_EN: dict[str, str] = {
    "과학": "science",
    "과학책": "science",
    "교양": "liberal_arts",
    "교양책": "liberal_arts",
    "인문학": "humanities",
    "인문학책": "humanities",
    "인문책": "humanities",
}


def canonical_category(raw: str) -> str:
    """Map a collector label to its canonical English form."""
    return CATEGORY_EN.get(raw.strip(), raw.strip())


# ---------------------------------------------------------------------------
# Per-phase prompts
# ---------------------------------------------------------------------------
#
# One string per stage of the episode instead of one per episode. E6 is the
# evidence: v2 conditioned on a single episode-level string and failed on the
# robot; v16 onward published a per-frame phase prompt and worked, and v23 -- the
# run adopted as the E7 starting point -- was trained that way.
#
# The reason is structural. pi0.5 sees ONE frame, with no history. Approaching a
# shelf and withdrawing from it put the arm in nearly the same place, and the
# gripper only separates them once it has already opened. A phase prompt supplies
# the stage the observation cannot.
#
# These live here rather than in the converter because the robot needs the same
# strings at inference. A phase prompt that differs by one character between
# training and serving conditions the policy on a sentence it never saw, and the
# tokenizer will not complain. Being in this module also puts them in the
# contract hash, so such a drift is caught at load time instead of in behaviour.
#
# {obj} comes from episode_meta["prompt_object_name"], {tgt} from the operator's
# declared destination -- never from the label-derived field, for the reason
# given in `render.py`.
PHASE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "planar": (
        "approach the {obj}",
        "pick up the {obj}",
        "lift the {obj}",
        "move the {obj} to the target area",
        "place the {obj} in the target area",
        "release the {obj}",
    ),
    "insertion": (
        "approach the {obj}",
        "grasp the {obj}",
        "lift the {obj}",
        # No align stage. It was meant for lining the book up with the opening
        # before pushing, but the shelf button puts the arm in front of the slot
        # already aligned, so there was nothing between arriving and pushing: the
        # phase came out empty on the first real insertion episode.
        "carry the {obj} to the {tgt} shelf",
        "insert the {obj} into the {tgt} shelf",
        "release the {obj}",
        "retract from the {tgt} shelf",
    ),
}

PHASE_NAMES: dict[str, tuple[str, ...]] = {
    "planar": ("approach", "pick", "lift", "transport", "place", "release"),
    "insertion": ("approach", "grasp", "lift", "carry", "insert",
                  "release", "retract"),
}


def render_phase_prompts(schema: str, obj: str, tgt: str) -> list[str]:
    """Every phase string for one episode, in phase order.

    Raises on an unknown schema rather than defaulting: a silently wrong schema
    produces prompts that read correctly and label the wrong frames.
    """
    if schema not in PHASE_TEMPLATES:
        raise KeyError(f"unknown phase schema {schema!r}; have {sorted(PHASE_TEMPLATES)}")
    return [t.format(obj=obj, tgt=tgt) for t in PHASE_TEMPLATES[schema]]
