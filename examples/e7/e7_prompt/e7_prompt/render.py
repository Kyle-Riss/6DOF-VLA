"""The single prompt renderer shared by the converter and the Jetson client.

There is exactly one code path from enum fields to a prompt string. If training
and inference each formatted their own string, the two contracts could drift
apart silently -- the tokens would differ, the policy would degrade, and the
degradation would be misattributed to the policy. Hence one function, installed
as a pinned dependency on both sides, gated by ``compute_contract_hash``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from .templates import (
    PROMPT_STYLES,
    PROMPT_TEMPLATES,
    STYLES_NEEDING_TARGET,
    canonical_category,
    category_text,
)


@dataclasses.dataclass(frozen=True)
class ContextSpec:
    """Everything the renderer needs, and nothing it does not.

    The Jetson MCP client builds this from tool responses; the converter builds
    it from ``episode_meta.json``. Both then call :func:`render_prompt`.

    ``rule_table`` is the FULL mapping for this ``rule_version``, not a resolved
    destination. Passing only the answer would mean the rule was applied outside
    the policy, which is exactly what the experiment is trying to measure.
    """

    category: str
    target: str
    rule_version: str = ""
    rule_table: Mapping[str, str] = dataclasses.field(default_factory=dict)
    prompt_style: str = "single_rule"

    def __post_init__(self) -> None:
        if self.prompt_style not in PROMPT_STYLES:
            raise ValueError(
                f"unknown prompt_style {self.prompt_style!r}; expected one of {PROMPT_STYLES}"
            )


def render_prompt(spec: ContextSpec) -> str:
    """Render the prompt string. Deterministic, pure, no I/O.

    The category reaches the text as its prompt spelling ("liberal arts") while
    tables stay keyed on the canonical enum ("liberal_arts"); see
    :func:`~e7_prompt.templates.category_text`.
    """
    cat = canonical_category(spec.category)
    tgt = spec.target.strip()
    if not cat:
        raise ValueError("ContextSpec.category is empty")
    cat_txt = category_text(cat)

    if spec.prompt_style == "category_only":
        # `target` stays optional here: at inference the classifier returns a
        # category and nothing else, which is the whole point of this style.
        return PROMPT_TEMPLATES["category_only"].format(cat=cat_txt)

    if not tgt:
        raise ValueError(
            f"ContextSpec.target is empty but prompt_style={spec.prompt_style!r} "
            "names the destination; use prompt_style='category_only' when the "
            "shelf is not known at inference time"
        )

    if spec.prompt_style == "rule_table":
        table = {canonical_category(k): v.strip() for k, v in spec.rule_table.items()}
        # A table missing the episode's own category cannot be rendered
        # faithfully; fill it in rather than emitting a prompt whose answer is
        # absent. The caller's validator should have caught this already.
        table.setdefault(cat, tgt)
        # Rows go in a FIXED canonical order -- sorted by category name, never
        # with the episode's own category first. Otherwise "read row 1" is a
        # shortcut that defeats the whole manipulation.
        rows = ", ".join(f"{category_text(c)}->{table[c]}" for c in sorted(table))
        return PROMPT_TEMPLATES["rule_table"].format(cat=cat_txt, tgt=tgt, table=rows)

    return PROMPT_TEMPLATES[spec.prompt_style].format(cat=cat_txt, tgt=tgt)


def render_from_meta(
    meta: Mapping,
    style: str,
    rule_tables: Mapping[str, Mapping[str, str]],
) -> str | None:
    """Adapter for ``episode_meta.json`` dicts. ``None`` if fields are missing.

    The destination comes from ``resolved_target_side`` and nowhere else. That
    field holds the operator's teach-button declaration; ``target_shelf`` holds a
    value the collector derives from the shelf labels and the book's category,
    and on a single-category batch that derivation returns the same shelf for
    every episode. Rendering from it produced twelve identical prompts for the
    08-04 batch while the arm had visited three different shelves -- the prompt
    then contradicts the demonstration, which teaches that the prompt is noise.

    This is the same fallback that had to be removed from the converter, and it
    matters more here: the prompt is the one string shared between training and
    inference, so a destination taken from the wrong field is wrong in both.
    """
    cat = canonical_category(str(meta.get("category") or meta.get("object_category") or ""))
    tgt = str(meta.get("resolved_target_side") or "").strip().lower()
    if not cat:
        return None
    if not tgt and style in STYLES_NEEDING_TARGET:
        return None
    ver = str(meta.get("rule_version") or "").strip()
    return render_prompt(
        ContextSpec(
            category=cat,
            target=tgt,
            rule_version=ver,
            rule_table=dict(rule_tables.get(ver, {})),
            prompt_style=style,
        )
    )
