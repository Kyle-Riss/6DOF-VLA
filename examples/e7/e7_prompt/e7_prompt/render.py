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
    """Render the prompt string. Deterministic, pure, no I/O."""
    cat = canonical_category(spec.category)
    tgt = spec.target.strip()
    if not cat:
        raise ValueError("ContextSpec.category is empty")

    if spec.prompt_style == "category_only":
        # `target` stays optional here: at inference the classifier returns a
        # category and nothing else, which is the whole point of this style.
        return PROMPT_TEMPLATES["category_only"].format(cat=cat)

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
        rows = ", ".join(f"{c}->{table[c]}" for c in sorted(table))
        return PROMPT_TEMPLATES["rule_table"].format(cat=cat, tgt=tgt, table=rows)

    return PROMPT_TEMPLATES[spec.prompt_style].format(cat=cat, tgt=tgt)


def render_from_meta(
    meta: Mapping,
    style: str,
    rule_tables: Mapping[str, Mapping[str, str]],
) -> str | None:
    """Adapter for ``episode_meta.json`` dicts. ``None`` if fields are missing.

    Accepts ``category`` or ``object_category``, and ``target_shelf`` or the
    legacy ``shelf_color``, so a collector schema change does not silently fall
    back to the baked-in prompt string.
    """
    cat = canonical_category(str(meta.get("category") or meta.get("object_category") or ""))
    tgt = str(meta.get("target_shelf") or meta.get("shelf_color") or "").strip()
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
