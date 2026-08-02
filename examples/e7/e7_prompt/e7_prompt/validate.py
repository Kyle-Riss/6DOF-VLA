"""Rule-table validation, shared by the converter guard and the MCP client abort.

Every check here answers one question: *could a policy reach the right shelf
without reading the rule?* If yes, the experiment measures nothing, so the run
must stop rather than produce a number nobody can interpret.
"""

from __future__ import annotations

import collections
from collections.abc import Mapping

from .templates import CANONICAL_CATEGORIES, canonical_category

# Violation codes. The Jetson client aborts on any of these; the converter
# reports all of them at once so one pass surfaces every problem.
INCOMPLETE = "INCOMPLETE_RULE_TABLE"
NOT_INJECTIVE = "NOT_INJECTIVE"
MULTISET_MISMATCH = "MULTISET_MISMATCH"
NOT_A_FUNCTION = "NOT_A_FUNCTION"


def check_rule_table(
    table: Mapping[str, str],
    categories: tuple[str, ...] = CANONICAL_CATEGORIES,
) -> list[str]:
    """Check one ``rule_version``'s table. Empty list means it passed."""
    norm = {canonical_category(k): v.strip() for k, v in table.items()}
    problems: list[str] = []

    missing = set(categories) - set(norm)
    if missing:
        problems.append(f"{INCOMPLETE}: missing {sorted(missing)}")

    # A many-to-one rule is still a function, but two categories sharing a
    # destination means the destination no longer identifies the category, and
    # a counterfactual that swaps between them changes nothing.
    if len(set(norm.values())) < len(norm):
        dupes = [d for d, n in collections.Counter(norm.values()).items() if n > 1]
        problems.append(f"{NOT_INJECTIVE}: destination(s) {sorted(dupes)} used twice")

    return problems


def check_multiset_invariance(
    rule_tables: Mapping[str, Mapping[str, str]],
) -> list[str]:
    """Every rule_version must use the SAME destination multiset.

    Under ``rule_table`` the whole mapping is in the prompt, so if one version
    offers {left, center, right} and another {left, left, center}, mere token
    presence leaks which version is active. The permutation has to be the only
    thing that varies.
    """
    signatures = {
        ver: tuple(sorted(v.strip() for v in tbl.values()))
        for ver, tbl in rule_tables.items()
    }
    distinct = set(signatures.values())
    if len(distinct) <= 1:
        return []
    detail = ", ".join(f"{ver}={list(sig)}" for ver, sig in sorted(signatures.items()))
    return [f"{MULTISET_MISMATCH}: destination multisets differ across versions -- {detail}"]


def validate_all(
    rule_tables: Mapping[str, Mapping[str, str]],
    categories: tuple[str, ...] = CANONICAL_CATEGORIES,
) -> dict[str, list[str]]:
    """Validate every version. Returns ``{rule_version: [problems]}``.

    Cross-version problems are filed under the key ``"*"``.
    """
    out: dict[str, list[str]] = {}
    for ver, tbl in sorted(rule_tables.items()):
        problems = check_rule_table(tbl, categories)
        if problems:
            out[ver] = problems
    cross = check_multiset_invariance(rule_tables)
    if cross:
        out["*"] = cross
    return out
