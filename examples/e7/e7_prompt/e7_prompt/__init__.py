"""Shared prompt contract for E7 (xArm6).

Installed on BOTH the training server and the Jetson. Never copy these files --
a copy drifts, and the drift is invisible until the policy quietly degrades.

    from e7_prompt import ContextSpec, render_prompt, compute_contract_hash

Pure Python, no dependencies.
"""

from .contract import (
    CONTRACT_VERSION,
    DEFAULT_TOKENIZER_ID,
    TokenizerSpec,
    build_manifest,
    compute_contract_hash,
    contract_payload,
    verify_contract,
)
from .render import ContextSpec, render_from_meta, render_prompt
from .templates import (
    CANONICAL_CATEGORIES,
    CANONICAL_DESTINATIONS,
    CATEGORY_EN,
    PROMPT_STYLES,
    PROMPT_TEMPLATES,
    RIG_LAYOUT,
    STYLES_NEEDING_TARGET,
    canonical_category,
    side_for_waypoint,
)
from .validate import (
    INCOMPLETE,
    MULTISET_MISMATCH,
    NOT_A_FUNCTION,
    NOT_INJECTIVE,
    check_multiset_invariance,
    check_rule_table,
    validate_all,
)

__version__ = CONTRACT_VERSION

__all__ = [
    "CANONICAL_CATEGORIES",
    "CANONICAL_DESTINATIONS",
    "CATEGORY_EN",
    "CONTRACT_VERSION",
    "DEFAULT_TOKENIZER_ID",
    "INCOMPLETE",
    "MULTISET_MISMATCH",
    "NOT_A_FUNCTION",
    "NOT_INJECTIVE",
    "PROMPT_STYLES",
    "PROMPT_TEMPLATES",
    "RIG_LAYOUT",
    "STYLES_NEEDING_TARGET",
    "ContextSpec",
    "TokenizerSpec",
    "build_manifest",
    "canonical_category",
    "check_multiset_invariance",
    "check_rule_table",
    "compute_contract_hash",
    "contract_payload",
    "verify_contract",
    "render_from_meta",
    "render_prompt",
    "side_for_waypoint",
    "validate_all",
]
