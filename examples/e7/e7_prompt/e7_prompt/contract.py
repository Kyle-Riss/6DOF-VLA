"""The prompt contract hash -- the gate between a checkpoint and an inference run.

The hash covers everything that can change the token sequence the policy sees.
If any of it differs between the run that produced a checkpoint and the run that
serves it, the tokens differ and the policy is being asked a question it was
never trained on. That failure is silent: the robot moves, it just moves wrong,
and the regression gets blamed on the policy.

``discrete_state_input`` is in the hash for a concrete reason. It is a plain
config flag with a ``False`` default, and flipping it makes the tokenizer wrap
the text as ``"Task: ... , State: <7 discretised values>;\\nAction: "`` -- about
35 extra tokens, with no error and no visible symptom. Exactly the kind of drift
this hash exists to catch.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping

from .templates import (
    CANONICAL_CATEGORIES,
    CANONICAL_DESTINATIONS,
    CATEGORY_EN,
    RIG_LAYOUT,
    PROMPT_TEMPLATES,
)

CONTRACT_VERSION = "1.1.0"

# PaliGemma tokenizer as pinned by openpi. Bump only if the tokenizer artefact
# itself changes -- the value is opaque, it just has to differ when the
# tokenization does.
DEFAULT_TOKENIZER_ID = "gs://big_vision/paligemma_tokenizer.model"


@dataclasses.dataclass(frozen=True)
class TokenizerSpec:
    """The tokenizer-side facts that change the token sequence."""

    max_token_len: int
    discrete_state_input: bool
    tokenizer_id: str = DEFAULT_TOKENIZER_ID


def _canonical_tables(
    rule_tables: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    return {
        ver: {k: v for k, v in sorted(tbl.items())}
        for ver, tbl in sorted(rule_tables.items())
    }


def contract_payload(
    prompt_style: str,
    rule_tables: Mapping[str, Mapping[str, str]],
    tokenizer: TokenizerSpec,
) -> dict:
    """The exact object that gets hashed. Kept public so a mismatch is debuggable.

    The category->shelf MAPPING is deliberately not hashed. Shelf signs get
    shuffled between episodes, so each episode carries its own ``target_shelf``
    and a pinned table would either churn the hash per episode or describe a
    layout the data does not have. What must stay fixed is the VOCABULARY: which
    category tokens and which destination tokens may appear at all. That is what
    determines tokenization, and an unexpected token there is a real defect.

    ``rule_tables`` is still accepted so callers need not change, but it only
    contributes its key set — the versions present, not what they map to.
    """
    return {
        "contract_version": CONTRACT_VERSION,
        "prompt_style": prompt_style,
        "templates": dict(sorted(PROMPT_TEMPLATES.items())),
        "category_map": dict(sorted(CATEGORY_EN.items())),
        "categories": sorted(CANONICAL_CATEGORIES),
        "destinations": sorted(CANONICAL_DESTINATIONS),
        "rig_layout": dict(sorted(RIG_LAYOUT.items())),
        "rule_versions": sorted(rule_tables),
        "tokenizer": {
            "id": tokenizer.tokenizer_id,
            "max_token_len": tokenizer.max_token_len,
            "discrete_state_input": tokenizer.discrete_state_input,
        },
    }


def compute_contract_hash(
    prompt_style: str,
    rule_tables: Mapping[str, Mapping[str, str]],
    tokenizer: TokenizerSpec,
) -> str:
    """Stable 16-hex-char digest of the full prompt contract."""
    blob = json.dumps(
        contract_payload(prompt_style, rule_tables, tokenizer),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def verify_contract(
    manifest: Mapping,
    prompt_style: str,
    rule_tables: Mapping[str, Mapping[str, str]],
    tokenizer: TokenizerSpec,
) -> list[str]:
    """Compare runtime context against a checkpoint's manifest.

    Empty list means the run may proceed. Anything else and the caller must
    refuse to serve -- a prompt the policy was not trained on produces motion,
    not an error, so there is no safe way to continue and find out later.

    Returns a field-by-field diff rather than a bare hash mismatch: knowing
    *which* field moved is the difference between a five-minute fix and a day
    of bisecting checkpoints.
    """
    problems: list[str] = []

    expected = manifest.get("prompt_contract_hash")
    if not expected:
        return ["MISSING_CONTRACT_HASH: manifest has no prompt_contract_hash"]

    actual = compute_contract_hash(prompt_style, rule_tables, tokenizer)
    if actual == expected:
        return []

    if (m := manifest.get("prompt_style")) != prompt_style:
        problems.append(f"prompt_style: checkpoint={m!r} runtime={prompt_style!r}")

    for key, live in (
        ("categories", sorted(CANONICAL_CATEGORIES)),
        ("destinations", sorted(CANONICAL_DESTINATIONS)),
    ):
        if key in manifest and sorted(manifest[key]) != live:
            problems.append(f"{key}: checkpoint={sorted(manifest[key])} runtime={live}")

    if "rig_layout" in manifest and dict(manifest["rig_layout"]) != dict(RIG_LAYOUT):
        problems.append(
            f"rig_layout: checkpoint={manifest['rig_layout']} runtime={dict(RIG_LAYOUT)} "
            "— the shelves themselves moved, or left/right got flipped somewhere"
        )

    m_tok = manifest.get("tokenizer") or {}
    for key, runtime in (
        ("id", tokenizer.tokenizer_id),
        ("max_token_len", tokenizer.max_token_len),
        ("discrete_state_input", tokenizer.discrete_state_input),
    ):
        if key in m_tok and m_tok[key] != runtime:
            problems.append(f"tokenizer.{key}: checkpoint={m_tok[key]!r} runtime={runtime!r}")

    if (m := manifest.get("contract_version")) != CONTRACT_VERSION:
        problems.append(f"contract_version: checkpoint={m!r} runtime={CONTRACT_VERSION!r}")

    if not problems:
        # Hash differs but every compared field matches: the templates or the
        # category map were edited without a version bump.
        problems.append(
            f"hash {expected} != {actual} with all compared fields equal — "
            "templates.py or CATEGORY_EN was edited since this checkpoint"
        )
    return problems


def build_manifest(
    prompt_style: str,
    rule_tables: Mapping[str, Mapping[str, str]],
    tokenizer: TokenizerSpec,
    *,
    image_keys: tuple[str, ...],
    action_dim: int,
    action_horizon: int,
) -> dict:
    """The block to ship alongside a checkpoint.

    Jetson reads this at startup and refuses to serve if its own computed hash
    differs. Sequence length is recorded rather than derived so a slot-count
    change cannot pass unnoticed.
    """
    image_tokens = 256 * len(image_keys)
    return {
        "prompt_contract_hash": compute_contract_hash(prompt_style, rule_tables, tokenizer),
        "contract_version": CONTRACT_VERSION,
        "prompt_style": prompt_style,
        "categories": sorted(CANONICAL_CATEGORIES),
        "destinations": sorted(CANONICAL_DESTINATIONS),
        "rig_layout": dict(sorted(RIG_LAYOUT.items())),
        # Reference layout only — NOT hashed. Signs get shuffled, so each episode
        # carries its own target_shelf and inference must read the live layout.
        "reference_layout": _canonical_tables(rule_tables),
        "tokenizer": {
            "id": tokenizer.tokenizer_id,
            "max_token_len": tokenizer.max_token_len,
            "discrete_state_input": tokenizer.discrete_state_input,
        },
        "image_keys": list(image_keys),
        "image_tokens": image_tokens,
        "action_dim": action_dim,
        "action_horizon": action_horizon,
        "total_sequence_length": image_tokens + tokenizer.max_token_len + action_horizon,
    }
