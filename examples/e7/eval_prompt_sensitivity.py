"""Does the policy actually read the prompt? Measured offline, no robot needed.

Dropping the rule permutation removed the counterfactual, but the question it
answered survives in a weaker form. Hold the observation fixed, vary only the
category in the prompt, and see whether the predicted chunk changes:

    same frame + "category=science ... into the left shelf"    -> chunk A
    same frame + "category=humanities ... into the right shelf" -> chunk B

If A and B are indistinguishable the policy is ignoring the text and steering
from the image alone, which is the failure mode no training loss reveals.

Flow matching is stochastic, so a raw distance between A and B means nothing on
its own. The comparison is always against the policy's OWN sampling noise: draw
several chunks per prompt and ask whether the between-prompt spread exceeds the
within-prompt spread. A separation ratio near 1 means the prompt did nothing.

    uv run examples/e7/eval_prompt_sensitivity.py \
        --config-name pi05_e7_v1_lora --checkpoint-dir checkpoints/.../19999
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path

import numpy as np
import tyro
from e7_prompt import ContextSpec, render_prompt

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"


@dataclasses.dataclass
class Args:
    config_name: str
    checkpoint_dir: Path
    rule_table_file: Path = Path("examples/e7/rule_tables.json")
    rule_version: str = "fixed"
    prompt_style: str = "resolved"
    n_frames: int = 8
    """Observations to test. Each is scored under every category."""
    n_samples: int = 4
    """Chunks drawn per (frame, prompt) to estimate sampling noise."""
    seed: int = 0


def _load_frames(repo_id: str, n: int, rng: np.random.Generator) -> list[dict]:
    """Pull raw observations straight from the converted dataset."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    ds = LeRobotDataset(repo_id)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    out = []
    for i in idx:
        s = ds[int(i)]
        out.append({
            "observation/exterior_image_1_left": np.asarray(s["exterior_image_1_left"]),
            "observation/exterior_image_2_left": np.asarray(s["exterior_image_2_left"]),
            "observation/state": np.asarray(s["state"], dtype=np.float32),
        })
    return out


def main(args: Args) -> None:
    rng = np.random.default_rng(args.seed)
    train_cfg = _config.get_config(args.config_name)
    tables = json.loads(args.rule_table_file.read_text(encoding="utf-8"))
    table = tables[args.rule_version]

    prompts = {
        cat: render_prompt(ContextSpec(
            category=cat, target=tgt, rule_version=args.rule_version,
            rule_table=table, prompt_style=args.prompt_style,
        ))
        for cat, tgt in sorted(table.items())
    }
    print("Prompts under test:")
    for c, p in prompts.items():
        print(f"  {c:14} {p}")

    policy = _policy_config.create_trained_policy(train_cfg, args.checkpoint_dir)
    frames = _load_frames(train_cfg.data.repo_id, args.n_frames, rng)
    print(f"\n{len(frames)} frame(s) x {len(prompts)} prompt(s) x {args.n_samples} sample(s)")

    # chunks[frame][category] -> (n_samples, horizon, 7)
    chunks: list[dict[str, np.ndarray]] = []
    for obs in frames:
        per_cat = {}
        for cat, prompt in prompts.items():
            draws = []
            for _ in range(args.n_samples):
                out = policy.infer({**obs, "prompt": prompt})
                draws.append(np.asarray(out["actions"], dtype=np.float64))
            per_cat[cat] = np.stack(draws)
        chunks.append(per_cat)

    # Compare on the CUMULATIVE joint displacement of a chunk: the deltas are
    # per-frame, and where the arm ends up is what distinguishes one shelf from
    # another. Gripper (index 6) is absolute and excluded.
    def endpoint(c: np.ndarray) -> np.ndarray:
        return c[..., :6].sum(axis=-2)

    within, between = [], []
    for per_cat in chunks:
        for cat, draws in per_cat.items():
            e = endpoint(draws)
            within += [np.linalg.norm(a - b) for a, b in itertools.combinations(e, 2)]
        for ca, cb in itertools.combinations(sorted(per_cat), 2):
            ea, eb = endpoint(per_cat[ca]).mean(0), endpoint(per_cat[cb]).mean(0)
            between.append(np.linalg.norm(ea - eb))

    w = float(np.mean(within)) if within else 0.0
    b = float(np.mean(between)) if between else 0.0
    ratio = b / w if w > 1e-9 else float("inf")

    print("\n  cumulative joint displacement over a chunk (degrees, j1..j6)")
    print(f"    within-prompt  (sampling noise) : {w:8.3f}")
    print(f"    between-prompt (prompt effect)  : {b:8.3f}")
    print(f"    separation ratio                : {ratio:8.2f}")

    print("\n  per-category mean endpoint:")
    for cat in sorted(prompts):
        m = np.mean([endpoint(c[cat]).mean(0) for c in chunks], axis=0)
        print(f"    {cat:14} " + "  ".join(f"{v:+7.2f}" for v in m))

    print()
    if ratio < 1.5:
        print("  ✗ The prompt is not steering the policy. Swapping the category barely")
        print("    moves the chunk further than resampling the same prompt does, so the")
        print("    trajectory is coming from the image alone. Training loss cannot show")
        print("    this — check that prompts actually vary in the dataset, and that the")
        print("    book's start position is not correlated with its category.")
    elif ratio < 3.0:
        print("  ~ Weak but present. The prompt shifts the chunk above noise; not enough")
        print("    to claim the destination is selected by the text. More data per")
        print("    category, or more start-position variety, before reading anything")
        print("    into the direction.")
    else:
        print("  ✓ The prompt dominates sampling noise: swapping the category sends the")
        print("    arm somewhere else. Confirm the DIRECTION matches the shelf layout")
        print("    in the per-category table above before calling it correct.")


if __name__ == "__main__":
    main(tyro.cli(Args))
