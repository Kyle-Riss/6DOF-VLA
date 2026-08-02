"""Assemble a deployable E7 bundle and refuse to ship an inconsistent one.

What the robot needs is produced by three unrelated steps and lands in three
unrelated places, in two different frameworks:

    checkpoints/<config>/<exp>/<step>/                    JAX, written by training
    checkpoints/<config>/<exp>/<step>_pytorch_lora_merged/  what actually deploys
    assets/<config>/<repo_id>/norm_stats.json             the de/normalisation
    <lerobot_home>/<repo_id>/meta/                        the prompt contract

Training writes JAX; the robot runs PyTorch. Copying the JAX ``params/`` is the
version of the classic deployment bug that at least fails loudly on the robot.
The quieter version is copying correct weights with a stale ``config.json``,
which leaves the robot to guess the slot count — and its guess is the three-slot
default, so the sequence length is wrong and nothing raises.

This script gathers the deployable pieces, recomputes the contract hash from the
config about to be deployed, and refuses rather than emitting a bundle whose
parts disagree. The result is exactly four files:

    model.safetensors  config.json  norm_stats.json  manifest.json

There is no ``params/`` and no ``tokenizer_config/`` in this format.

    uv run examples/e7/export_bundle.py \
        --config-name pi05_e7_v1_lora --exp-name e7_v1 --step 19999 \
        --out /path/to/e7_bundle_v1
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import tyro
from e7_prompt import TokenizerSpec, build_manifest, verify_contract

from openpi.training import config as _config

HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"


@dataclasses.dataclass
class Args:
    config_name: str
    exp_name: str
    step: int
    out: Path
    """Bundle directory to create. Refuses to overwrite unless --force."""
    repo_id: str | None = None
    """Dataset the checkpoint was trained on. Defaults to the config's own."""
    dataset_meta: Path | None = None
    """Where prompt_contract.json lives. Defaults to the LeRobot dataset meta dir."""
    force: bool = False
    no_params: bool = False
    """Skip the weights (~5 GB) — for verifying the contract wiring alone."""


def _fail(msg: str) -> None:
    raise SystemExit(f"REFUSING TO EXPORT — {msg}")


def main(args: Args) -> None:
    train_cfg = _config.get_config(args.config_name)
    model = train_cfg.model
    repo_id = args.repo_id or train_cfg.data.repo_id
    if not repo_id:
        _fail(f"config {args.config_name!r} has no repo_id")
    if not (HF_LEROBOT_HOME / repo_id).is_dir():
        _fail(
            f"no dataset at {HF_LEROBOT_HOME / repo_id} for repo_id={repo_id!r}. "
            f"Either {args.config_name!r} still points at a placeholder, or the "
            "dataset was never converted on this machine"
        )

    ckpt = Path("checkpoints") / args.config_name / args.exp_name / str(args.step)
    if not (ckpt / "params").is_dir():
        _fail(f"no JAX checkpoint at {ckpt}")

    # The robot runs PyTorch. Training writes JAX, and a separate conversion step
    # produces what actually deploys: model.safetensors + config.json + assets/.
    # Shipping the JAX params/ directory instead is silently useless — it loads
    # nowhere on the robot — so refuse and name the command that fixes it.
    torch_dir = ckpt.parent / f"{args.step}_pytorch_lora_merged"
    if not args.no_params:
        if not (torch_dir / "model.safetensors").is_file():
            _fail(
                f"no PyTorch bundle at {torch_dir}. The robot cannot load JAX params.\n"
                f"    uv run examples/convert_jax_to_pytorch_lora_merged.py \\\n"
                f"        --checkpoint-dir {ckpt} --config-name {args.config_name}"
            )
        # A config.json written before the fields below were added leaves the robot
        # to guess slot count and LoRA range, and its guess is the 3-slot default.
        tcfg_path = torch_dir / "config.json"
        if not tcfg_path.is_file():
            _fail(f"{torch_dir} has model.safetensors but no config.json")
        tcfg = json.loads(tcfg_path.read_text(encoding="utf-8"))
        missing = [k for k in ("config_name", "image_keys", "max_token_len", "pi05",
                               "vision_lora_layer_range") if k not in tcfg]
        if missing:
            _fail(
                f"{tcfg_path} predates the fields the robot needs: {missing}.\n"
                f"    Re-run the conversion; the current converter writes them."
            )
        if list(tcfg["image_keys"]) != list(model.image_keys):
            _fail(
                f"converted bundle has image_keys={tcfg['image_keys']} but the config "
                f"says {list(model.image_keys)} — the conversion is from a different run"
            )

    norm = Path(train_cfg.assets_base_dir or "assets") / args.config_name / repo_id / "norm_stats.json"
    if not norm.is_file():
        # assets_dir may be set on the data config instead.
        alt = Path("assets") / args.config_name / repo_id / "norm_stats.json"
        norm = alt if alt.is_file() else norm
    if not norm.is_file():
        _fail(f"no norm_stats at {norm} — run compute_norm_stats.py first")

    meta_dir = args.dataset_meta or (HF_LEROBOT_HOME / repo_id / "meta")
    contract_path = meta_dir / "prompt_contract.json"
    if not contract_path.is_file():
        _fail(
            f"no prompt_contract.json at {contract_path} — reconvert with a current "
            "converter so the contract is stamped alongside the dataset"
        )
    shipped = json.loads(contract_path.read_text(encoding="utf-8"))

    # The shipped contract was stamped at conversion time. Recompute it from the
    # config that is about to be deployed: if the model's tokenizer settings have
    # moved since, the tokens differ and this bundle would silently mis-serve.
    tok = TokenizerSpec(
        max_token_len=int(model.max_token_len),
        discrete_state_input=bool(model.discrete_state_input),
    )
    problems = verify_contract(
        shipped, shipped.get("prompt_style", ""), shipped.get("reference_layout") or {}, tok
    )
    if problems:
        _fail("contract disagrees with the live config:\n    " + "\n    ".join(problems))

    # verify_contract only recomputes the hash from (style, tables, live tokenizer).
    # The tokenizer block PRINTED in the contract is a separate copy, and Jetson
    # reads that copy — so a stale or edited block would be believed even though
    # the hash still matches. Compare it explicitly.
    shipped_tok = shipped.get("tokenizer") or {}
    for key, live in (
        ("max_token_len", tok.max_token_len),
        ("discrete_state_input", tok.discrete_state_input),
    ):
        if key in shipped_tok and shipped_tok[key] != live:
            _fail(
                f"contract's tokenizer.{key}={shipped_tok[key]!r} but the live config "
                f"says {live!r} — the stored block is what Jetson reads, so this would "
                "be believed despite a matching hash. Reconvert."
            )

    if list(shipped.get("image_keys") or []) != list(model.image_keys):
        _fail(
            f"image_keys moved since conversion: contract={shipped.get('image_keys')} "
            f"config={list(model.image_keys)} — the sequence length changed, reconvert"
        )

    out = args.out
    if out.exists():
        if not args.force:
            _fail(f"{out} exists (pass --force to replace)")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    manifest = build_manifest(
        shipped["prompt_style"],
        shipped.get("reference_layout") or {},
        tok,
        image_keys=tuple(model.image_keys),
        action_dim=int(model.action_dim),
        action_horizon=int(model.action_horizon),
    )
    manifest |= {
        # Written down because the JAX checkpoint and the deployable bundle are
        # different shapes, and only one of them loads on the robot.
        "bundle_format": "pytorch_lora_merged",
        "bundle_entries": ["model.safetensors", "config.json", "norm_stats.json", "manifest.json"],
        "config_name": args.config_name,
        "exp_name": args.exp_name,
        "step": args.step,
        "repo_id": repo_id,
        # Everything the executor has to get right, stated rather than implied.
        "state": {
            "dim": 7,
            "layout": "[j1..j6 degrees absolute, gripper]",
            "prenormalised": False,
            "note": "pass raw values; the policy applies norm_stats internally",
        },
        "action": {
            "shape": [int(model.action_horizon), 7],
            "semantics": "sequential joint delta, deg/frame",
            "gripper_index": 6,
            "gripper_semantics": "absolute, not a delta",
            "denormalised_on_output": True,
            "integration": (
                "anchor q=q_measured at the start of a chunk, accumulate deltas open-loop, "
                "re-anchor when the next chunk is taken"
            ),
            # How many of the 16 get executed before re-inferring is NOT settled for
            # this arm. The E6 runtime consumed the first 8 and re-inferred, which
            # makes the policy refresh 2 Hz while commands still leave at 16 Hz.
            # Recorded as unset so nobody reads a default as a decision.
            "steps_executed_per_chunk": None,
            "steps_executed_precedent": {
                "value": 8, "source": "E6 ROS2 receding-horizon runtime",
                "note": "not verified on xArm6",
            },
        },
        "control": {"fps": 16, "dt_ms": 62.5, "chunk_seconds": int(model.action_horizon) / 16},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    shutil.copy2(norm, out / "norm_stats.json")
    for extra in ("e7_context.json", "tasks.jsonl"):
        if (meta_dir / extra).is_file():
            shutil.copy2(meta_dir / extra, out / extra)

    if args.no_params:
        print("  (--no-params: weights not copied)")
    else:
        print(f"  copying PyTorch bundle from {torch_dir} …")
        shutil.copy2(torch_dir / "model.safetensors", out / "model.safetensors")
        shutil.copy2(torch_dir / "config.json", out / "config.json")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e9
    print(f"\nBundle written: {out}  ({size:.2f} GB)")
    print(f"  prompt_contract_hash {manifest['prompt_contract_hash']}")
    print(f"  style {manifest['prompt_style']}  seq {manifest['total_sequence_length']}"
          f"  chunk {manifest['control']['chunk_seconds']:.3f}s")
    print("\nBundle contents: model.safetensors · config.json · norm_stats.json · manifest.json")
    print("  (no params/ and no tokenizer_config/ — those are not part of this format)")
    print("\nJetson side:")
    print("  1. install the SAME e7_prompt version (pip install examples/e7/e7_prompt)")
    print("  2. compare the rendered prompt strings against meta/tasks.jsonl, and the")
    print("     plaintext manifest values (rig_layout, tokenizer.discrete_state_input).")
    print("     Do NOT compare prompt_contract_hash across machines: the payloads are")
    print("     assembled independently and never agree. That hash is for this repo's")
    print("     own export-time check, where both sides come from the same package.")
    print("  3. target_shelf is NOT pinned here. Signs get shuffled, so read the live")
    print("     layout and reverse-index it; reference_layout is a default, not truth")


if __name__ == "__main__":
    main(tyro.cli(Args))
