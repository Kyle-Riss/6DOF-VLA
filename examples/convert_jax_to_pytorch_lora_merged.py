"""Convert JAX checkpoint to PyTorch with LoRA adapters merged into base weights.

The original ``convert_jax_model_to_pytorch.py`` only reads base weights
(``q_einsum_1/w``, ``kv_einsum_1/w``, ``attn_vec_einsum_1/w``,
``mlp_1/gating_einsum``, ``mlp_1/linear``) and silently drops the trained LoRA
adapters (``lora_a`` / ``lora_b``). The PyTorch model class has no LoRA module,
so the resulting checkpoint is effectively the base ``pi05`` model with newly
trained action heads only — exactly the diagnosis seen for ``pi05_e6_v2_lora``.

This wrapper merges each LoRA pair into its base weight before slicing:

    W_merged = W + scaling * (lora_a @ lora_b)

For ``gemma_300m_lora`` (rank=32, alpha=32, rslora=False):
    scaling = alpha / rank = 1.0

Then the existing slicing logic from ``convert_jax_model_to_pytorch`` is reused.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
from typing import Literal

from flax.nnx import traversals
import numpy as np
import safetensors
import torch
import tyro

import openpi.models.gemma
import openpi.models.model
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.training.config as _config

_THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from convert_jax_model_to_pytorch import (  # noqa: E402
    slice_gemma_state_dict,
    slice_paligemma_state_dict,
)


# rank=32, alpha=32, rslora=False -> scaling = alpha / rank = 1.0
LORA_SCALING = 1.0


def _detect_suffix(flat_params: dict) -> str:
    """Detect whether keys end with '/value' (orbax wrapping) or not."""
    return "/value" if any(k.endswith("/value") for k in flat_params) else ""


def _merge_pair(flat_params: dict, base_key: str, a_key: str, b_key: str, label: str) -> bool:
    if base_key not in flat_params:
        print(f"  [skip] missing base: {base_key}")
        return False
    if a_key not in flat_params or b_key not in flat_params:
        print(f"  [skip] missing lora: {label}")
        return False

    base = flat_params[base_key]
    a = flat_params.pop(a_key)
    b = flat_params.pop(b_key)

    # matmul on last two dims, broadcasting any leading layer/head dims.
    delta = np.matmul(a.astype(np.float32), b.astype(np.float32)) * LORA_SCALING

    if delta.shape != base.shape:
        raise ValueError(f"shape mismatch for {label}: delta {delta.shape} vs base {base.shape}")

    merged = (base.astype(np.float32) + delta).astype(base.dtype)
    flat_params[base_key] = merged
    delta_rel = float(np.linalg.norm(delta) / max(np.linalg.norm(base.astype(np.float32)), 1e-12))
    print(f"  [ok] {label:<28s} base{tuple(base.shape)}  ||delta||/||base||={delta_rel:.4f}")
    return True


def merge_lora_into_base(flat_params: dict) -> dict:
    """Merge LoRA adapters into action-expert base weights, then drop lora keys."""
    suffix = _detect_suffix(flat_params)
    print(f"Detected suffix: {suffix!r}")

    n_merged = 0
    # Attention: 3 einsums per layer block (all 18 layers stacked in axis 0).
    for name in ("attn_vec_einsum_1", "kv_einsum_1", "q_einsum_1"):
        ok = _merge_pair(
            flat_params,
            f"llm/layers/attn/{name}/w{suffix}",
            f"llm/layers/attn/{name}/lora_a{suffix}",
            f"llm/layers/attn/{name}/lora_b{suffix}",
            label=f"attn/{name}",
        )
        n_merged += int(ok)

    # FFN: 2 einsums per layer block.
    for name in ("gating_einsum", "linear"):
        ok = _merge_pair(
            flat_params,
            f"llm/layers/mlp_1/{name}{suffix}",
            f"llm/layers/mlp_1/{name}_lora_a{suffix}",
            f"llm/layers/mlp_1/{name}_lora_b{suffix}",
            label=f"mlp_1/{name}",
        )
        n_merged += int(ok)

    print(f"Merged {n_merged} LoRA pairs.")
    leftover_lora = [k for k in flat_params if "lora" in k]
    if leftover_lora:
        print(f"WARNING: {len(leftover_lora)} lora keys still present:")
        for k in leftover_lora[:10]:
            print(f"   {k}")
    return flat_params


# PyTorch state_dict prefix for SigLIP encoder layers (matches PI0Pytorch).
_VISION_PT_PREFIX = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers"


def extract_vision_lora_per_layer(
    flat_params: dict, layer_range: tuple[int, int]
) -> dict[str, torch.Tensor]:
    """Slice the stacked (27, ...) vision LoRA tensors into per-layer PyTorch keys.

    JAX side stores ``img/Transformer/encoderblock/{attn,mlp}_lora_{a,b}`` with
    shape ``(num_layers, ...)`` because :func:`flax.linen.scan` stacks all
    encoder blocks along axis 0. v3 only trains layers in ``layer_range``
    (inclusive, 0-indexed); we copy those layers into per-layer PyTorch params
    and drop the rest. Layers outside ``layer_range`` had mask=0 throughout
    training so their values stayed near init — discarding them is exact.

    The matched JAX keys are popped from ``flat_params`` so they don't trigger
    leftover-lora warnings later.
    """
    suffix = _detect_suffix(flat_params)
    lo, hi = layer_range
    out: dict[str, torch.Tensor] = {}
    n_copied = 0
    for branch in ("attn", "mlp"):
        for ab in ("a", "b"):
            jax_key = f"img/Transformer/encoderblock/{branch}_lora_{ab}{suffix}"
            if jax_key not in flat_params:
                print(f"  [vision-lora skip] missing {jax_key}")
                continue
            stacked = flat_params.pop(jax_key)
            n_layers = stacked.shape[0]
            if not (0 <= lo <= hi < n_layers):
                raise ValueError(
                    f"layer_range {layer_range} out of bounds for {jax_key} (n_layers={n_layers})"
                )
            for layer_idx in range(lo, hi + 1):
                pt_key = f"{_VISION_PT_PREFIX}.{layer_idx}.{branch}_lora_{ab}"
                out[pt_key] = torch.from_numpy(np.array(stacked[layer_idx]))
                n_copied += 1
    print(f"Vision LoRA: copied {n_copied} per-layer tensors (layers {lo}..{hi}).")
    return out


def slice_initial_orbax_checkpoint_with_lora(
    checkpoint_dir: str,
    restore_precision: str | None = None,
    vision_lora_layer_range: tuple[int, int] | None = None,
):
    params = openpi.models.model.restore_params(
        f"{checkpoint_dir}/params/", restore_type=np.ndarray, dtype=restore_precision
    )
    paligemma_flat = traversals.flatten_mapping(params["PaliGemma"], sep="/")
    print(f"Loaded {len(paligemma_flat)} paligemma flat keys.")
    paligemma_flat = merge_lora_into_base(paligemma_flat)
    vision_lora_pt_params: dict[str, torch.Tensor] = {}
    if vision_lora_layer_range is not None:
        vision_lora_pt_params = extract_vision_lora_per_layer(paligemma_flat, vision_lora_layer_range)
    print(f"After merge + vision-lora extract: {len(paligemma_flat)} paligemma keys.")
    return {
        "paligemma_params": paligemma_flat,
        "projection_params": params,
        "vision_lora_pt_params": vision_lora_pt_params,
    }


def convert_pi0_checkpoint_with_lora_merge(
    checkpoint_dir: str,
    precision: str,
    output_path: str,
    model_config: openpi.models.pi0_config.Pi0Config,
    config_name: str = "",
):
    print(f"=== Convert (LoRA-merged): {checkpoint_dir}  ->  {output_path} ===")
    print(f"Model config: {model_config}")

    initial_params = slice_initial_orbax_checkpoint_with_lora(
        checkpoint_dir,
        restore_precision="float32",
        vision_lora_layer_range=getattr(model_config, "vision_lora_layer_range", None),
    )

    if model_config.pi05:
        keys = ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
    else:
        keys = ["state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"]

    projection_params = {}
    for key in keys:
        kernel_params = initial_params["projection_params"][key]["kernel"]
        bias_params = initial_params["projection_params"][key]["bias"]
        if isinstance(kernel_params, dict):
            weight = kernel_params["value"]
            bias = bias_params["value"]
        else:
            weight = kernel_params
            bias = bias_params
        projection_params[f"{key}.weight"] = torch.from_numpy(np.array(weight)).T
        projection_params[f"{key}.bias"] = torch.from_numpy(np.array(bias))

    class PaliGemmaConfig:
        def __init__(self):
            self.vision_config = type("obj", (object,), {
                "hidden_size": 1152, "num_hidden_layers": 27, "num_attention_heads": 16,
                "intermediate_size": 4304, "patch_size": 14, "projection_dim": 2048,
            })()
            self.text_config = type("obj", (object,), {
                "hidden_size": 2048, "num_hidden_layers": 18, "num_attention_heads": 8,
                "head_dim": 256, "intermediate_size": 16384,
            })()

    paligemma_config = PaliGemmaConfig()
    action_expert_config = openpi.models.gemma.get_config("gemma_300m")

    paligemma_params, expert_params = slice_paligemma_state_dict(initial_params["paligemma_params"], paligemma_config)
    gemma_params = slice_gemma_state_dict(
        expert_params, action_expert_config, num_expert=1, checkpoint_dir=checkpoint_dir, pi05=model_config.pi05
    )

    pi0_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)
    vision_lora_pt_params = initial_params.get("vision_lora_pt_params", {})
    all_params = {**paligemma_params, **gemma_params, **projection_params, **vision_lora_pt_params}
    missing, unexpected = pi0_model.load_state_dict(all_params, strict=False)
    if unexpected:
        print(f"unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    if missing:
        print(f"missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")

    if precision == "float32":
        pi0_model = pi0_model.to(torch.float32)
    elif precision == "bfloat16":
        pi0_model = pi0_model.to(torch.bfloat16)
    else:
        raise ValueError(f"Invalid precision: {precision}")

    os.makedirs(output_path, exist_ok=True)
    safetensors.torch.save_model(pi0_model, os.path.join(output_path, "model.safetensors"))

    # Assets live inside the step folder for our checkpoints (e.g. <step>/assets/<asset_id>/...).
    assets_source = pathlib.Path(checkpoint_dir) / "assets"
    if assets_source.exists():
        assets_dest = pathlib.Path(output_path) / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_source, assets_dest)
        print(f"Copied assets from {assets_source}")
    else:
        print(f"WARNING: no assets dir at {assets_source}")

    # Everything the deployment side needs to rebuild an identical model. The
    # fields below are NOT cosmetic: ``image_keys`` decides how many camera slots
    # the sequence has (a 2-slot checkpoint fed through the 3-slot default silently
    # produces a different token layout), and ``vision_lora_*`` must match or the
    # extracted per-layer LoRA tensors have nowhere to load into. ``config_name`` is
    # recorded so the executor can resolve the same TrainConfig rather than
    # reconstructing one by hand.
    config_dict = {
        "config_name": config_name,
        "action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "max_token_len": model_config.max_token_len,
        "pi05": model_config.pi05,
        "paligemma_variant": model_config.paligemma_variant,
        "action_expert_variant": model_config.action_expert_variant,
        "image_keys": list(model_config.image_keys),
        "vision_lora_rank": model_config.vision_lora_rank,
        "vision_lora_alpha": model_config.vision_lora_alpha,
        "vision_lora_layer_range": list(model_config.vision_lora_layer_range)
        if model_config.vision_lora_layer_range else None,
        "action_expert_lora_layer_range": list(model_config.action_expert_lora_layer_range)
        if model_config.action_expert_lora_layer_range else None,
        "precision": precision,
        "lora_merged": True,
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Done. Saved to {output_path}")


def main(
    checkpoint_dir: str,
    config_name: str,
    output_path: str,
    precision: Literal["float32", "bfloat16", "float16"] = "bfloat16",
):
    model_config = _config.get_config(config_name).model
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise ValueError(f"Config {config_name} is not a Pi0Config")
    convert_pi0_checkpoint_with_lora_merge(checkpoint_dir, precision, output_path, model_config, config_name)


if __name__ == "__main__":
    tyro.cli(main)
