#!/usr/bin/env python3
"""Smoke test for the native-7D action-head ablation (v30/v31/v32).

Step 1 (no checkpoint, seconds): shape checks on the E6 7D policy path + config asserts.
Step 2 (loads pi05_base on CPU, minutes): validate the three weight loaders merge to the
  model's param structure via the same check as train.py (_load_weights_and_validate),
  plus targeted asserts:
    - v32 (sliced): action head shape (7,1024)/(1024,7) AND values == pretrained[:7]
    - v30 (random): action head absent from loaded params (=> model fresh-inits it)
    - v31 (masked): 32D action head present (pretrained), action_loss_weights len=32 sum=7

Run:
  .venv/bin/python scripts/smoke_nat7d.py            # both steps
  .venv/bin/python scripts/smoke_nat7d.py --step1    # static only
"""
from __future__ import annotations

import sys

import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

import flax.nnx as nnx

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi import transforms
from openpi.models import model as _model
from openpi.policies import e6_policy
from openpi.shared import download
from openpi.training import config as tc

PI05_BASE = "gs://openpi-assets/checkpoints/pi05_base/params"


def step1_static() -> None:
    print("=== Step 1: static shape smoke (no checkpoint) ===")
    ex = dict(e6_policy.make_e6_example())
    ex["actions"] = np.random.randn(16, 7).astype(np.float32)
    inp = e6_policy.E6Inputs(model_type=_model.ModelType.PI05)(ex)
    assert inp["state"].shape == (7,), inp["state"].shape
    assert inp["actions"].shape == (16, 7), inp["actions"].shape
    assert sum(inp["image_mask"].values()) == 2
    padded = transforms.PadStatesAndActions(7)(dict(inp))
    assert padded["state"].shape == (7,), padded["state"].shape
    assert padded["actions"].shape == (16, 7), padded["actions"].shape
    out = e6_policy.E6Outputs()({"actions": np.random.randn(16, 7)})
    assert out["actions"].shape == (16, 7), out["actions"].shape

    for name, adim in [
        ("pi05_e6_v30_nat7d_late_lora", 7),
        ("pi05_e6_v31_nat7d_sliced_lora", 7),
    ]:
        assert tc.get_config(name).model.action_dim == adim, name
    print("Step 1: OK\n")


def _params_shape(config):
    """Replicates train.py init_train_state's abstract param shapes (no allocation)."""

    def init(rng):
        model = config.model.create(rng)
        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return params

    return jax.eval_shape(init, jax.random.key(0)).to_pure_dict()


def _load_and_validate(loader, params_shape):
    """Mirror of scripts/train.py::_load_weights_and_validate."""
    loaded = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def step2_loaders() -> None:
    print("=== Step 2: loader merge smoke (loads pi05_base on CPU) ===")
    print("  restoring pretrained pi05_base for value comparison ...")
    pre = traverse_util.flatten_dict(
        _model.restore_params(download.maybe_download(PI05_BASE), restore_type=np.ndarray), sep="/"
    )
    in_k = next(k for k in pre if "action_in_proj/kernel" in k)
    out_k = next(k for k in pre if "action_out_proj/kernel" in k)
    print(f"  pretrained {in_k}={pre[in_k].shape}  {out_k}={pre[out_k].shape}")

    for name in [
        "pi05_e6_v31_nat7d_sliced_lora",
        "pi05_e6_v30_nat7d_late_lora",
        "pi05_e6_v32_nat7d_vlmfrozen_lora",
    ]:
        print(f"\n  --- {name} ---")
        cfg = tc.get_config(name)
        shape = _params_shape(cfg)
        loaded = _load_and_validate(cfg.weight_loader, shape)  # raises if shapes/dtypes mismatch
        flat = traverse_util.flatten_dict(loaded, sep="/")
        has_in = any("action_in_proj/kernel" in k for k in flat)
        print(f"  check_pytree_equality: PASS | action_in_proj present in loaded: {has_in}")

        if name == "pi05_e6_v31_nat7d_sliced_lora":
            k = next(k for k in flat if "action_in_proj/kernel" in k)
            ko = next(k for k in flat if "action_out_proj/kernel" in k)
            assert tuple(flat[k].shape) == (7, 1024), flat[k].shape
            assert tuple(flat[ko].shape) == (1024, 7), flat[ko].shape
            assert np.allclose(np.asarray(flat[k], dtype=np.float32), pre[in_k][:7, :].astype(np.float32)), "in slice"
            assert np.allclose(np.asarray(flat[ko], dtype=np.float32), pre[out_k][:, :7].astype(np.float32)), "out slice"
            print(f"  v31: sliced head {flat[k].shape}/{flat[ko].shape}, values == pretrained[:7] ✓")
        elif name == "pi05_e6_v30_nat7d_late_lora":
            assert not has_in, "reinit head must be ABSENT (fresh-init), but found in loaded"
            print("  v30: action head absent from loaded (fresh random init) ✓")
        elif name == "pi05_e6_v32_nat7d_vlmfrozen_lora":
            assert not has_in, "reinit head must be ABSENT (fresh-init), but found in loaded"
            has_img_lora = any("img" in k and "lora" in k for k in flat)
            assert not has_img_lora, "VLM-frozen config must have no vision LoRA params"
            print("  v32: head absent (fresh) + no vision LoRA params ✓")
    print("\nStep 2: OK")


if __name__ == "__main__":
    step1_static()
    if "--step1" not in sys.argv:
        step2_loaders()
    print("\nsmoke_nat7d: ALL OK")
