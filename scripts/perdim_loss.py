"""Break the flow-matching loss down by action dimension.

``compute_loss`` averages the squared error over the action dimension before
returning, so the reported number cannot say which dimensions it came from.
This replicates that computation and stops one step earlier, keeping the
per-dimension term.

Two things this is meant to answer:

  * how much of the loss belongs to the 25 padding dimensions, given that the
    model's action space is 32-wide and the robot only has 7 real dimensions;
  * how the 7 real dimensions divide it up.

Run against the checkpoint whose loss you want to attribute:

    uv run scripts/perdim_loss.py --config-name pi05_e6_v23_lora \\
        --checkpoint checkpoints/pi05_e6_v23_lora/e6_2cam_lora_v23/19999

The __main__ guard is load-bearing: without it every dataloader worker
re-imports this module, re-instantiates the model and exhausts the GPU.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.models.pi0 import make_attn_mask


@dataclasses.dataclass
class Args:
    config_name: str
    checkpoint: pathlib.Path
    batches: int = 6
    """Batches to average over. The estimate is noisy below about 4."""
    batch_size: int = 8
    seed: int = 0
    out: pathlib.Path | None = None
    """Optional JSON destination for the raw per-dimension means."""


def per_dim_sq_err(model, rng, observation, actions) -> at.Array:
    """``Pi0.compute_loss`` with the final ``mean(axis=-1)`` left off.

    Kept deliberately close to the original so a change there is easy to
    mirror here; the divergence is the last line.
    """
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(
        preprocess_rng,
        observation,
        train=False,
        image_keys=model._image_keys,  # noqa: SLF001
        wrist_image_keys=model._wrist_image_keys,  # noqa: SLF001
    )

    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])

    # compute_loss would return mean(..., axis=-1) here.
    return jnp.square(v_t - u_t)


def main(args: Args) -> None:
    train_cfg = _config.get_config(args.config_name)
    loader = _data_loader.create_data_loader(
        dataclasses.replace(train_cfg, batch_size=args.batch_size),
        shuffle=False,
        num_batches=args.batches,
    )

    print(f"loading {args.checkpoint} …")
    params = _model.restore_params(args.checkpoint / "params", restore_type=jnp.ndarray, dtype=jnp.bfloat16)
    model = train_cfg.model.load(params)

    rng = jax.random.key(args.seed)
    totals = None
    n = 0
    for observation, actions in tqdm.tqdm(loader, total=args.batches, desc="batches"):
        rng, step_rng = jax.random.split(rng)
        sq = per_dim_sq_err(model, step_rng, observation, actions)  # (b, ah, dim)
        totals = np.asarray(sq.mean(axis=(0, 1))) if totals is None else totals + np.asarray(sq.mean(axis=(0, 1)))
        n += 1
        if n >= args.batches:
            break
    per_dim = totals / n

    real = per_dim[:7]
    pad = per_dim[7:]
    names = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]

    print(f"\naveraged over {n} batches of {args.batch_size}, horizon {train_cfg.model.action_horizon}")
    print(f"reported loss (mean over all {len(per_dim)} dims) = {per_dim.mean():.6f}\n")
    print(f"{'dim':>4s} {'name':8s} {'sq_err':>10s} {'% of real':>10s} {'% of total':>11s}")
    for i, v in enumerate(real):
        print(f"{i:>4d} {names[i]:8s} {v:10.5f} {v / real.sum() * 100:9.1f}% {v / per_dim.sum() * 100:10.1f}%")
    print(f"\n  7 real dims   sum {real.sum():.5f}   mean {real.mean():.5f}")
    print(f" 25 pad dims    sum {pad.sum():.5f}   mean {pad.mean():.5f}")
    print(f"\n  padding is {pad.sum() / per_dim.sum() * 100:.2f}% of the total loss")
    print(f"  real-dim mean is {real.mean() / pad.mean():.0f}x the padding mean")
    top = int(np.argmax(real))
    print(f"  largest real dim: {names[top]} at {real[top] / real.sum() * 100:.1f}% of the real-dim loss")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "config": args.config_name,
                    "checkpoint": str(args.checkpoint),
                    "batches": n,
                    "batch_size": args.batch_size,
                    "per_dim": per_dim.tolist(),
                    "names": names,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
