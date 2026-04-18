#!/usr/bin/env python3
"""Quick shape checks for E6 v1 policy path (no dataset, no GPU).

Run from repo root, after `uv sync` (or any env with openpi + numpy), e.g.:

  PYTHONPATH=src uv run python scripts/smoke_e6_policy.py
"""

from __future__ import annotations

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import e6_policy
from openpi.training import config as tc


def main() -> None:
    ex = dict(e6_policy.make_e6_example())
    ex["actions"] = np.random.randn(16, 7).astype(np.float32)

    inp = e6_policy.E6Inputs(model_type=_model.ModelType.PI05)(ex)
    assert inp["state"].shape == (7,), inp["state"].shape
    assert inp["actions"].shape == (16, 7), inp["actions"].shape
    assert sum(inp["image_mask"].values()) == 2

    padded = transforms.PadStatesAndActions(32)(dict(inp))
    assert padded["state"].shape == (32,), padded["state"].shape
    assert padded["actions"].shape == (16, 32), padded["actions"].shape

    out = e6_policy.E6Outputs()({"actions": np.random.randn(16, 32)})
    assert out["actions"].shape == (16, 7), out["actions"].shape

    cfg = tc.get_config("pi05_e6_v1")
    assert cfg.model.action_dim == 32
    assert cfg.model.discrete_state_input is False

    print("smoke_e6_policy: OK")


if __name__ == "__main__":
    main()
