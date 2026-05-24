import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_e6_example() -> dict:
    """Creates a random input example for an E6-style policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/exterior_image_2_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(7),
        "prompt": "approach red object",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class E6Inputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType
    # Align state/action to DROID 8D format: insert dummy j7=0 at index 6 so
    # gripper lands at index 7, matching pi05_base pretraining (Franka 7-DOF).
    # v14+ only; set False to preserve v1-v13 behaviour.
    align_droid_state: bool = False

    def __call__(self, data: dict) -> dict:
        hik_image = _parse_image(data["observation/exterior_image_1_left"])
        zed_image = _parse_image(data["observation/exterior_image_2_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                # HIK → base slot, ZED → left_wrist slot; right_wrist padded with zeros.
                images = (hik_image, zed_image, np.zeros_like(hik_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (hik_image, zed_image, np.zeros_like(hik_image))
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        raw_state = np.asarray(data["observation/state"])  # (7,)
        if self.align_droid_state:
            # [j1..j6, gripper] → [j1..j6, 0, gripper]: gripper at index 7 = DROID format
            state = np.insert(raw_state, 6, 0.0)  # (8,)
        else:
            state = raw_state

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            acts = np.asarray(data["actions"])
            if self.align_droid_state:
                # [Δj1..Δj6, Δgripper] → [Δj1..Δj6, 0, Δgripper]
                acts = np.insert(acts, 6, 0.0, axis=-1)  # (..., 8)
            inputs["actions"] = acts

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class E6Outputs(transforms.DataTransformFn):
    # Must match E6Inputs.align_droid_state for the same training run.
    align_droid_state: bool = False

    def __call__(self, data: dict) -> dict:
        acts = np.asarray(data["actions"])
        if self.align_droid_state:
            # Model outputs at index 7 = Δgripper; skip dummy j7 at index 6.
            return {"actions": np.concatenate([acts[:, :6], acts[:, 7:8]], axis=-1)}
        # Legacy 7D: first 7 dims = [Δj1..Δj6, Δgripper].
        return {"actions": acts[:, :7]}
