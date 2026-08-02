import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_e7_example() -> dict:
    """Creates a random input example for an E7-style policy (xArm 6)."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/exterior_image_2_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(7),  # xArm 6: [j1..j6, gripper] = 7D
        "prompt": "approach the target object",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class E7Inputs(transforms.DataTransformFn):
    """xArm 6 (E7) inputs — 7D state/action, same contract as E6 v16+.

    Collection note: the xArm6 teleop path commands Cartesian velocity
    (``/xarm/vc_set_cartesian_velocity`` @ 20 Hz), but the training label is
    ``q_measured[t+1] - q_measured[t]`` from the 16 Hz joint recording, so the
    joint-delta contract holds regardless of how the arm was driven.
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        hik_image = _parse_image(data["observation/exterior_image_1_left"])
        zed_image = _parse_image(data["observation/exterior_image_2_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                # HIK → base slot, ZED → left_wrist slot. Unlike E6, ``right_wrist_0_rgb``
                # is NOT emitted: xArm6 has no third camera, and a zeros+mask-False slot
                # still costs 256 sequence tokens and a full SigLIP forward while
                # contributing nothing. Requires the matching ``image_keys`` on
                # :class:`Pi0Config` (see ``pi05_e7_v1_lora``).
                names = ("base_0_rgb", "left_wrist_0_rgb")
                images = (hik_image, zed_image)
                image_masks = (np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (hik_image, zed_image, np.zeros_like(hik_image))
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        # xArm 6: [j1..j6, gripper] = 7D — same shape as E6.
        # No DROID 8D alignment (gripper stays at index 6, NOT 7). This matches
        # E6 v23 (``align_droid_state=False``), which is the reference run for the
        # E6→E7 cross-embodiment comparison; inserting a dummy j7 here would change
        # the action contract and break that comparison.
        state = np.asarray(data["observation/state"])  # (7,)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class E7Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        acts = np.asarray(data["actions"])
        # action contract: [Δj1..Δj6, gripper] = 7D (same layout as E6 v16+)
        # joint(6D): velocity delta (deg/frame) — target = q_measured + Δ, see
        #            ``delta_reference`` in the executor config (NOT q_target_prev + Δ)
        # gripper(1D): absolute command, 0.0 = open .. 1.0 = close.
        #   ⚠ OPEN: binary vs continuous is undecided. E6 was a vacuum (true {0,1});
        #   the xArm6 G2 is a parallel gripper with continuous aperture. Record the
        #   raw continuous value at collection — binarising later is lossless, the
        #   reverse is not — and fix the contract in the conversion script, not here.
        #   Binary executor rule (cmd = 1 if action[6] > 0.5) applies only if the
        #   conversion thresholds it.
        return {"actions": acts[:, :7]}
