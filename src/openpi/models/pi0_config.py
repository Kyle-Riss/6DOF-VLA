import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Optional LoRA on the SigLIP vision tower (separate from
    # ``paligemma_variant`` which controls the Gemma LLM side). Set
    # ``vision_lora_rank`` to a positive int to enable; layers outside
    # ``vision_lora_layer_range`` (inclusive, 0-indexed) are masked out.
    # The base SigLIP weights remain unchanged regardless — LoRA is added
    # as a parallel residual.
    vision_lora_rank: int | None = None
    vision_lora_alpha: float = 16.0
    vision_lora_layer_range: tuple[int, int] | None = None

    # Optional LoRA layer scoping for the action expert (Gemma 300m, depth 18).
    # Only meaningful when ``action_expert_variant`` already enables LoRA
    # (e.g. ``"gemma_300m_lora"``). Inclusive 0-indexed range; layers outside
    # have their LoRA contribution multiplied by 0 (no forward effect, no
    # gradient flow). When ``None``, LoRA is active on all 18 layers (legacy
    # v2/v3 behavior). Used by v4 to constrain which expert layers adapt.
    action_expert_lora_layer_range: tuple[int, int] | None = None

    # Per-dimension loss weights applied to the squared flow-matching error
    # before averaging over action_dim. Length must equal action_dim (32).
    # E.g. set index 3 to 3.0 to up-weight j4 during training (v7).
    # When None, all dimensions are weighted equally (legacy behavior).
    action_loss_weights: tuple[float, ...] | None = None

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


def freeze_filter_vlm_frozen_action_expert_lora_only() -> nnx.filterlib.Filter:
    """Freeze all of ``PaliGemma`` except action-expert LoRA, for narrow π0.5 fine-tuning.

    Use with ``paligemma_variant="gemma_2b"`` and ``action_expert_variant="gemma_300m_lora"``.

    :meth:`Pi0Config.get_freeze_filter` does not freeze SigLIP (``PaliGemma/img``) or the main Gemma stack when only
    the action expert has LoRA, which would train the full vision-language tower. This filter freezes both the image
    encoder and all LLM base weights, while keeping trainable only (1) LoRA tensors on the action expert (paths
    matching both ``lora`` and ``_1``) and (2) the small action-side heads (``action_in_proj``, ``time_mlp_*``,
    ``action_out_proj``), which live outside ``PaliGemma``.
    """
    llm = nnx_utils.PathRegex("PaliGemma/llm/.*")
    img = nnx_utils.PathRegex("PaliGemma/img/.*")
    has_lora = nnx_utils.PathRegex(".*lora.*")
    has_1 = nnx_utils.PathRegex(".*_1.*")
    expert_lora = nnx.All(has_lora, has_1)
    freeze_llm = nnx.All(llm, nnx.Not(expert_lora))
    return nnx.Any(img, freeze_llm)


def freeze_filter_v3_vision_late_lora() -> nnx.filterlib.Filter:
    """Freeze base PaliGemma, train only (action-expert LoRA + vision LoRA + action heads).

    Use with ``paligemma_variant="gemma_2b"``, ``action_expert_variant="gemma_300m_lora"``,
    AND ``vision_lora_rank`` set on :class:`Pi0Config`.

    Trainable:
      - Action-expert LoRA tensors  (paths matching both ``lora`` and ``_1``)
      - Vision LoRA tensors          (paths under ``PaliGemma/img/...`` containing ``lora``)
      - Action-side heads outside ``PaliGemma`` (``action_in_proj``, ``time_mlp_*``, ``action_out_proj``)

    Frozen:
      - All LLM base weights (``PaliGemma/llm/...`` minus action-expert LoRA)
      - All SigLIP base weights (``PaliGemma/img/...`` minus vision LoRA)

    Note: layers excluded by ``vision_lora_layer_range`` still allocate LoRA params
    (because :func:`scan` stacks them along the depth axis), but their contribution
    is multiplied by a zero mask so they never influence the loss and never receive
    gradient. They effectively stay at init values throughout training.
    """
    llm = nnx_utils.PathRegex("PaliGemma/llm/.*")
    img = nnx_utils.PathRegex("PaliGemma/img/.*")
    has_lora = nnx_utils.PathRegex(".*lora.*")
    has_1 = nnx_utils.PathRegex(".*_1.*")
    expert_lora = nnx.All(has_lora, has_1)
    img_lora = nnx.All(img, has_lora)
    freeze_llm = nnx.All(llm, nnx.Not(expert_lora))
    freeze_img = nnx.All(img, nnx.Not(img_lora))
    return nnx.Any(freeze_img, freeze_llm)


def freeze_filter_v4_combined_lora() -> nnx.filterlib.Filter:
    """Same gradient mask as v3: vision LoRA + action-expert LoRA + small action heads trainable.

    v4 adds layer-range scoping on the action expert via
    :attr:`Pi0Config.action_expert_lora_layer_range`, but that scoping happens
    at forward time inside ``gemma.Module`` (via per-layer mask × LoRA output).
    The freeze filter — which decides which params get a gradient slot in the
    optimizer — does not need to change: out-of-range expert LoRA params are
    still allocated, still listed as "trainable" by the filter, and the mask
    zeroes their forward contribution so their gradient is exactly 0. This
    function therefore delegates to :func:`freeze_filter_v3_vision_late_lora`
    and is kept as an alias for clarity in v4 ``TrainConfig``s.
    """
    return freeze_filter_v3_vision_late_lora()
