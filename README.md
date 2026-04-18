# 6DOF-VLA

π₀.₅ fine-tuning pipeline for the Dobot MagicianE6 6-DOF robot arm, based on [Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi).

## Hardware

| Component | Spec |
|-----------|------|
| Training GPU | NVIDIA RTX A5000 (25 GB) |
| Robot | Dobot MagicianE6 (6-DOF) |
| Camera 1 | HIKRobot — `exterior_image_1_left` |
| Camera 2 | ZED — `exterior_image_2_left` |
| Resolution | 224 × 224, 18 fps |

## Installation

Requires Ubuntu 22.04 and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/Kyle-Riss/6DOF-VLA.git
cd 6DOF-VLA

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

GPU memory requirements:

| Mode | Memory |
|------|--------|
| Fine-Tuning (LoRA) | > 22.5 GB |
| Fine-Tuning (Full) | > 70 GB |

## Dataset Format

E6 LeRobot dataset contract (`billy/dobot_e6_pick_place_random_v1`):

| Field | Shape | Description |
|-------|-------|-------------|
| `state` | (7,) | `[j1, j2, j3, j4, j5, j6, gripper]` — absolute degree, gripper 0~1 |
| `action` | (7,) | absolute next-step joint position (degree) + gripper cmd |
| `exterior_image_1_left` | (3, 224, 224) | HIK top-view camera |
| `exterior_image_2_left` | (3, 224, 224) | ZED scene camera |
| `prompt` | str | task language instruction |

**action semantics**: `action[t] ≈ state[t+1]` (absolute next-frame position, not delta).

Supported task prompts:

```
approach red object
pick red object
move object to left / right / middle
place object to left / right / middle
```

Data conversion:

```bash
uv run examples/e6/convert_e6_episode_to_lerobot.py --data_dir /path/to/raw_episodes
```

## Training

### 1. Compute normalization statistics

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_e6_v1
```

LoRA config reuses the same norm stats — no need to recompute separately.

### 2. Run training

**Full fine-tune** (`pi05_e6_v1`):
- Base checkpoint: `gs://openpi-assets/checkpoints/pi05_base/params`
- 20,000 steps, batch size 32

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_e6_v1 \
    --exp-name my_experiment
```

**LoRA** (`pi05_e6_v1_lora`) — recommended:
- VLM (SigLIP + Gemma) frozen, action-expert LoRA only
- 32,000 steps, batch size 8, checkpoint every 1,000 steps

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_e6_v1_lora \
    --exp-name my_experiment
```

Checkpoints are saved to:

```
checkpoints/pi05_e6_v1_lora/<exp-name>/<step>/
```

### 3. Monitor training

Training progress is logged to Weights & Biases and the console:

```
Step 9700: grad_norm=0.1026, loss=0.0089, param_norm=1803.09
```

## Model Architecture

| Item | Value |
|------|-------|
| Base model | π₀.₅ (`pi05_base`) |
| Action dim (internal) | 32 |
| Action horizon | 16 |
| State dim | 7 |
| Image slots | HIK → `base_0_rgb`, ZED → `left_wrist_0_rgb`, right masked |
| Freeze (LoRA) | SigLIP + Gemma frozen / action-expert LoRA + action heads trained |

## Acknowledgments

This repository is a fork of [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi). Core model architecture (π₀, π₀.₅), training infrastructure, and base checkpoints are from the Physical Intelligence team.
