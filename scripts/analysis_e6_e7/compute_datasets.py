#!/usr/bin/env python3
"""
E6 정책 실험 검증: 캐시된 LeRobot 데이터셋에서 action contract / 분포 / phase / lift 궤적 실증.

대상: ~/.cache/huggingface/lerobot/Kyle-Riss/dobot_e6_pick_place_orange_{v16,v13}
출력: docs/figures/e6_e7_policy/data/datasets_analysis.json

검증 항목:
  1) action-semantics: joint action == delta(state[t+1]-state[t]) 인가? vs next-position 가설과 RMSE 비교
  2) gripper: action[6] 이 absolute(next gripper, {0,1}) 인가?
  3) v16 per-joint 분포 (clean 여부 실증; 오염 bimodal 반증)
  4) phase 분포 (task_index -> phase)
  5) lift 궤적: 대표 에피소드에서 pick 후 j2/j3 동시 감소
  6) v13 gripper delta {+1,-1,0} 카운트 (release fix 실증)

읽기 전용. 재현: `uv run --no-sync python scripts/analysis_e6_e7/compute_datasets.py`
"""
import json
import glob
import os
import numpy as np
import pyarrow.parquet as pq

HOME = os.path.expanduser("~")
BASE = os.path.join(HOME, ".cache/huggingface/lerobot/Kyle-Riss")
REPO = "/home/billy/26kp/openpi_upstream_clean"
OUT_DIR = os.path.join(REPO, "docs/figures/e6_e7_policy/data")
os.makedirs(OUT_DIR, exist_ok=True)

JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]


def load_episode(path):
    t = pq.read_table(path, columns=["state", "action", "task_index", "frame_index"])
    state = np.array(t.column("state").to_pylist(), dtype=np.float64)   # (T,7)
    action = np.array(t.column("action").to_pylist(), dtype=np.float64)  # (T,7)
    task_index = np.array(t.column("task_index").to_pylist(), dtype=np.int64)
    return state, action, task_index


def analyze_v16():
    ds = os.path.join(BASE, "dobot_e6_pick_place_orange_v16")
    files = sorted(glob.glob(os.path.join(ds, "data/chunk-000/episode_*.parquet")))
    tasks = {}
    with open(os.path.join(ds, "meta/tasks.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            tasks[d["task_index"]] = d["task"]

    # action-semantics: within-episode delta vs next-position RMSE, per joint
    sq_delta = np.zeros(7)   # sum sq err for action vs (state[t+1]-state[t])
    sq_nextpos = np.zeros(7)  # sum sq err for action vs state[t+1]
    n_pairs = 0
    # gripper: is action[6] in {0,1}? and == next gripper?
    grip_action_vals = []
    # distributions: collect all state joints (subsample for hist)
    state_all = []
    # phase distribution
    phase_frames = {}
    # lift trajectory: store one representative left->right episode
    lift_example = None

    for fp in files:
        state, action, ti = load_episode(fp)
        T = state.shape[0]
        if T < 2:
            continue
        d = state[1:] - state[:-1]          # (T-1,7) true delta
        a = action[:-1]                     # action at t (aligns with delta t->t+1)
        nextp = state[1:]                   # next position
        sq_delta += np.sum((a - d) ** 2, axis=0)
        sq_nextpos += np.sum((a - nextp) ** 2, axis=0)
        n_pairs += (T - 1)
        grip_action_vals.append(action[:, 6])
        state_all.append(state[::3])  # subsample every 3rd frame for histograms
        for k in ti:
            ph = tasks.get(int(k), str(k))
            phase_frames[ph] = phase_frames.get(ph, 0) + 1
        # pick a representative L->R episode for lift trajectory (task_index 0 anchor)
        if lift_example is None and 0 in set(ti.tolist()):
            # find gripper close event (0->1)
            g = state[:, 6]
            close_idx = None
            for i in range(1, T):
                if g[i - 1] < 0.5 <= g[i]:
                    close_idx = i
                    break
            if close_idx is not None and close_idx + 60 < T:
                lift_example = {
                    "file": os.path.basename(fp),
                    "close_idx": int(close_idx),
                    "frames": list(range(T)),
                    "j2": state[:, 1].tolist(),
                    "j3": state[:, 2].tolist(),
                    "gripper": state[:, 6].tolist(),
                }

    rmse_delta = np.sqrt(sq_delta / n_pairs)
    rmse_nextpos = np.sqrt(sq_nextpos / n_pairs)
    grip = np.concatenate(grip_action_vals)
    state_all = np.concatenate(state_all, axis=0)

    # gripper next-value RMSE (absolute check): action[t,6] vs state[t+1,6]
    # recompute cleanly
    sq_grip_next = 0.0
    sq_grip_delta = 0.0
    ng = 0
    for fp in files:
        state, action, ti = load_episode(fp)
        if state.shape[0] < 2:
            continue
        a6 = action[:-1, 6]
        gnext = state[1:, 6]
        gdelta = state[1:, 6] - state[:-1, 6]
        sq_grip_next += np.sum((a6 - gnext) ** 2)
        sq_grip_delta += np.sum((a6 - gdelta) ** 2)
        ng += len(a6)

    result = {
        "dataset": "dobot_e6_pick_place_orange_v16",
        "n_episodes": len(files),
        "n_frames": int(n_pairs + len(files)),
        "tasks": {int(k): v for k, v in tasks.items()},
        "action_semantics": {
            "joint_names": JOINTS,
            "rmse_action_vs_delta": rmse_delta.tolist(),
            "rmse_action_vs_nextpos": rmse_nextpos.tolist(),
            "n_pairs": int(n_pairs),
            "note": "joints(0-5): rmse_vs_delta≈0 이면 delta 의미. gripper(6)는 아래 gripper 블록 참조.",
        },
        "gripper": {
            "unique_rounded": sorted(list({round(float(x), 3) for x in np.unique(np.round(grip, 3))}))[:10],
            "min": float(grip.min()),
            "max": float(grip.max()),
            "mean": float(grip.mean()),
            "frac_zero": float(np.mean(np.isclose(grip, 0.0))),
            "frac_one": float(np.mean(np.isclose(grip, 1.0))),
            "rmse_vs_next_absolute": float(np.sqrt(sq_grip_next / ng)),
            "rmse_vs_delta": float(np.sqrt(sq_grip_delta / ng)),
        },
        "joint_distributions": {
            JOINTS[j]: {
                "mean": float(state_all[:, j].mean()),
                "std": float(state_all[:, j].std()),
                "min": float(state_all[:, j].min()),
                "max": float(state_all[:, j].max()),
                "hist_counts": np.histogram(state_all[:, j], bins=40)[0].tolist(),
                "hist_edges": np.histogram(state_all[:, j], bins=40)[1].tolist(),
            }
            for j in range(7)
        },
        "phase_frames": phase_frames,
        "lift_example": lift_example,
    }
    return result


def analyze_v13_gripper():
    ds = os.path.join(BASE, "dobot_e6_pick_place_orange_v13")
    files = sorted(glob.glob(os.path.join(ds, "data/chunk-000/episode_*.parquet")))
    close = openn = noop = 0
    for fp in files:
        t = pq.read_table(fp, columns=["action"])
        a = np.array(t.column("action").to_pylist(), dtype=np.float64)
        g = a[:, 6]
        close += int(np.sum(g > 0.5))
        openn += int(np.sum(g < -0.5))
        noop += int(np.sum(np.abs(g) <= 0.5))
    return {
        "dataset": "dobot_e6_pick_place_orange_v13",
        "note": "v13 gripper = delta {+1 close, -1 open, 0 noop}. release fix 후 open(-1) 이 0이 아니어야 함.",
        "gripper_action_counts": {"close_+1": close, "open_-1": openn, "noop_0": noop},
    }


def main():
    out = {"v16": analyze_v16(), "v13": analyze_v13_gripper()}
    with open(os.path.join(OUT_DIR, "datasets_analysis.json"), "w") as f:
        json.dump(out, f)

    v16 = out["v16"]
    print("=== v16 action-semantics (RMSE) ===")
    print(f"{'joint':>8} {'vs_delta':>12} {'vs_nextpos':>12}")
    for j, name in enumerate(JOINTS):
        print(f"{name:>8} {v16['action_semantics']['rmse_action_vs_delta'][j]:>12.4f} "
              f"{v16['action_semantics']['rmse_action_vs_nextpos'][j]:>12.4f}")
    g = v16["gripper"]
    print("\n=== v16 gripper (action[6]) ===")
    print(f"min={g['min']:.3f} max={g['max']:.3f} mean={g['mean']:.3f} "
          f"frac0={g['frac_zero']:.3f} frac1={g['frac_one']:.3f}")
    print(f"rmse_vs_next_absolute={g['rmse_vs_next_absolute']:.4f}  rmse_vs_delta={g['rmse_vs_delta']:.4f}")
    print("\n=== v16 phase frames ===")
    for k, v in sorted(v16["phase_frames"].items(), key=lambda x: -x[1]):
        print(f"  {v:>6}  {k}")
    print("\n=== v13 gripper counts ===")
    print(" ", out["v13"]["gripper_action_counts"])
    le = v16["lift_example"]
    print(f"\nlift_example: {le['file'] if le else None} close_idx={le['close_idx'] if le else None}")
    print(f"\nwrote {os.path.join(OUT_DIR, 'datasets_analysis.json')}")


if __name__ == "__main__":
    main()
