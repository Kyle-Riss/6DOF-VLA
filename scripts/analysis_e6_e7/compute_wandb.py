#!/usr/bin/env python3
"""
E6/E7 정책 실험 검증: wandb 로컬 로그에서 버전별 loss curve와 최종 loss를 추출.

입력: <repo>/wandb/run-*/files/{output.log, wandb-metadata.json, wandb-summary.json}
출력: docs/figures/e6_e7_policy/data/wandb_curves.json

output.log 포맷: "Step N: grad_norm=.., loss=.., param_norm=.."
버전 매핑: wandb-metadata.json 의 args ("pi05_e6_vN_lora --exp-name ...")에서 추출.
smoke_test / smoke_v* 런은 제외하고, 버전별로 Step 라인이 가장 많은 런을 canonical 로 선택.

읽기 전용. 재현: `uv run --no-sync python scripts/analysis_e6_e7/compute_wandb.py`
"""
import json
import re
import glob
import os

REPO = "/home/billy/26kp/openpi_upstream_clean"
WANDB = os.path.join(REPO, "wandb")
OUT_DIR = os.path.join(REPO, "docs/figures/e6_e7_policy/data")
os.makedirs(OUT_DIR, exist_ok=True)

STEP_RE = re.compile(r"^Step (\d+): grad_norm=([\d.eE+-]+), loss=([\d.eE+-]+), param_norm=([\d.eE+-]+)")
VER_RE = re.compile(r"pi05_e6_v(\d+)_lora")


def parse_output_log(path):
    steps, losses, gnorms = [], [], []
    with open(path, errors="ignore") as f:
        for line in f:
            m = STEP_RE.match(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(3)))
                gnorms.append(float(m.group(2)))
    return steps, losses, gnorms


def main():
    runs = {}  # version -> list of candidate run dicts
    for d in sorted(glob.glob(os.path.join(WANDB, "run-*"))):
        rid = d.split("-")[-1]
        meta_p = os.path.join(d, "files", "wandb-metadata.json")
        summ_p = os.path.join(d, "files", "wandb-summary.json")
        out_p = os.path.join(d, "files", "output.log")
        if not os.path.exists(meta_p):
            continue
        meta = json.load(open(meta_p))
        args = " ".join(meta.get("args", []))
        mver = VER_RE.search(args)
        if not mver:
            continue
        version = int(mver.group(1))
        exp = ""
        me = re.search(r"--exp[-_]name\s+(\S+)", args)
        if me:
            exp = me.group(1)
        # smoke test 제외
        if "smoke" in exp.lower():
            continue
        steps, losses, gnorms = ([], [], [])
        if os.path.exists(out_p):
            steps, losses, gnorms = parse_output_log(out_p)
        final_loss = final_step = None
        if os.path.exists(summ_p):
            s = json.load(open(summ_p))
            final_loss = s.get("loss")
            final_step = s.get("_step")
        runs.setdefault(version, []).append({
            "run_id": rid,
            "dir": os.path.basename(d),
            "exp_name": exp,
            "n_steps": len(steps),
            "steps": steps,
            "losses": losses,
            "grad_norms": gnorms,
            "summary_final_loss": final_loss,
            "summary_final_step": final_step,
        })

    # 버전별 canonical = Step 라인 최다
    canonical = {}
    for version, cands in runs.items():
        best = max(cands, key=lambda c: c["n_steps"])
        # 최종 loss: summary 우선, 없으면 curve 마지막
        fl = best["summary_final_loss"]
        fs = best["summary_final_step"]
        if fl is None and best["losses"]:
            fl = best["losses"][-1]
            fs = best["steps"][-1]
        canonical[version] = {
            "run_id": best["run_id"],
            "exp_name": best["exp_name"],
            "n_steps": best["n_steps"],
            "final_loss": fl,
            "final_step": fs,
            "steps": best["steps"],
            "losses": best["losses"],
            "all_run_ids": [c["run_id"] for c in cands],
        }

    out = {"canonical": {str(k): v for k, v in sorted(canonical.items())}}
    with open(os.path.join(OUT_DIR, "wandb_curves.json"), "w") as f:
        json.dump(out, f)

    print(f"{'ver':>4} {'run_id':12} {'#pts':>5} {'final_step':>10} {'final_loss':>10}  exp_name")
    for v in sorted(canonical):
        c = canonical[v]
        fl = f"{c['final_loss']:.5f}" if c["final_loss"] is not None else "?"
        print(f"v{v:<3} {c['run_id']:12} {c['n_steps']:>5} {str(c['final_step']):>10} {fl:>10}  {c['exp_name']}")
    print(f"\nwrote {os.path.join(OUT_DIR, 'wandb_curves.json')}")


if __name__ == "__main__":
    main()
