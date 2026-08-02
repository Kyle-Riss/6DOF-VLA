#!/usr/bin/env python3
"""
E6→E7 정책 실험 검증 시각화 (F1~F10). dataviz 스킬 팔레트 준수(검증된 categorical hues).
입력: docs/figures/e6_e7_policy/data/{wandb_curves.json, datasets_analysis.json}
출력: docs/figures/e6_e7_policy/*.png (light mode, github.io 임베드용)

모든 그림 텍스트는 영문/숫자 (matplotlib Korean glyph 회피). 캡션/해설은 문서(md/html)에서.
재현: `uv run --no-sync python scripts/analysis_e6_e7/make_figures.py`
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import font_manager as fm

# 한글 폰트 등록 (캡션에 한글 사용)
_KFONT = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(_KFONT):
    fm.fontManager.addfont(_KFONT)
    _KNAME = fm.FontProperties(fname=_KFONT).get_name()
else:
    _KNAME = "DejaVu Sans"

REPO = "/home/billy/26kp/openpi_upstream_clean"
FIG = os.path.join(REPO, "docs/figures/e6_e7_policy")
DATA = os.path.join(FIG, "data")

# ---- dataviz 검증 팔레트 (light) ----
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
GOOD, WARN, CRIT = "#0ca30c", "#eda100", "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": _KNAME,
    "font.sans-serif": [_KNAME, "DejaVu Sans"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#c3c2b7",
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})


from matplotlib.ticker import FuncFormatter, NullFormatter


def plain_log_yaxis(ax):
    """로그 y축 눈금을 지수(10^-1) 대신 일반 숫자(0.1)로 → 마이너스 글리프 회피 + 가독성."""
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


wandb = json.load(open(os.path.join(DATA, "wandb_curves.json")))["canonical"]
ds = json.load(open(os.path.join(DATA, "datasets_analysis.json")))
v16 = ds["v16"]


# ============ F1: loss curves (key versions) ============
def f1_loss_curves():
    keys = [("2", "v2 400ep abs-pos (FAIL, 80k)", RED),
            ("6", "v6 contaminated data", ORANGE),
            ("8", "v8 clean re-collect", YELLOW),
            ("10", "v10 dual-home", AQUA),
            ("16", "v16 absolute gripper", BLUE),
            ("23", "v23 best (vision 18-26)", GREEN)]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for k, label, c in keys:
        if k not in wandb:
            continue
        st = np.array(wandb[k]["steps"]); ls = np.array(wandb[k]["losses"])
        if len(st) == 0:
            continue
        ax.plot(st, ls, color=c, lw=2, label=label)
    ax.axvline(25000, color=MUTED, ls="--", lw=1)
    ax.text(25600, 0.07, "plateau ~25k", color=MUTED, fontsize=9)
    ax.set_yscale("log")
    plain_log_yaxis(ax)
    ax.set_xlabel("training step"); ax.set_ylabel("loss (log)")
    ax.set_title("F1. Loss curves — all versions converge ~0.01\n"
                 "loss was never the discriminator (failures were data/contract, not fit)",
                 fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, "F1_loss_curves.png")


# ============ F2: final loss per version ============
def f2_final_loss():
    vers = sorted(wandb.keys(), key=int)
    finals = [wandb[v]["final_loss"] for v in vers]
    # outcome classification (from verified history)
    fail = {"1", "2", "3", "4", "5", "6", "7"}       # trajectory-up / contaminated
    partial = {"9", "13", "15", "26"}                 # killed / real-fail
    labels = [f"v{v}" for v in vers]
    colors = [CRIT if v in fail else (WARN if v in partial else GOOD) for v in vers]
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(vers))
    ax.barh(y, finals, color=colors, height=0.7)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    for i, f in enumerate(finals):
        ax.text(f + 0.0004, i, f"{f:.4f}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("final training loss")
    ax.set_title("F2. Final loss by version (verified from wandb summaries)\n"
                 "red=실기 실패/오염, yellow=중단/부분, green=성공 계열", fontsize=12, loc="left")
    leg = [Patch(color=CRIT, label="fail / contaminated"),
           Patch(color=WARN, label="killed / partial"),
           Patch(color=GOOD, label="success line")]
    ax.legend(handles=leg, frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "F2_final_loss.png")


# ============ F3: gripper norm_stats before/after patch ============
def f3_norm_stats():
    import glob as _g
    assets = os.path.join(REPO, "assets")
    # 버전 -> (gripper index, q01, q99) 실제 norm_stats.json 에서 로드
    want = {"v13": 6, "v15": 6, "v16": 6, "v17": 6, "v14": 7}
    rows = []
    for ver, gi in want.items():
        cands = _g.glob(os.path.join(assets, f"pi05_e6_{ver}_lora", "**", "norm_stats.json"), recursive=True)
        if not cands:
            continue
        d = json.load(open(cands[0]))
        act = d["norm_stats"]["actions"] if "norm_stats" in d else d["actions"]
        q01 = act["q01"][gi]; q99 = act["q99"][gi]
        rows.append((ver, gi, q01, q99))
    order = ["v13", "v15", "v14", "v16", "v17"]
    rows.sort(key=lambda r: order.index(r[0]) if r[0] in order else 99)
    labels = [f"{r[0]}\n(idx{r[1]})" for r in rows]
    q01s = [r[2] for r in rows]; q99s = [r[3] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    b1 = ax.bar(x - w/2, q01s, w, color=ORANGE, label="gripper q01")
    b2 = ax.bar(x + w/2, q99s, w, color=BLUE, label="gripper q99")
    ax.axhline(0, color="#c3c2b7", lw=1)
    for i, r in enumerate(rows):
        ax.text(i - w/2, r[2] - 0.12, f"{r[2]:.2f}", ha="center", fontsize=8, color=INK2)
        ax.text(i + w/2, r[3] + 0.04, f"{r[3]:.4f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("norm_stats value")
    ax.set_title("F3. Gripper norm_stats — delta(수동 패치) vs absolute(자연 정상)\n"
                 "v13/v15(delta): 수동 q01=-1,q99=1  ·  v16/v17(absolute): 자연 0/0.9998  ·  v14: 8D idx7 패치",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="x", visible=False)
    save(fig, "F3_norm_stats_gripper.png")


# ============ F4: action semantics (delta proof) ============
def f4_action_semantics():
    names = v16["action_semantics"]["joint_names"][:6]
    rd = v16["action_semantics"]["rmse_action_vs_delta"][:6]
    rn = v16["action_semantics"]["rmse_action_vs_nextpos"][:6]
    x = np.arange(len(names)); w = 0.38
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [2.4, 1]})
    floor = 0.85
    ax.bar(x - w/2, np.maximum(rd, floor), w, color=BLUE, label="RMSE vs delta (state[t+1]-state[t])")
    ax.bar(x + w/2, rn, w, color=ORANGE, label="RMSE vs next-position (state[t+1])")
    ax.set_yscale("log")
    ax.set_ylim(0.7, 320)
    plain_log_yaxis(ax)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("RMSE (deg), log")
    for i, v in enumerate(rd):
        ax.text(i - w/2, floor * 1.15, f"{v:.3f}", ha="center", fontsize=8, color="white",
                fontweight="bold")
    ax.set_title("F4a. Joint action = velocity DELTA\nRMSE vs delta ≈ 0 ; vs next-position = 28–178°",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    # gripper panel: value distribution {0,1}
    g = v16["gripper"]
    ax2.bar([0, 1], [g["frac_zero"], g["frac_one"]], color=[AQUA, MAGENTA], width=0.6)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["0.0\n(open)", "1.0\n(close)"])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("fraction of frames")
    ax2.set_title(f"F4b. Gripper = ABSOLUTE\nonly {{0,1}}; RMSE vs next={g['rmse_vs_next_absolute']:.3f}",
                  fontsize=11, loc="left")
    ax2.grid(axis="x", visible=False)
    save(fig, "F4_action_semantics.png")


# ============ F5: v16 per-joint distributions (clean evidence) ============
def f5_joint_dist():
    jd = v16["joint_distributions"]
    names = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    axes = axes.ravel()
    for i, name in enumerate(names):
        ax = axes[i]
        counts = np.array(jd[name]["hist_counts"]); edges = np.array(jd[name]["hist_edges"])
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, counts, width=(edges[1]-edges[0])*0.95, color=BLUE if i < 6 else MAGENTA)
        ax.set_title(f"{name}  μ={jd[name]['mean']:.1f} σ={jd[name]['std']:.1f}", fontsize=10, loc="left")
        ax.grid(axis="x", visible=False)
        ax.tick_params(labelsize=8)
    axes[7].axis("off")
    axes[7].text(0.05, 0.5,
                 "v16 state distributions\n(198 ep, cleaned)\n\n"
                 "각 관절이 single-sign,\nconsistent range →\nep272-399 식 부호반전\n"
                 "bimodal 오염 없음.\n(스파이크=표준 홈포즈)\n(contrast: F10)",
                 fontsize=10, color=INK2, va="center")
    fig.suptitle("F5. v16 per-joint state distributions — single-sign, no flip-contamination (from parquet)",
                 fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "F5_joint_distributions.png")


# ============ F6: phase distribution ============
def f6_phase():
    pf = v16["phase_frames"]
    # aggregate L/R into phase groups
    groups = {"approach": 0, "pick": 0, "lift": 0, "transport (move)": 0, "place": 0, "release": 0}
    for k, v in pf.items():
        kl = k.lower()
        if "approach" in kl: groups["approach"] += v
        elif "pick up" in kl: groups["pick"] += v
        elif "lift" in kl: groups["lift"] += v
        elif "move" in kl: groups["transport (move)"] += v
        elif "place" in kl: groups["place"] += v
        elif "release" in kl: groups["release"] += v
    total = sum(groups.values())
    order = ["approach", "pick", "lift", "transport (move)", "place", "release"]
    vals = [groups[k] for k in order]
    pct = [100*v/total for v in vals]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    y = np.arange(len(order))
    ax.barh(y, pct, color=BLUE, height=0.68)
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.invert_yaxis()
    for i, (p, v) in enumerate(zip(pct, vals)):
        ax.text(p + 0.4, i, f"{p:.1f}%  ({v})", va="center", fontsize=9, color=INK2)
    ax.set_xlabel("% of frames")
    ax.set_xlim(0, max(pct) * 1.25)
    ax.set_title("F6. v16 phase distribution (per-frame 6-phase prompt, task_index)\n"
                 f"total {total} frames across 198 episodes", fontsize=12, loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "F6_phase_distribution.png")


# ============ F7: lift trajectory (j2/j3 simultaneous drop) ============
def f7_lift():
    le = v16["lift_example"]
    fr = np.array(le["frames"]); j2 = np.array(le["j2"]); j3 = np.array(le["j3"])
    g = np.array(le["gripper"]); ci = le["close_idx"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    # shade gripper-closed region (one-axis rule: background, not 2nd axis)
    closed = g > 0.5
    ax.fill_between(fr, 0, 1, where=closed, transform=ax.get_xaxis_transform(),
                    color=AQUA, alpha=0.08, step="mid", label="_gripper closed")
    ax.plot(fr, j2, color=BLUE, lw=2, label="j2")
    ax.plot(fr, j3, color=ORANGE, lw=2, label="j3")
    ax.axvline(ci, color=CRIT, ls="--", lw=1.3)
    ax.text(ci + 1, ax.get_ylim()[1], "pick (gripper 0→1)", color=CRIT, fontsize=9, va="top")
    ax.set_xlabel("frame"); ax.set_ylabel("joint angle (deg)")
    ax.set_title("F7. Lift = j2 AND j3 decrease together (not j3 alone)\n"
                 f"{le['file']} — shaded = gripper closed; scripted-lift bug fixed via RelMovLUser (Z)",
                 fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    save(fig, "F7_lift_trajectory.png")


# ============ F8: v13 gripper release fix ============
def f8_gripper_fix():
    c = ds["v13"]["gripper_action_counts"]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    cats = ["close (+1)", "open (-1)", "no-op (0)"]
    before = [c["close_+1"], 0, c["noop_0"] + c["open_-1"]]  # pre-fix: open=0
    after = [c["close_+1"], c["open_-1"], c["noop_0"]]
    x = np.arange(3); w = 0.38
    ax.bar(x - w/2, before, w, color=MUTED, label="before fix (open=0 BUG)")
    ax.bar(x + w/2, after, w, color=BLUE, label="after fix (get_active_segments)")
    ax.set_yscale("log")
    ax.set_ylim(110, 90000)
    plain_log_yaxis(ax)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("action-frame count (log)")
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - w/2, max(b, 130)*1.08, str(b), ha="center", fontsize=8, color=INK2)
        ax.text(i + w/2, max(a, 130)*1.08, str(a), ha="center", fontsize=8, color=BLUE)
    ax.annotate("release events\n0 → 198", xy=(1+w/2, c["open_-1"]), xytext=(1.35, 900),
                fontsize=9, color=CRIT, ha="center",
                arrowprops=dict(arrowstyle="->", color=CRIT))
    ax.set_title("F8. v13 gripper-release fix (verified on converted v13 dataset)\n"
                 "close→open transition inside idle-run now captured (seg_end=k+2)",
                 fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    ax.grid(axis="x", visible=False)
    save(fig, "F8_gripper_release_fix.png")


# ============ F9: vision LoRA ablation coverage map ============
def f9_ablation():
    # (version, start, end, note, tested)  — from config.py (verified) + 실기 결과
    rows = [
        ("v17 (baseline)", 14, 25, "Mid+Late", "grounded"),
        ("v18", 0, 25, "all-26", "skip"),
        ("v19", 14, 18, "Mid-low", "skip"),
        ("v20", 19, 25, "Late-hi", "tested"),
        ("v21", 0, 8, "Early", "slow 115s/49chunk"),
        ("v22", 9, 17, "Mid", "tested"),
        ("v23  ★BEST", 18, 26, "Late-full", "62-67s/26-28chunk"),
        ("v24", 0, 26, "full-27", "tested"),
        ("v25", 15, 19, "Mid/Late", "tested"),
        ("v26", 22, 25, "Late-top", "tested(7.5k)"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (name, s, e, note, res) in enumerate(rows):
        best = "BEST" in name
        c = GREEN if best else (ORANGE if "slow" in res else BLUE)
        ax.barh(i, e - s + 1, left=s, height=0.62, color=c,
                edgecolor=INK if best else "none", linewidth=1.6 if best else 0)
        ax.text(e + 0.4, i, f"{note} · {res}", va="center", fontsize=8.5,
                color=INK if best else INK2, fontweight="bold" if best else "normal")
    # SigLIP taxonomy bands
    ax.axvspan(-0.5, 8.5, color=GRID, alpha=0.4)
    ax.axvspan(8.5, 17.5, color=GRID, alpha=0.2)
    ax.text(4, len(rows)-0.2, "Early 0-8", fontsize=8, color=MUTED, ha="center")
    ax.text(13, len(rows)-0.2, "Mid 9-17", fontsize=8, color=MUTED, ha="center")
    ax.text(22, len(rows)-0.2, "Late 18-26", fontsize=8, color=MUTED, ha="center")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, 34); ax.set_xlabel("SigLIP layer index (0–26)")
    ax.set_title("F9. Vision LoRA ablation — layer range per version (config-verified)\n"
                 "v23 = Late-full (18–26) fastest inference → adopted for E7. "
                 "실기 측정치는 v21/v23만 정량, 나머지는 정성적 '열세'.",
                 fontsize=11.5, loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "F9_vision_lora_ablation.png")


# ============ F10: contamination (memory-sourced, gap flagged) ============
def f10_contamination():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    # table data (from memory — NOT reproducible: raw v2/v6 400ep not cached)
    cols = ["", "ep 0–271 (68%)", "ep 272–399 (32%)"]
    data = [
        ["j1 mean", "+90°", "-65°  (flip)"],
        ["j4 mean", "-21°", "+22°  (flip)"],
        ["j5 mean", "-87°", "+90°  (flip)"],
        ["grasp j4 min", "-43°  (normal)", "-5°  (barely bent)"],
    ]
    tbl = ax.table(cellText=data, colLabels=cols, cellLoc="center", loc="center",
                   colColours=[SURFACE, "#eef4fc", "#fdeee6"])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.9)
    for (r, cc), cell in tbl.get_celldata_items() if hasattr(tbl, "get_celldata_items") else tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(color=INK, fontweight="bold")
    ax.set_title("F10. Data contamination ep 272–399 (v1–v7 primary failure cause)",
                 fontsize=13, loc="left", y=0.92)
    ax.text(0.5, 0.06,
            "math: 0.68 x (-43°) + 0.32 x (-5°) ~= -31°  ->  model converged to j4 ~= -12°\n"
            "[주의] MEMORY-SOURCED: raw v2/v6 400-ep collection NOT cached -> not reproducible here.\n"
            "Positive counter-evidence: cleaned v16 distributions are unimodal (see F5).",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=CRIT)
    save(fig, "F10_contamination.png")


if __name__ == "__main__":
    f1_loss_curves()
    f2_final_loss()
    f3_norm_stats()
    f4_action_semantics()
    f5_joint_dist()
    f6_phase()
    f7_lift()
    f8_gripper_fix()
    f9_ablation()
    f10_contamination()
    print("\nAll figures written to", FIG)
