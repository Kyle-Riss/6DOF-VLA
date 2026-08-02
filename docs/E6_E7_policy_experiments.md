# E6 → E7 VLA 정책 실험 검증 리포트

> **검증판** · 작성 2026-07-24 · 대상: pi0.5(pi05) 기반 로봇팔 pick&place 정책 (LoRA fine-tuning)
>
> 이 문서의 모든 수치는 다음 셋 중 하나로 **라벨**되어 있습니다:
> **[코드검증]** 실제 `config.py`/policy/norm_stats 파일에서 확인 · **[데이터검증]** 실제 캐시 데이터셋/parquet/wandb 로그에서 계산 · **[메모리]** 실험 노트 기반, 현 환경에서 재현 불가(사유 명시).
>
> 재현 스크립트: [`scripts/analysis_e6_e7/`](../scripts/analysis_e6_e7) · 그림 원본: [`docs/figures/e6_e7_policy/`](figures/e6_e7_policy)
>
> ⚠️ repo `CLAUDE.md`의 "action = 절대 next-position" 규칙은 **v1~v5 전용**이며 v6부터 무효(§2에서 정정). SoT 충돌 시 이 문서의 검증 결과 우선.

---

## 0. 요약 (TL;DR)

- **26개 버전(v1~v26)** 학습 완료 로그를 wandb에서 전수 확보. loss는 모두 ~0.01로 수렴 → **loss는 성패의 판별자가 아니었음**. 실패는 전부 (a) 데이터 오염 (b) action contract (c) 추론 파이프라인 문제였음.
- **확정된 action contract** (v16+, 실데이터로 증명): joint = **velocity delta**, gripper = **absolute {0,1}**.
- **Vision LoRA ablation 최종 승자: v23 (SigLIP 18~26 = Late 전체)** → E7 채택.
- **E7 = xArm6 (6축, 7D)** 로 전환 중. `e7_policy.py`가 현재 8D native로 작성돼 있어 **7D+align_droid_state로 재작성 필요**.

---

## 1. 아키텍처

```
이미지(HIK top-view + ZED side-view) → SigLIP (27-layer) → visual tokens
언어(phase prompt)                    → Gemma 2B (frozen)  → language tokens
                                                    ↓ prefix (VLM features)
state [j1~j6, grip] 7D→32D padded → Action Expert(Gemma 300m) LoRA → action
                                                    ↓ 학습 타깃
```

**[코드검증]** `pi0.py`: `vision_lora_rank`가 설정되면 `LoRAConfig`를 만들어 SigLIP에 전달, `vision_lora_layer_range`로 layer masking (`siglip.py:_layer_mask`, 범위 밖 layer는 mask=0 → gradient 0). action expert scope는 `action_expert_lora_layer_range`로 별도 전달.

**E6는 pretraining OOD 로봇** — pi05는 Franka/UR/ALOHA만 봄. SigLIP이 E6 카메라 뷰를 본 적 없고 action expert가 E6 kinematics를 처음부터 학습 → **데이터 품질이 전부**.

---

## 2. Action Contract 변천 (★ 최다 사고 지점)

### 2.1 Joint action — v6부터 delta

| 버전 | joint action 의미 | 추론 적용 | 근거 |
|---|---|---|---|
| v1~v5 | 절대 next-position (deg) | `target = action` | [메모리] RMSE 실험 |
| **v6~** | **velocity delta** = `state[t+1]-state[t]` | `target = current + action` | **[데이터검증] ↓** |

**[데이터검증]** v16 데이터셋(198ep, 42,495f)에서 프레임쌍 전수 계산:

![F4 action semantics](figures/e6_e7_policy/F4_action_semantics.png)

- joint j1~j6: **RMSE(action vs delta) = 0.0000** (정확히 0) / RMSE(action vs next-position) = 27.9~178.4°.
  → 저장된 joint action은 delta가 확정. (변환 시 delta로 구성됐음을 실증; 모델 주장 아님.)

### 2.2 Gripper action — v16부터 absolute

| 버전 | gripper action | norm_stats | 추론 |
|---|---|---|---|
| v6~v15 | **delta** Δgrip ∈ {-1,0,+1} | **수동 패치 필수** (q01=q99=0 버그) | 누산 + threshold |
| **v16~** | **absolute** grip[t+1] ∈ {0,1} | **패치 불필요** (자연 0/0.9998) | `ON if a[6]>0.5` |

**[데이터검증]** v16 `action[6]`: min=0.0, max=1.0, 값은 {0,1}만 (frac0=0.307, frac1=0.693), **RMSE(vs next-absolute)=0.0000**, RMSE(vs delta)=0.834 → absolute 확정.

**[코드검증]** norm_stats 실측:

![F3 norm_stats](figures/e6_e7_policy/F3_norm_stats_gripper.png)

- v13/v15 (delta): gripper q01=**-1.0**, q99=**1.0** (수동 패치). v16/v17 (absolute): q01=**0.0**, q99=**0.9998** (자연 정상, 패치 불필요). v14: 8D idx7에 -1/1 패치.
- **버그 원리**: gripper delta가 sparse(대부분 0) → quantile q01=q99=0 → 정규화 분모 1e-6에 delta±1이 곱해져 loss 폭발. `compute_norm_stats` 재실행해도 동일 → 수동 패치가 정답.

### 2.3 State 차원 — align_droid_state (v14)

**[코드검증]** `e6_policy.py`: `align_droid_state=True`이면 `np.insert(state, 6, 0.0)` → 8D `[j1..j6, 0, gripper]`, gripper가 index 7로 이동(pi05_base DROID 포맷 정렬). `E6Outputs`는 dummy j7 제거 후 7D 반환. **v16/v17/v23(최우수)는 `align_droid_state=False` (7D native)**.

> ⚠️ **[정정]** repo `CLAUDE.md`의 "action은 절대 next-position, `current+action`하면 로봇이 날아감" 규칙은 **v1~v5에만** 해당. v6+ 체크포인트는 정반대로 **적분(`current+action`)이 필수**. CLAUDE.md는 ~v8 시점에서 갱신이 멈춰 stale.

---

## 3. 전체 실험 매트릭스 (config + wandb 전수 검증)

**[데이터검증]** wandb 로컬 로그 50개 run에서 버전별 canonical run과 최종 loss를 전수 추출:

![F1 loss curves](figures/e6_e7_policy/F1_loss_curves.png)

![F2 final loss](figures/e6_e7_policy/F2_final_loss.png)

> loss는 v7(0.0301, 오염+kill)·v6(0.0193)·v9(0.0172)가 높고 v10/v18/v24가 낮지만, **모두 실기 성패와 무관**. loss 곡선은 ~25k에서 plateau. (F2 색상은 거친 결과 분류일 뿐 — final loss는 성공 지표가 아님.)

| 버전 | wandb run | final step | final loss | vision LoRA | action scope | 비고 |
|---|---|---|---|---|---|---|
| v1 | o5tlyes1 | 31,950 | 0.0074 | none | 18L r32 | [메모리] 실기 위로 뜸 |
| v2 | 46dt0sel | 79,950 | 0.0118 | none | 18L r32 | 80k=5ep도 j4 못 풀음 |
| v3 | tc9buorn | 29,950 | 0.0111 | (22,26) | 18L r32 | vision late 단독 |
| v4 | 1v6ufvih | 21,700 | 0.0143 | (22,26) | (11,15) r32 | combined, 20k kill |
| v5 | 5f7jjoze | 8,400 | 0.0166 | (22,26) | (15,17) r16 | 8k kill |
| v6 | 1dlqvaex | 46,100 | 0.0193 | none | 18L r32 | 오염 발견 → 중단 |
| v7 | 6ydhdsov | 30,000 | 0.0301 | none | 18L r32 | approach 3x, 오염 kill |
| v8 | 9ll9vruz | 49,950 | 0.0157 | (22,26) | 18L r32 | 오염 제거 재수집(549ep) |
| v9 | ay3xyhz0 | 49,950 | 0.0172 | (22,26) | (11,15) r32 | v8 vs scope 비교 |
| v10 | lt4wv2wo | 49,950 | 0.0101 | (22,26) | 18L r32 | 이중 홈포즈(192ep) |
| v11 | 1xp520f2 | 29,950 | 0.0150 | (22,26) | (11,15) r16 | rank·scope 보수적 |
| v12 | 4por27ew | 29,950 | 0.0130 | (22,26) | 18L r16 | rank만 축소 |
| v13 | ha35lvbe | 29,950 | 0.0137 | (22,26) | (11,15) r16 | sub-ep, **실기 실패** |
| v14 | ue4w3kfs | 23,000 | 0.0115 | (22,26) | 18L r16 | full-ep, 8D align |
| v15 | 6isoxpk3 | 5,850 | 0.0133 | (22,26) | 18L r16 | 5k kill (v16 우선) |
| **v16** | 7hfgpik5 | 19,950 | 0.0109 | (22,26) | 18L r16 | **absolute gripper 도입** |
| **v17** | qg9easzr | 29,950 | 0.0113 | **(14,25)** | 18L r16 | **★ ablation 기준선** |
| v18 | 9wz4xw5y | 19,950 | 0.0101 | (0,25) | 18L r16 | upper bound |
| v19 | bb2xvh7q | 19,950 | 0.0110 | (14,18) | 18L r16 | Mid 하위 |
| v20 | gtffmt4l | 17,500 | 0.0112 | (19,25) | 18L r16 | Late(26제외) |
| v21 | ideftmxl | 19,950 | 0.0106 | (0,8) | 18L r16 | Early, 실기 느림 |
| v22 | w87ynfk1 | 19,950 | 0.0105 | (9,17) | 18L r16 | Mid 전체 |
| **v23** | 2btcwot3 | 19,950 | 0.0110 | **(18,26)** | 18L r16 | **★ 최우수** |
| v24 | bnjb9w48 | 19,950 | 0.0101 | (0,26) | 18L r16 | full-27 |
| v25 | 7h70cctk | 19,950 | 0.0111 | (15,19) | 18L r16 | Mid/Late 경계 |
| v26 | q0v381ru | 8,850 | 0.0157 | (22,25) | 18L r16 | 7.5k 중단 |

*(vision/action scope 열은 `config.py` [코드검증]. 실기 결과는 §4·§5.)*

---

## 4. 근본 원인 발견들 (디버깅 히스토리)

### 4.1 데이터 오염 ep 272~399 — v1~v7 전멸의 주범 · [메모리]

![F10 contamination](figures/e6_e7_policy/F10_contamination.png)

**⚠️ [메모리]** 오염 버전(v2/v6 400ep 수집분)은 현재 캐시에 없어 **재현 불가**. 수학적 설명: `0.68×(-43°)+0.32×(-5°)≈-31°` → 모델 j4≈-12° 수렴을 정확히 설명.

**반증(긍정 증거) [데이터검증]**: 정제된 v16 분포는 모든 관절이 single-sign·일관 범위 → ep272~399식 부호반전 bimodal 오염 없음.

![F5 joint distributions](figures/e6_e7_policy/F5_joint_distributions.png)

*(스파이크는 표준 홈포즈. j1 전부 양수, j4 전부 음수, j5 전부 ~-88° → 부호반전 없음.)*

### 4.2 v13 gripper release 버그 — close→open 0건 · [데이터검증]

**[코드검증]** `convert_e6_v13_to_lerobot.py:get_active_segments` — idle run 내부의 gripper 전환 감지 시 `seg_end = k+2`로 전환쌍을 포함(fix).

**[데이터검증]** 변환된 v13 데이터셋 실측: close(+1)=198, **open(-1)=198**, no-op=43,971. 메모리상 fix 전 open=0 → fix 후 198 캡처 확인.

![F8 gripper release fix](figures/e6_e7_policy/F8_gripper_release_fix.png)

→ v13 실기 실패 3원인(놓기 학습 전무 / j3 궤적 방향 불일치 / gripper oscillation)이 **v14 재설계**(full episode + align_droid_state + 18L LoRA) 근거.

### 4.3 lift = j2·j3 동시 감소 · [데이터검증]

**[데이터검증]** v16 대표 에피소드(ep0, pick@frame62)에서 pick 후 j2·j3가 **함께** 변하며 z 상승 (j3 단독 아님):

![F7 lift trajectory](figures/e6_e7_policy/F7_lift_trajectory.png)

**[메모리→수정 방향 확정]** scripted lift는 joint-space(`j3-=0.5`) 대신 **`RelMovLUser(0,0,dz,0,0,0)`** (세계좌표 Z, IK가 j2/j3 동시 처리) 사용. non-blocking 큐 + `tcp_z>=target` flag, **lift 중 VLA MovJ 완전 스킵**.

### 4.4 scripted lift 후 하강 버그 체인 · [메모리]

lift 청크가 grip≈0.026 출력 → 흡착 해제 → `_released=True` 오염 → z>180mm 돌파 시 phase="return" → 하강. **수정 3가지**(`executor_supervisor_node.py`): ① lift stage에서 grip=1 강제 ② 완료 후 32f grip hold ③ 재진입 시 hold 리셋.

### 4.5 phase 분포 · [데이터검증]

**[데이터검증]** v16 per-frame 6-phase prompt(task_index) 분포:

![F6 phase distribution](figures/e6_e7_policy/F6_phase_distribution.png)

---

## 5. Vision LoRA Ablation (v18~v26) → v23 채택

**SigLIP taxonomy (27층):** Early 0~8(저수준) / Mid 9~17(구조) / Late 18~26(고수준+공간+텍스트정렬).

![F9 vision LoRA ablation](figures/e6_e7_policy/F9_vision_lora_ablation.png)

**[코드검증]** 각 버전의 layer range는 config.py에서 확인. **[메모리]** 실기 측정치는 v21/v23만 정량:

| 버전 | vision LoRA | 실기 (suction ON 소요 / 청크) | 판정 |
|---|---|---|---|
| v21 | (0,8) Early | 115.6s / 49청크 | ⚠️ 느림·망설임 |
| **v23** | **(18,26) Late 전체** | **62.6~67.4s / 26~28청크** | **✅ 최우수** |
| v20/v22/v24/v25/v26 | 각 범위 | (정성) | v23 대비 열세 |

> **[메모리] 최종 결론 (2026-06-11): v23 (18~26)이 최우수 → E7 v1 채택.** 26번 레이어(텍스트정렬 최종층) 포함이 v20(19~25) 대비 우세.

---

## 6. 추론 서버 계약 (v16/v17, ROS2)

- **state** 7D `[j1..j6, gripper]` (`align_droid_state=False`)
- **action[0:6]** delta → `joint_target = current + action[0:6]`
- **action[6]** absolute → `suction = ON if action[6] > 0.5`
- 16Hz, chunk=16, 앞 8개 실행. scripted lift: `grasp_z_max=130`, `RelMovLUser`로 z=185mm, grip hold 32f.

| phase | 프롬프트 |
|---|---|
| approach | "move the arm down to approach the orange box on the {side}" |
| grasp | "pick up the orange box on the {side}" |
| lift | "lift the orange box from the {side}" |
| transport | "lift and carry the orange box to the {target_side}" |
| place | "lower the orange box onto the {target_side}" |
| release | "release the orange box on the {target_side}" |
| return | "return the arm to the ready position" |

**노드 파라미터:** `executor_supervisor_node.py action_mode:=delta`, `task_node.py prompt_mode:=per_frame source_side:=left target_side:=right`.

---

## 7. E7 (xArm6) 전환 계획

> ⚠️ **[정정 2026-07-23]** 로봇은 **xArm6 (6축) + 그리퍼 = 7D**. 이전 "xArm7/8D" 기록은 오기.

**[코드검증]** `config.py:2131` `pi05_e7_v1_lora` 존재:
- vision_lora_layer_range=**(18,26)** (v23 채택), action_expert=`gemma_300m_lora_r16`, num_train_steps=20k, freeze_filter=`freeze_filter_v4_combined_lora()`.
- repo_id=`Kyle-Riss/xarm_e7_v1`는 **TODO placeholder**(주석 2곳), action semantics `TBD`.

| 항목 | 값 |
|---|---|
| 모델 | UFACTORY xArm6 (6-DOF), 7D `[j1..j6, gripper]` |
| 그리퍼 | 기계식 G2 평행. **연속 정규화 [0,1] aperture** (absolute; 0=open,1=close). force는 학습 차원 아님(meta만) |
| 카메라 | HIK + ZED, 224×224, **수집 15Hz(DROID 정렬)** |
| action | joint delta + gripper absolute (E6 v16+ 방식) |

### 해야 할 작업 (TODO)
- [ ] **⚠️ `e7_policy.py` 재작성** — **[코드검증] 현재 8D native로 작성됨** (`state=np.random.rand(8)`, align_droid_state 없음, 주석 "xArm7=8D 의도적"). 로봇이 실제로 xArm6=7D이므로 잘못된 가정 → `e6_policy.py`처럼 **7D + `align_droid_state=True`** 로 재작성 필요.
- [ ] 데이터 수집 후 repo_id/asset_id 교체 (config.py TODO 2곳).
- [ ] G2 연속 aperture 기록 파이프라인, Dobot→xArm API 교체.

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_e7_v1_lora
uv run scripts/train.py pi05_e7_v1_lora --exp-name e7_2cam_lora_v1
```

---

## 8. ⚠️ 검증 한계 (재현 불가·미확보 항목)

이 리포트가 **실데이터로 재현하지 못한** 것들 — 추후 보강 대상:

1. **오염 ep272~399 재현 불가** — v2/v6 400ep 원본 수집분이 HF 캐시에 없음. repo `data/2CAM-Orange-init`은 clean 199-ep(v13/v14/v16 소스)라 오염 없음. → F10은 [메모리] 수치, F5로 간접 반증.
2. **외장 체크포인트 백업 미마운트** — `/media/billy/새 볼륨2·3/e6_checkpoints/` 현재 접근 불가. 백업본 step 검증 불가. (로컬 `checkpoints/`에는 v14/v11/v1_lora merged 없음, v6/v7/v15 merged 없음.)
3. **E7 자산 부재** — `pi05_e7_*` 체크포인트/assets/`xarm_e7_*` 데이터셋 아직 없음(데이터 수집 전).
4. **실기 정량치는 v21/v23만** — 나머지 ablation은 정성적 "열세" 판정([메모리]).
5. **`j4 min>-20°` 품질검사 코드 부재** — 메모리의 이 기준은 실제 convert 스크립트에 없음. 실재하는 것은 v10의 `mode=9` IK-flip 제외(`exclude=(16,97,98,129,149,150,193)`), v13/v14는 `exclude=(193,)`뿐.

---

## 부록: 재현 방법

```bash
cd /home/billy/26kp/openpi_upstream_clean
uv run --no-sync python scripts/analysis_e6_e7/compute_wandb.py      # loss/wandb → data/wandb_curves.json
uv run --no-sync python scripts/analysis_e6_e7/compute_datasets.py   # action/분포 → data/datasets_analysis.json
uv run --no-sync python scripts/analysis_e6_e7/make_figures.py       # F1~F10 PNG
```

데이터 소스: `~/.cache/huggingface/lerobot/Kyle-Riss/dobot_e6_pick_place_orange_{v16,v13}`, `wandb/run-*/files/`, `assets/pi05_e6_v*_lora/*/norm_stats.json`, `src/openpi/training/config.py`, `src/openpi/policies/e{6,7}_policy.py`.
