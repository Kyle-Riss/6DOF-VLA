# 프로젝트: 6DOF-VLA (OpenPi E6)

> 이 파일은 Claude Code / Cursor / Codex 3개 도구가 모두 참조하는 **Source of Truth**.
> `AGENTS.md`는 이 파일의 symlink이며, `.cursor/rules/main.mdc`가 이 파일을 가리킴.

## Memory Policy (공유 계정 환경용)
- 이 프로젝트의 맥락은 **이 파일(`CLAUDE.md`)과 `.cursor/memory/` 폴더**만 신뢰.
- Claude Code의 글로벌 `Recalled memories`, Codex의 `~/.codex/memories`는 **다른 사용자 작업이 섞여 있을 수 있으므로 참고용으로만** 사용.
- 이 파일과 글로벌 memory가 충돌하면 **이 파일이 우선**.
- 개인 작업 진행 상황은 `.cursor/memory/progress.md`에만 기록 (gitignore 처리됨).
- 팀 공유가 필요한 규칙/결정은 이 파일 또는 `.cursor/memory/decisions.md`에 기록 후 Git commit.

## 개요
π₀.₅ (pi0.5) 모델을 Dobot MagicianE6 6-DOF 로봇에 파인튜닝하는 VLA 프로젝트.
Kyle-Riss/6DOF-VLA 기반, Physical Intelligence openpi 프레임워크 사용.

## 기술 스택
- **언어/프레임워크:** Python 3.11, JAX/Flax (학습), PyTorch (추론 변환)
- **패키지 관리:** `uv` (pip 직접 사용 금지, `uv pip` 또는 `uv run` 사용)
- **모델:** π₀.₅ (SigLIP + Gemma2b VLM frozen, action-expert LoRA만 학습)
- **로봇:** Dobot MagicianE6 (6-DOF), 관절 7개 (6 joint + 1 gripper)
- **카메라:** HIKRobot (top-view) + ZED (scene view), 224×224, 18fps
- **GPU:** NVIDIA RTX A5000 (25GB)

## 핵심 아키텍처
```
scripts/train.py          ← JAX 학습 진입점
src/openpi/training/config.py  ← 모든 실험 config 정의 (pi05_e6_v1_lora 등)
src/openpi/policies/e6_policy.py  ← E6 전용 policy
examples/e6/              ← 데이터 변환, task contract 등 E6 유틸
checkpoints/pi05_e6_v1_lora/e6_2cam_lora_v1/  ← 학습된 체크포인트
```

## 절대 규칙 (코드 수정 시 반드시 준수)

### 1. Action Semantics — 가장 중요 (v1/v2 공통)
E6 데이터셋의 action은 **절대 next-position (degree)** 이다. **delta가 아님.**
- `action[t]` = 다음 타임스텝의 관절 각도 (degrees)
- 추론 시 `current_joints + action`으로 계산하면 로봇이 날아감 → 즉시 `action`을 그대로 목표로 사용
- 관련 파일: `run_e6_client.py:612`, `executor_supervisor_node.py:210`

### 2. 실행 주기 (버전별 다름)
- **v1**: 학습 데이터 fps=18Hz → 추론 타이머 `1/18s` (≈0.0556s)
- **v2**: 학습 데이터 fps=16Hz → 추론 타이머 `1/16s` (=0.0625s)
- 추론 코드는 학습된 모델의 버전(config 이름)을 따라 분기 — v1/v2 모델 섞어서 호환 안 됨

### 3. v1/v2 데이터셋 격리
- **v1과 v2는 서로 다른 데이터셋**으로 취급. 학습 시 절대 합치지 않음.
- v1 자산(config / norm_stats / checkpoint) 은 v2 학습에 재사용 금지.
- v2 자산은 `assets/pi05_e6_v2/`, 체크포인트는 `checkpoints/pi05_e6_v2_lora/` 별도 관리.

### 4. Safety
모든 로봇 제어 로직에는 관절 범위 체크 포함. 절대값 기준 확인 후 명령 전송.

## 데이터

### v1 (기존)
- **위치:** `/data/Pick-place-2CAM/` (LeRobot 포맷, 100+ 에피소드)
- **HuggingFace:** `billy/dobot_e6_pick_place_random_v1` (101 에피소드)
- **fps:** 18Hz
- **이미지:** 캡처 원본 → 변환 시 224×224 resize
- **태스크:** 빨간 물체 집기/놓기 (left/right/middle, approach/pick/move/place) — **segment-level prompt**
- **state/action 차원:** 7D `[j1..j6, gripper_tooldo1]`
- **학습 config:** `pi05_e6_v1_lora`

### v2 (신규, 2026-04-25 수집 완료)
- **로컬 위치:** `/media/billy/새 볼륨/Dobot/2CAM-Orange/{N}/` (raw csv/jpg)
- **HuggingFace:** `kyle-riss/dobot_e6_pick_place_orange_v2` (400 에피소드, ep 382 제외)
- **fps:** 16Hz (실측 16.05Hz)
- **이미지:** 수집 시 이미 224×224로 저장됨 (resize 단계 불필요)
- **태스크:** 오렌지 박스 left↔right pick & place — **episode-level 단일 prompt**
  - A→B: `"pick up the orange box from the left side and place it on the right side"`
  - B→A: `"pick up the orange box from the right side and place it on the left side"`
  - 코드 `_zone = {"A":"left","B":"right"}` 매핑으로 저장 시점부터 left/right 표기
- **state/action 차원:** 7D `[j1..j6, gripper_tooldo1]` (v1과 동일 contract, vacuum binary 0/1)
- **에피소드 평균 길이:** ~324 frames
- **학습 config:** `pi05_e6_v2_lora` / 실험명 `e6_2cam_lora_v2`
- **변환 스크립트:** `examples/e6/convert_e6_v2_to_lerobot.py`

## 학습 버전 상태 (v1 / v2 / v3 / v4)

| 버전 | config | 상태 | 변수 가설 | 결과 / 다음 |
|---|---|---|---|---|
| v1 | `pi05_e6_v1_lora` (config.py:981) | ✅ 완료 (32k step, 2026-04-19) | baseline | wandb `o5tlyes1`, PyTorch 변환 완료 |
| v2 | `pi05_e6_v2_lora` (config.py:1015) | ✅ 학습 완료 (80k step, 2026-04-28 새벽) — **실기 성능 부족** | 데이터 4배 + episode-level prompt | wandb `46dt0sel` |
| **v3** | `pi05_e6_v3_lora` (config.py 추가됨) | 🟢 **코드 완료, 학습 대기** | **Vision tower LoRA last 5 (layers 23-27, 0-indexed 22-26)** | "phase 감지 실패가 vision frozen 때문" 가설 검증 |
| **v4** | `pi05_e6_v4_lora` (예정) | 🟡 코드 작업 대기 | **Vision 23-27 + Action expert 12-16 동시 LoRA (combined)** | "vision + geometry 둘 다 풀어줘야 함" 가설 검증 (v3 결과 보고 진행) |

### v2 실기 평가 요약 (2026-04-28)
- **증상:** 추론 시 `j4`(wrist pitch)가 충분히 안 꺾여서 박스에 손이 안 닿음 (z 120 이하 안 내려감).
- **데이터 진단:** 학습 데이터에서 grip 직전 `j4` 평균 변화량 17.8°(joint 중 1위), 최종 `j4≈-32°`. 추론 시 모델은 `j4≈-12°`에서 멈춤.
- **재진단 (이전 가설 정정):**
  - ~~base pi05 wrist-neutral prior~~ — pi05는 Dobot E6 j4를 모름. 의인화된 비유였음.
  - ~~LoRA rank=32 capacity 부족~~ — v1은 같은 rank 32로 작동했으므로 단독 원인은 아님.
  - ✅ **진짜 후보 1**: vision frozen → SigLIP이 "approach 시점 visual cue" 추출 못 함
  - ✅ **진짜 후보 2**: 18 layer 전체 LoRA가 geometry/smoothness prior 손상 → 평균값 회귀
  - ✅ **진짜 후보 3**: episode-level prompt만 → phase 신호 약함 (v1은 segment-level)
  - ✅ **진짜 후보 4**: approach phase 학습 frame 11.7%만 → conditional 학습 신호 부족
- **Loss:** 10k step 이후 plateau. step/capacity 늘려도 효과 한정.

### v3 — Vision tower LoRA (코드 완료, 학습 대기)
- **이름:** `pi05_e6_v3_lora` / 실험명 `e6_2cam_lora_v3`
- **데이터:** v2 그대로 (`kyle-riss/dobot_e6_pick_place_orange_v2`)
- **모델 변경 (구현됨):**
  - Action expert: **v2와 동일** (전체 18 layer rank 32, 변화 없음)
  - **Vision tower (PaliGemma SigLIP-So400m/14, depth 27)**: layer-mask 기반 parallel-residual LoRA, **0-indexed (22, 26)** 범위 (= 1-indexed 23-27, last 5).
    - `attn_lora_a/b`, `mlp_lora_a/b` 각 block에 추가 (rank=16, alpha=16, scaling=1.0)
    - 27 layer 전부 LoRA 파라미터 할당됨 (`nn.scan` 구조상). layer mask로 22-26만 활성, 나머지 22개 layer는 mask=0 → forward 기여 0 → gradient 0 → 사실상 frozen.
  - 그 외 vision 0-21, 모든 Gemma 2B (LLM) 부분: frozen 그대로
- **구현 파일 (이번 작업으로 변경됨):**
  - `src/openpi/models/siglip.py` — `Encoder1DBlock`, `Encoder`, `_Module`에 optional `lora_config` + `lora_layer_range` 추가. `_lora_residual()` 헬퍼로 parallel residual 구현.
  - `src/openpi/models/pi0_config.py` — `Pi0Config`에 `vision_lora_rank`, `vision_lora_alpha`, `vision_lora_layer_range` 필드 추가. `freeze_filter_v3_vision_late_lora()` 헬퍼 추가.
  - `src/openpi/models/pi0.py` — `vision_lora_rank`이 set이면 `LoRAConfig` 생성하여 SigLIP `_siglip.Module`에 forward.
  - `src/openpi/training/config.py` — `pi05_e6_v3_lora` TrainConfig 추가.
- **검증 완료 (CPU dry-run, 2026-04-28):**
  - 모델 인스턴스화 성공 (lazy_init OK)
  - 파라미터 확인:
    - `PaliGemma/img/Transformer/encoderblock/attn_lora_a` shape `(27, 1152, 16)` ✅
    - `PaliGemma/img/Transformer/encoderblock/attn_lora_b` shape `(27, 16, 1152)` ✅
    - `mlp_lora_a/b` 동일 패턴 ✅
    - 기존 action expert LoRA 그대로 (`PaliGemma/llm/layers/.../lora_a/b`, 18 layer rank 32) ✅
  - freeze_filter regex 정상 (img-base / llm-base 동결, img-lora / expert-lora trainable)
- **스텝/스케줄:** `num_train_steps=30_000`, `CosineDecaySchedule(decay_steps=30_000)`
- **save/keep:** `save_interval=2000`, `keep_period=10_000` (10k/20k/30k 영구 보존)
- **freeze_filter:** `freeze_filter_v3_vision_late_lora()` (v2와 다름! v2 filter는 img 전체 동결이라 vision LoRA가 학습 안 됨)
- **assets:** `assets/pi05_e6_v3/` (v2 재사용 금지)
- **가설 검증 기준:**
  - j4 -25° 이하 도달 (v2는 -12에서 멈춤)
  - 학습 image trajectory 재현률 17% → 40%+
  - 이게 잘 됨 → "vision frozen이 진짜 병목" 확정 → v4 안 가도 됨

### v4 계획 — Vision 23-27 + Action 12-16 동시 LoRA (combined, 사용자 확정)
> ⚠️ **사용자 지시 (2026-04-28):** "v4도 이다음단계로 짜둬, 무조건 23-27 + 12-16. 코드는 짜지 말고 md만 정리." — 즉 **vision late + action middle-late를 동시에 푸는 결합 실험**.

- **이름:** `pi05_e6_v4_lora` / 실험명 `e6_2cam_lora_v4`
- **데이터:** v2 그대로 (`kyle-riss/dobot_e6_pick_place_orange_v2`)
- **모델 변경 (계획):**
  - **Vision tower (SigLIP, depth 27)**: layer 23-27 (0-indexed 22-26) parallel-residual LoRA. **v3과 동일 설정** (rank=16, alpha=16, scaling=1.0). 이미 코드 인프라 구축 완료 (siglip.py 수정분).
  - **Action expert (Gemma 300m, depth 18)**: **layers 12-16 (0-indexed 11-15)** 만 LoRA 적용. rank=32, alpha=32 (v2 default). layer 0-10, 16-17은 base weight frozen, LoRA 미적용.
  - Gemma 2B (LLM): frozen 그대로
- **상태:** 🟡 **코드 작업 대기 (구현 X)**
- **이게 v3 단독과 다른 점:**
  - v3: vision만 풀고 action은 v2와 완전 동일 → "vision frozen이 진짜 원인인가?" 단독 검증
  - v4: vision + action 둘 다 부분 unfreeze → "둘 다 풀어야 v2 문제 해결되나?" combined 검증
  - v3 결과가 부족할 때만 진행 (v3로 충분히 좋아지면 v4 skip)

#### v4 코드 작업 항목 (아직 미구현, 계획만)
1. **`src/openpi/models/gemma.py`** `Config`에 `lora_layer_range: tuple[int, int] | None` 필드 추가.
2. **`gemma.py`의 `Module` (action expert)**: siglip.py에서 만든 `_lora_residual` + layer-mask 패턴 그대로 이식. 즉 18개 layer 모두 LoRA 파라미터는 할당되지만 mask로 11-15만 활성, 나머지 13개 layer mask=0 → forward 기여 0 → gradient 0.
   - 단, 현재 gemma_300m_lora variant는 이미 nn.scan 안에서 LoRA가 모든 layer에 흐르는 구조 → siglip.py처럼 mask 추가만 하면 됨.
3. **`src/openpi/models/pi0_config.py`**:
   - `Pi0Config`에 `action_expert_lora_layer_range: tuple[int, int] | None = None` 필드 추가.
   - **`freeze_filter_v4_combined_lora()`** 헬퍼 추가 — v3 freeze_filter (vision img-base 동결, img-lora trainable, llm-base 동결) **+ action expert LoRA trainable** 조합. (사실상 v3 freeze_filter와 거의 같음. 단지 action expert의 LoRA가 일부 layer에만 있다는 점만 다름.)
4. **`src/openpi/models/pi0.py`**: `Pi0Config.action_expert_lora_layer_range`를 gemma.Module로 forward.
5. **`src/openpi/training/config.py`**: `pi05_e6_v4_lora` TrainConfig 추가.
   - `vision_lora_rank=16, vision_lora_alpha=16.0, vision_lora_layer_range=(22, 26)` (v3과 동일)
   - `action_expert_variant="gemma_300m_lora"` + 새 필드 `action_expert_lora_layer_range=(11, 15)` (0-indexed)
   - `freeze_filter=pi0_config.freeze_filter_v4_combined_lora()`
   - `assets_dir="assets/pi05_e6_v4"`
   - 그 외 (스텝, batch, lr, save_interval) v3과 동일.

#### v4 학습 시작 절차 (코드 완성 후 사용 예정)
```bash
cd /home/billy/26kp/openpi_upstream_clean
uv run scripts/compute_norm_stats.py --config-name pi05_e6_v4_lora
uv run scripts/train.py pi05_e6_v4_lora --exp-name e6_2cam_lora_v4
```

- **예상 코드 작업 시간:** 2~3h (v3에서 만든 mask 인프라 그대로 재사용)
- **예상 학습 시간:** ~14h (v3과 비슷, 학습 가능 파라미터 수가 미세하게 적음)
- **가설 검증 기준:**
  - j4 -25° 이하 도달 (v2 미달 / v3 결과와 비교)
  - mode flip (j2 OOD) 발생률 v2 대비 감소
  - smoothness 향상 (jitter 감소)

### v3 / v4 실험 설계 (사용자 확정 버전)
- v3는 **단일 변수** (vision late LoRA만), v4는 **결합** (vision late + action middle-late).
- v3 → v4 순서로 진행. v3 결과에 따라 v4 진행 여부 결정.
- 결과 매트릭스:
  - v3 ✅ 충분 → v4 진행 안 함, v3로 종결
  - v3 ⚠️ 부분 개선 → v4 진행 (action도 같이 풀어줘야 하는지 검증)
  - v3 ❌ 효과 없음 → v4 진행 + 동시에 prompt(v5) / 데이터 가설(v6) 병행 검토

### v3 학습 시작 절차 (코드 완료된 상태에서 그대로 따라가기)
```bash
cd /home/billy/26kp/openpi_upstream_clean
# 1. norm_stats 계산 (v2 데이터 그대로지만 assets 디렉토리는 별도 v3로)
uv run scripts/compute_norm_stats.py --config-name pi05_e6_v3_lora
# → assets/pi05_e6_v3/kyle-riss/dobot_e6_pick_place_orange_v2/norm_stats.json 생성됨

# 2. 학습 실행
uv run scripts/train.py pi05_e6_v3_lora --exp-name e6_2cam_lora_v3
# → checkpoints/pi05_e6_v3_lora/e6_2cam_lora_v3/{2000,4000,...,30000}
# → keep_period=10k이므로 영구 보존: 10000, 20000, 29999
```

### v3 / v4 모두 부족 시 escalation
- **v5:** segment-level prompt 복원 (v1 방식, 가설 3 검증) — 데이터 재라벨링만
- **v6:** approach oversampling 3x + j4 loss weight 3x (가설 4) — dataloader/loss 수정
- **v7:** v4 + v5 결합 (vision LoRA + action partial LoRA + segment prompt) — 종합 솔루션
- **v8 이후:** 데이터 재수집 (carefully, 기존 v2와 mix)

## 변경 사항 발생 시
항상 `.cursor/memory/progress.md`를 업데이트하거나 업데이트를 제안할 것.
