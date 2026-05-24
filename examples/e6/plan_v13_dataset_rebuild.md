# E6 v13 데이터셋 재구축 계획

## 목표

DROID pretraining 포맷과 일치하도록 데이터셋을 재구축한다.
v10~v12의 핵심 문제인 **per-frame phase prompt**를 **episode-level prompt × 3변형**으로 교체한다.

---

## 현황 진단

### 원본 데이터 (새 볼륨2/Dobot/)

| 폴더 | 에피소드 수 | 방향 |
|------|------------|------|
| `2CAM-Orange/` | 553 | left → right |
| `2CAM-Orange-init/` | 199 | left → right |
| `2CAM-Orange-Opposite/` | 132 | right → left |
| `2CAM-Orange-Excluded/` | 4 | 제외 |
| **합계 (유효)** | **884** | 양방향 |

각 에피소드: `episode_meta.json` + `robot_data.csv` + `images/hik/` + `images/zed/`

### DROID 포맷 (pretraining 기준)

```
language_instruction   : "Put the marker in the pot"                          ← 짧게
language_instruction_2 : "Get the marker from the table and put it..."        ← 길게  
language_instruction_3 : "Put the marker inside the silver pot"               ← 중간
```

- episode-level (프레임마다 동일 문장 반복)
- 학습 시 3개 중 랜덤 1개 선택 → 언어 augmentation

### v10 문제 목록

| 심각도 | # | 문제 | 원인 |
|--------|---|------|------|
| 🔴 Critical | 1 | per-frame phase prompt | DROID pretraining과 distribution 불일치 |
| 🔴 Critical | 5 | gripper norm_stats 수동 패치 | 자동화 없음, 빠뜨리면 학습 망가짐 |
| 🔴 Critical | 6 | idle 제거 후 action_horizon 불연속 | 경계 프레임 인접, 16f 윈도우 오염 |
| 🔴 Critical | 7 | minority phase 과소 학습 | grasp+place+release 합계 18% |
| 🟠 품질 | 8 | image augmentation 없음 | 192ep 학습 → overfitting |
| 🟠 품질 | 9 | classify_phases() 경계 오류 | heuristic, 수동 검증 없음 |
| 🟠 품질 | 10 | source/target_zone 신뢰성 | 틀리면 prompt 전부 반대 |
| 🟠 품질 | 11 | release 없는 에피소드 edge case | 후반 gripper=0 → approach 오태깅 |
| 🔵 설계 | 12 | action_horizon=16 phase 경계 걸침 | 이질 행동이 하나의 윈도우에 |
| 🔵 설계 | 13 | instruction 다양성 없음 | 고정 문자열 1개 |
| 🔵 설계 | 14 | approach_oversample_factor=1.0 | oversampling 전무 |

**episode-level 전환으로 해결되는 항목:** 1, 7, 9, 10, 11, 12, 13 (7개)
**별도 코드 수정 필요:** 5, 6, 8, 14

---

## v13 설계 결정

### 1. Prompt: episode-level × 3변형 (DROID 동일)

```python
INSTRUCTIONS = {
    "left_to_right": [
        "pick up the orange box from the left side and place it on the right side",
        "move the orange box from the left to the right",
        "grasp the orange box on the left and put it down on the right",
    ],
    "right_to_left": [
        "pick up the orange box from the right side and place it on the left side",
        "move the orange box from the right to the left",
        "grasp the orange box on the right and put it down on the left",
    ],
}
```

- `episode_meta.json`의 `source_zone` 으로 방향 결정
- 학습 시 3개 중 랜덤 1개 선택 (PromptFromLeRobotTask 수정)
- **per-frame phase 분류 완전 제거** → classify_phases() 삭제

### 2. 카메라 슬롯

```
HIK → exterior_image_1_left  ✅ (기존 그대로)
ZED → exterior_image_2_left  ✅ (기존 그대로, 문제 없음)
```

### 3. gripper norm_stats 자동화

변환 스크립트 내에서 gripper(index 6) stats를 강제 고정:

```python
# norm_stats.json 생성 후 자동 패치
stats["action"]["q01"][6] = -1.0
stats["action"]["q99"][6] =  1.0
stats["state"]["q01"][6]  =  0.0
stats["state"]["q99"][6]  =  1.0
```

또는 compute_norm_stats.py 후처리 훅으로 자동 적용.

### 4. idle 경계 불연속 수정

idle 제거 후 경계 프레임에 **episode boundary 마킹** 추가:
- 제거된 idle 구간 앞/뒤를 별개 sub-episode로 분리
- 또는 idle 구간을 제거하지 않고 action=0 프레임으로 대체 (단순)
- **권장:** idle이 5f 이상이면 해당 에피소드를 앞/뒤 두 개로 분리하여 save_episode() 각각 호출

### 5. 데이터 범위

| 폴더 | 포함 여부 | 비고 |
|------|----------|------|
| `2CAM-Orange/` | ✅ | 553 → exclude 리스트 적용 |
| `2CAM-Orange-init/` | ✅ | 199 → exclude 적용 |
| `2CAM-Orange-Opposite/` | ✅ | 132 (right→left) |
| `2CAM-Orange-Excluded/` | ❌ | 완전 제외 |

예상 유효 에피소드: **~850**

### 6. HF repo

`Kyle-Riss/dobot_e6_pick_place_orange_v13`

---

## 구현 계획

### Step 1 — 변환 스크립트 작성 (`convert_e6_v13_to_lerobot.py`)

기반: `convert_e6_v10_to_lerobot.py`

**삭제:**
- `classify_phases()` 전체
- `PHASE_PROMPTS` dict
- `TRANSITION_WINDOW`, `Z_LIFT` 상수
- `phase_counts` 집계

**추가/수정:**
- `INSTRUCTIONS` dict (방향별 3변형)
- `source_zone` → instruction 선택 로직 (변환 시점은 1번 변형으로 저장, 학습 시 랜덤)
- tasks.jsonl에 6개 task 저장 (left_to_right 3개 + right_to_left 3개)
- task_index를 episode마다 방향 기반으로 할당
- idle 경계 → sub-episode 분리
- gripper norm_stats 자동 패치 함수

**멀티 root 지원:**
```python
--roots "/media/billy/새 볼륨2/Dobot/2CAM-Orange" \
         "/media/billy/새 볼륨2/Dobot/2CAM-Orange-init" \
         "/media/billy/새 볼륨2/Dobot/2CAM-Orange-Opposite"
```

### Step 2 — PromptFromLeRobotTask 수정

`src/openpi/training/data_loader.py`

현재: task_index → 고정 문자열 1개
수정: 같은 direction에 해당하는 task index 3개 중 랜덤 선택

```python
# tasks.jsonl 구조 (v13)
{"task_index": 0, "task": "pick up the orange box from the left side and place it on the right side"}
{"task_index": 1, "task": "move the orange box from the left to the right"}
{"task_index": 2, "task": "grasp the orange box on the left and put it down on the right"}
{"task_index": 3, "task": "pick up the orange box from the right side and place it on the left side"}
{"task_index": 4, "task": "move the orange box from the right to the left"}
{"task_index": 5, "task": "grasp the orange box on the right and put it down on the left"}
```

parquet에는 `task_index ∈ {0,1,2}` (left→right) 또는 `{3,4,5}` (right→left) 중 하나 저장.
학습 시 같은 direction 그룹(0~2 또는 3~5) 내에서 랜덤 선택.

> 단순하게: 변환 시 direction별로 task_index 하나만 저장하고(0 or 3),
> data_loader에서 {0→[0,1,2], 3→[3,4,5]} 매핑으로 랜덤 치환.

### Step 3 — config.py에 pi05_e6_v13_lora 추가

```python
# v12 대비 변경:
# - dataset: dobot_e6_pick_place_orange_v13
# - assets_dir: assets/pi05_e6_v13
# - 나머지 동일 (rank=16, scope 11~15, 30k steps)
```

### Step 4 — norm_stats 재계산 + gripper 자동 패치 확인

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_e6_v13_lora
# → assets/pi05_e6_v13/.../norm_stats.json
# gripper q01/q99 패치 자동 적용 확인
```

### Step 5 — 학습 실행

```bash
uv run scripts/train.py pi05_e6_v13_lora --exp-name e6_2cam_lora_v13
```

---

## 검증 체크리스트

변환 후 확인:

- [ ] `tasks.jsonl` 6개 항목 (left_to_right 3개 + right_to_left 3개)
- [ ] parquet `task_index` 분포: left→right 에피소드 = 0~2 중 하나, right→left = 3~5 중 하나
- [ ] 에피소드 총 수 ~850
- [ ] gripper norm_stats: `action.q01[6]=-1.0`, `action.q99[6]=1.0`
- [ ] idle 경계 sub-episode 분리 확인 (원래 1에피소드가 2개로 나뉘는지)
- [ ] 프레임당 단일 prompt (phase 변화 없음)

---

## 미해결 항목 (v13 이후)

| # | 항목 | 우선순위 | 비고 |
|---|------|---------|------|
| 8 | image augmentation | 🟠 | training config에서 color_jitter 추가 가능 |
| 14 | approach oversampling | 🟠 | episode-level 전환 후 의미 변경됨 (phase 없으므로 불필요) |

---

## 요약

```
v10~v12 문제: per-frame phase prompt (14개 이슈 중 7개의 근본 원인)
v13 핵심 변경: episode-level prompt × 3변형 (DROID 포맷 일치)
데이터 규모: 884 → ~850 에피소드 (3개 폴더 통합)
예상 효과: pretraining distribution 일치 → language conditioning 실제 작동
```
