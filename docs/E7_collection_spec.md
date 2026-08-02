# E7 (xArm 6) 데이터 수집 스펙 — Jetson 수집 노드용

> 작성 2026-07-27. 학습서버(`openpi_upstream_clean`) ↔ Jetson(`vla_data_collector_node.py`) 대조 후 확정본.
> E6 파이프라인을 실제 코드로 검증한 뒤 작성했으며, xArm6 실측 아키텍처(Cartesian velocity)를 반영해 초안에서 정정된 부분을 포함한다.

---

## 0. 한 장 요약

| 항목 | 값 | 비고 |
|---|---|---|
| 로봇 | UFACTORY **xArm 6** (6축 + 그리퍼) | xArm 7 아님 |
| state / action | **7D** `[j1..j6, gripper]` | E6와 동일 shape·semantics |
| 녹화 주기 | **16 Hz** | 이미 적용됨. E6 `fps=16`과 일치 = 비교 전제 |
| 명령 주기 | **20 Hz** (Cartesian velocity) | 32Hz 상향 검토 (§6) |
| 이미지 | HIK + ZED, **224×224 RGB JPEG** | 리사이즈 없이 그대로 사용 |
| 제어 인터페이스 | `/xarm/vc_set_cartesian_velocity` | ServoJ 아님. 라벨에는 영향 없음 (§1) |
| 클럭 | ROS2 system clock | `host_monotonic` 아님. 정직히 기록 |
| QP / MPC | **없음** | `control_log.csv`·`qp_*` 필드 전부 제외 |

**핵심**: 학습 라벨은 `Δq = q_measured[t+1] − q_measured[t]`이므로 **로봇을 무엇으로 명령했든 무관하게 유효**하다. Cartesian velocity로 텔레옵해도 joint-delta 계약은 그대로 성립한다.

---

## 1. ✅ 블로커 해소 — 관절공간 실행 경로 (2026-07-27 실측)

**Joint delta 7D 계약 확정.** 두 경로 모두 존재하고 정상 동작함이 실물에서 확인됐다.

| 경로 | 타입 | 인터페이스 |
|---|---|---|
| `/xarm/set_servo_angle_j` | `xarm_msgs/srv/MoveJoint` | ServoJ (mode 1). **절대 관절각** `angles[]` |
| `/xarm/vc_set_joint_velocity` | `xarm_msgs/msg/MoveVelocity` | joint velocity (mode 4). `speeds[]` + `duration`(자동정지) |

**실물 테스트**: J1에 +0.02 rad(≈1.15°) `set_servo_angle_j` 전송 → `/xarm/robot_states`에서 정확히 해당 값 도달 확인 → 원위치 복귀. err/warn 0.

**도달 rate** (지속 연결 클라이언트로 60회 반복, RTT 기준):

```
mean 1.94 ms   p95 3.28 ms   max 7.26 ms   → 이론 최대 ≈ 515 Hz
```

> CLI `ros2 service call`로 측정하면 프로세스 생성 오버헤드 때문에 0.6Hz라는 무의미한 값이 나온다. 반드시 지속 연결 클라이언트로 측정할 것.

515Hz 여유는 16Hz 정책 청크를 고주파로 보간해 집행하기에 충분하다. `Δq → 자코비안 → twist` 역변환 우려는 해소됐다.

### 1.1 추론 executor 선택 — ServoJ 권장

| | 명령 누락 시 거동 | Δq 매핑 | 판정 |
|---|---|---|---|
| **ServoJ (mode 1)** | 마지막 목표각에 **정지 유지** | `q_target = q + Δq` 직접 | ✅ **채택** |
| joint velocity (mode 4) | 마지막 속도로 **계속 주행**, `duration` 만료까지 | `v = Δq / 0.0625` | ❌ 배제 |
| MoveJ | **명령이 큐에 누적** — 실시간 스트리밍 부적합 | — | ❌ 배제 |

정책 추론이 지연되거나 멈췄을 때 ServoJ는 그 자리에 서고 mode 4는 계속 움직인다. MoveJ는 큐 누적 때문에 16Hz 연속 스트리밍 자체가 성립하지 않는다. **ServoJ 확정.**

단, ServoJ는 서보 모드 특성상 **일정한 고주파 스트림**을 전제한다. 16Hz(62.5ms 간격) 원본 그대로 쏘면 계단형 움직임이 되므로 반드시 보간층을 둔다:

```
π0.5 16Hz → 16스텝 청크(1.0초) → 보간(≥100Hz 권장) → set_servo_angle_j
```

### 1.2 `delta_reference` — 이제 실제 결정 사항

ServoJ는 절대각을 받으므로 실행기가 둘 중 하나를 골라야 한다. **executor 설정에 명시하고 파일럿에서 비교할 것.**

| | 장점 | 단점 |
|---|---|---|
| `q_measured + Δq` | 드리프트 자기보정 | 측정 노이즈 유입, 서보 지연과 상충 |
| `q_target_prev + Δq` | 부드러움 | 추종 오차만큼 **드리프트 누적** |

학습 라벨이 `q_measured[t+1] − q_measured[t]`이므로 `measured` 기준이 라벨과 자기일관적이다. 다만 E6의 미검증 "delta 누적" 가설이 정확히 이 지점이므로 실측으로 확인할 것.

**정책 없이 지금 바로 테스트 가능하다 — replay 방식:**
수집한 에피소드 하나의 `Δq` 시퀀스를 꺼내서, 두 적분 규칙 각각으로 ServoJ에 재생하고 최종 관절각을 기록된 궤적과 비교한다.

```
녹화된 Δq[0..N] → (a) q_measured + Δq  로 재생 → 최종 q vs 녹화 q 오차
                → (b) q_target_prev + Δq 로 재생 → 최종 q vs 녹화 q 오차
```

(b)의 오차가 N에 비례해 커지면 드리프트 누적이 확인되는 것이고, 그게 곧 E6 가설의 검증이다. **학습된 정책이 필요 없으므로 파일럿 단계에서 수행 가능**하다. (원래 스펙에는 "파일럿에서 비교"라고만 적었는데, 정책 rollout으로 비교하려면 3단계까지 가야 하므로 replay 방식이 맞다.)

---

## 2. 폴더 구조 (E6와 동일하게 유지)

```
<root>/<N>/                      # N = 1, 2, 3 ... 숫자만. 숫자 크기순 정렬됨
  robot_data.csv                 # 16Hz, 학습 기준 프레임
  episode_meta.json
  images/hik/frame_%06d.jpg      # 224×224 RGB
  images/zed/frame_%06d.jpg      # 224×224 RGB
```

**변환기가 실제로 여는 파일은 이 3종뿐이다.** (E6 변환기 `examples/e6/convert_e6_v16_to_lerobot.py` 기준 검증됨)
`dataset.npy`, `metadata.txt`, `episode_events.csv`는 사람이 보기 편하면 남겨도 되지만 변환기는 읽지 않는다. 그리퍼 이벤트도 CSV 열에서 다시 찾는다.

**주의**
- 폴더명은 **숫자만** (`1`, `2`, ...). 숫자 아닌 폴더는 무시된다.
- 이미지는 **반드시 224×224 RGB**. 변환기가 `resize=None`이라 원본 크기를 그대로 쓰고, 다르면 `info.json`의 feature shape가 어긋나 학습이 붙지 않는다.

---

## 3. `robot_data.csv` — 16 Hz

### 3.1 변환기가 직접 소비 (컬럼명 고정, 바꾸지 말 것)

```
frame_id
image_path_hik            # "hik/frame_000002.jpg" 형태 (images/ 기준 상대경로)
image_path_zed
j1, j2, j3, j4, j5, j6    # degree, 절대 관절 위치
gripper_command           # 명령값. E6의 gripper_tooldo1 자리
```

> **그리퍼는 반드시 명령값(commanded)이어야 한다.** E6의 `gripper_tooldo1`은 tool digital output = 명령이었다. 피드백값(`gripper_state`)을 쓰면 지연이 섞여 변환기의 phase 검출(`find_close_idx`/`find_open_idx`)이 밀리고, 6개 phase 경계가 전부 어긋난다.
>
> **값은 연속 [0,1] 원값으로 기록한다.** 이진화는 나중에 무손실이지만 역은 불가능하다. 이진/연속 계약 확정은 변환 스크립트에서 한다 (§7).

### 3.2 시간

```
timestamp                 # ROS2 system clock (기존 유지)
timestamp_monotonic       # ★ 신규. time.monotonic() — 지연 계산 전용
joint_state_timestamp
image_timestamp_hik       # ★ 센서 캡처 시각. 파일 저장 시각 아님
image_timestamp_zed       # ★
teleop_timestamp
command_send_timestamp
```

> ROS system clock은 NTP 보정으로 에피소드 중간에 점프할 수 있어 지연 계산이 오염된다. 카메라 staleness 체크에 이미 `time.monotonic()`을 쓰고 있으니 컬럼 하나만 추가하면 된다.
>
> 이미지 timestamp를 저장 직전 `time.time()`으로 찍으면 캡처 시각이 아니라 저장 지연을 기록하는 것이다. ZED SDK는 프레임 타임스탬프를 제공하고 HIK도 프레임 헤더에 있다.

### 3.3 명령 (twist 공간 — 사후 복원 불가, 반드시 수집 시점에)

```
teleop_twist_raw_vx, _vy, _vz, _wx, _wy, _wz     # 텔레옵 원본
twist_sent_vx, _vy, _vz, _wx, _wy, _wz           # 드라이버에 실제 전달
```

> QP가 없어도 **두 값을 모두 남긴다.** `vc_set_cartesian_velocity`가 자체 속도 한계로 클램핑할 가능성이 있고, 클램핑이 일어나면 `raw ≠ sent`다. 그 차이가 곧 공짜로 얻는 intervention 신호이므로 파일럿에서 실제로 갈리는지 확인할 것.
>
> QP가 없으므로 `command_filtered_*`는 두지 않는다. 나중에 QP를 도입하면 그때 추가한다.

### 3.4 명령/프레임 정합

```
n_commands_in_frame       # 그 프레임 구간에 발행된 twist 명령 수 (20/16 → 1 또는 2)
last_command_timestamp
```

### 3.5 드롭 검출

```
frame_dropped_before      # 이 프레임 직전 드롭 여부
dt_from_prev              # 실제 경과 시간
```

> `frame_id`는 순차 카운터라 카메라 드롭이 나도 구멍이 생기지 않는다. 그런데 action은 인접쌍 차분이라 **드롭된 자리의 Δq가 조용히 2배로 튄다.** 에피소드 단위 집계(`camera_sync.pair_drops`)만으로는 어느 프레임인지 알 수 없으므로 프레임 단위 플래그가 필요하다. 변환기 쪽에도 `dt` 이상치 검사를 추가할 예정이다.

### 3.6 상태 (전이 이벤트는 래치로)

```
gripper_state             # SDK 피드백. 없으면 빈 값
joint_velocity_1..6       # 실측인 경우만. 내부 차분값이면 정보가 0이므로 생략
xarm_state, xarm_mode
error_latched, error_code_first     # 구간 내 1회라도 발생 시 래치
warn_latched, warn_code_first
```

> `error_code`를 16Hz로 샘플링하면 62.5ms 안에 떴다 사라진 전이를 놓친다. 래치 방식이 필요하다.

---

## 4. `episode_meta.json`

### 4.1 변환기 필수

```json
"source_zone": "left",     // "left" | "right" 만 허용. 이외 값이면 에피소드 SKIP
"target_zone": "right"
```

### 4.2 제어 계약 (없으면 나중에 원인 추적 불가)

```json
"control_mode":       "cartesian_velocity",
"command_semantics":  "cartesian_twist",
"command_rate_hz":    20,
"record_rate_hz":     16,
"joint_unit":         "degree",
"gripper_encoding":   "continuous_aperture_0_1",
"clock_domain":       "ros_system_clock",
"driver_params":      { "...": "set_mode/set_state, 속도·가속도 한계, 보간 방식" }
```

> 나중에 드라이버 파라미터를 바꾸면 학습된 정책이 무효화되는데, 기록이 없으면 성능 저하 원인을 찾을 수 없다.
>
> `delta_reference`(`"measured"` vs `"commanded"`)는 **수집 메타가 아니라 추론 executor 설정**에 둔다. 수집 중에는 관절 타깃이 존재하지 않으므로 해당 없음. 다만 추론에서는 `q_measured + Δ`인지 `q_target_prev + Δ`인지가 누적 오차를 가르므로 반드시 명시해야 한다.

### 4.3 기존 유지 + 채워야 할 것

```
task_name, object_label, prompt, success, total_frames, date
pick_x, pick_y        (pick_anchor — 이미 있음)
place_x, place_y      ★ 현재 스텝만 있고 실제로 안 채워짐 → 채울 것
camera_sync           (pair_drops / pair_waits — 유지)
```

---

## 5. 제외 항목

| 항목 | 사유 |
|---|---|
| `control_log.csv` (고주파 별도 로그) | 명령 20Hz / 녹화 16Hz → 프레임당 1.25개. 2단 분리가 과설계. §3.3~3.4로 충분 |
| `qp_enabled`, `qp_status`, `qp_solve_time_ms`, `qp_intervention_norm`, `qp_active_frac`, `deadline_miss_count` | xArm6에 QP/MPC 자체가 없음. 도입 시 추가 |
| `teleop_raw_j1..j6`, `command_filtered_j1..j6`, `command_sent_j1..j6` | 관절공간 3단 필드는 아키텍처 불일치. §3.3의 twist 필드로 대체 |
| `actual_delta` | `q[t+1] − q[t]`로 변환 시 계산 가능. 원본에 넣지 않음 |

---

## 6. 20 Hz × 16 Hz 맥놀이

명령 20Hz / 녹화 16Hz = 5:4라 프레임당 명령 수가 `1,1,1,2`로 반복되고 명령과 프레임의 위상이 계속 어긋난다.

- **권장**: twist 발행을 **32 Hz**(정확히 2:1)로 상향 (`collection_control.yaml: publish_rate_hz`)
- 16의 정수배면 무엇이든 정합한다. 경로가 감당하면 **48Hz(3:1)**가 더 부드럽다. 최소 요건은 32Hz
- 드라이버가 안 받으면 그대로 두고 `n_commands_in_frame`으로 기록만 남긴다

> ⚠️ **515Hz는 `set_servo_angle_j`(관절 서비스) 측정치다.** 텔레옵은 `vc_set_cartesian_velocity`라는 **다른 경로**를 쓴다. 32Hz 상향 전에 그 경로가 실제로 32Hz를 지속 유지하는지 별도로 확인할 것 — 감당 못 하면 불규칙한 발행이 되어 깨끗한 20Hz보다 나쁘다. 수 초간 실제 발행 간격의 p95/max를 재보면 된다.
- **녹화를 20Hz로 올리는 것은 권장하지 않는다** — E6와의 `fps=16` 일치가 비교의 전제이고 그게 더 비싸다

### ⚠️ 적용 순서 주의

**32Hz 변경은 파일럿 수집 *전에* 끝내야 한다.** 발행 주기를 올리면 명령 간 open-loop 구간이 50ms → 31.25ms로 줄어 텔레옵 움직임이 더 부드러워지고, 그만큼 **`Δq` 분포가 달라진다.** 파일럿을 20Hz로 받고 본 수집을 32Hz로 하면 두 데이터가 서로 다른 분포가 되어 파일럿의 검증 결과(특히 §7 #3 축별 range, #4 스파이크)를 본 수집에 그대로 적용할 수 없다.

순서: **`publish_rate_hz` 변경 → 파일럿 20~30ep → 검증 → 본 수집.**

혼입 방지를 위해 `episode_meta.json`의 `command_rate_hz`에 실제 발행 주기를 반드시 기록한다. 이미 20Hz로 받아둔 데이터가 있다면 폐기하지 말고 이 필드로 분리 관리할 것.

---

## 7. 파일럿 (0단계) 체크리스트 — 20~30 에피소드

본 수집 전에 파이프라인 검증만 목적으로 한다.

| # | 확인 항목 | 방법 |
|---|---|---|
| 1 | **관절공간 실행 경로 존재 + 도달 rate** | §1. 최우선. 실패 시 계약 재설계 |
| 2 | `twist_raw` vs `twist_sent` 클램핑 발생 여부 | 두 컬럼 차이 분포 |
| 3 | **축별 `Δq`의 q01/q99 range** | ★ 아래 설명 |
| 4 | 특이점 근처 `Δq` 스파이크 | `Δq` 절대값 분포 상위 꼬리 |
| 5 | joint 순서 / degree 단위 / delta 정상성 | `scripts/analyze_parquet_action_semantics.py` |
| 6 | 그리퍼 값 범위·전이 | `scripts/analyze_parquet_gripper_dim7.py` |
| 7 | 이미지 ↔ joint timestamp 정합 | `scripts/verify_phase5_manifest_hf_alignment.py` |
| 8 | task_index → prompt | `scripts/verify_lerobot_task_contract.py` |
| 9 | end-to-end 1바퀴 | 수집 → 변환 → `compute_norm_stats` → 100 step 학습 |

### #3이 특히 중요한 이유

E6에서 j5가 기계적으로 ≈-88°에 고정돼 있었고, 그 결과 action의 quantile range가 **0.245**(다른 축 3.0~4.3의 1/13)였다. 정규화가 그 축의 센서 지터를 [-1,1]로 펴버려서 **학습 그래디언트의 30.5%가 노이즈 피팅에 쓰이고 있었다** (v23 체크포인트 실측: dim4 sq_err 0.2256 / 실제 7차원 합 0.7388).

xArm6는 j5 고정이 없지만, **Cartesian velocity 제어는 관절 운동을 상관시키는 경향이 있어** 다른 형태로 저분산 축이 생길 수 있다. 파일럿 데이터가 나오는 즉시 5분 안에 확인할 것. 나오지 않는다면 그것도 결과다 — E6의 병리가 embodiment-specific임을 보이는 대조군이 된다.

---

## 8. 학습서버 쪽 진행 상황

| 항목 | 상태 |
|---|---|
| `src/openpi/policies/e7_policy.py` | ✅ 8D → **7D** 정리 완료 (align 플래그 없음, `acts[:, :7]`). 스모크 통과 |
| `LeRobotE7DataConfig` | ✅ 7D로 정정 |
| `pi05_e7_v1_lora` | ✅ 주석 블록 xArm6/7D로 정정. `repo_id` 는 TODO (수집 후 교체) |
| `model.action_dim=32` | 유지 — 모델 내부 패딩 차원. v23과 동일해야 비교 성립. 패딩 차원 비용은 로스의 1%로 실측됨 |
| `align_droid_state` | **False 확정.** 전수 확인 결과 `True`는 v14 하나뿐, v15~v32(v16/v17/**v23 최우수**) 전부 False |
| `convert_e7_to_lerobot.py` | ⏳ 미작성. 본 스펙 확정 후 `convert_e6_v16_to_lerobot.py` 기반으로 작성 |

### 변환 시 적용될 규칙 (E6와 동일)

```
후방 트림: 그리퍼 1→0 (place) + PLACE_SETTLE(10) 까지
전방 트림: 첫 움직임(max|Δjoint| ≥ 0.1도) - 1 프레임부터
인접쌍   : state = [j(t), gripper(t)],  action = [j(t+1)-j(t), gripper(t+1)]
phase    : 그리퍼 전이 + 평활화된 |dj1/dt| 로 6구간 분할 → 방향 2 × 6 = task_index
출력 프레임 수 = trimmed - 1
```

> **phase 분할 로직은 "j1 회전이 좌↔우 이동의 주축"이라는 E6 가정에 묶여 있다.** xArm6도 베이스 회전축이 j1이라 대체로 이식되겠지만, 워크스페이스 배치가 바뀌면 fallback(30/45/25 비례 분할)으로 떨어질 수 있다. 파일럿에서 phase 경계를 눈으로 확인할 것.

---

## 9. Jetson 수집 노드 패치 목록

2026-07-27 대조 기준, 스펙과 현재 코드의 차이. **파일럿 수집 전에 전부 적용할 것.**

### `collection_control.yaml`

| # | 항목 | 현재 | 목표 |
|---|---|---|---|
| 1 | `publish_rate_hz` | `20.0` | **`32.0`** (§6 — 순서 주의: 파일럿 *전에*) |

### `vla_data_collector_node.py`

| # | 항목 | 현재 | 목표 |
|---|---|---|---|
| 2 | 그리퍼 컬럼명 | `gripper` | **`gripper_command`** (값은 연속 [0,1] 원값) |
| 3 | 이중 클럭 | ROS system clock만 | `timestamp` + **`timestamp_monotonic`** |
| 4 | 타임스탬프 세분화 | 없음 | **`joint_state_timestamp`, `teleop_timestamp`, `command_send_timestamp`** |
| 5 | 이미지 timestamp 출처 | 확인 필요 | **센서 캡처 시각**(ZED/HIK 프레임 헤더). 저장 시각 금지 |
| 6 | 명령/프레임 카운트 | 없음 | **`n_commands_in_frame`, `last_command_timestamp`** |
| 7 | twist 2종 | 확인 필요 | **`teleop_twist_raw_*`(6), `twist_sent_*`(6)** — 클램핑 검출용 |
| 8 | 프레임 단위 드롭 | `episode_meta.camera_sync`만 (에피소드 집계) | **`frame_dropped_before`, `dt_from_prev`** |
| 9 | 에러 래치 | 확인 필요 | **`error_latched`, `error_code_first`, `warn_latched`, `warn_code_first`** |
| 10 | `place_x/y` | `place_anchor` 스텝만, 미기입 | **실제 좌표 기입** |
| 11 | `episode_meta.json` 제어 계약 | 일부 | §4.2 필드 전체 |

### 적용 후 검증

파일럿 1~2 에피소드로 확인:
- 컬럼명·개수가 §3과 일치
- `n_commands_in_frame`이 32Hz 적용 후 **일관되게 2** (20Hz면 1/2 혼재)
- `dt_from_prev` 분포가 62.5ms 근처에 모임, 이상치 없음
- `timestamp_monotonic` 단조 증가
- `teleop_twist_raw_*` vs `twist_sent_*` 차이 발생 여부 (드라이버 클램핑)
- `gripper_command` 값 범위와 0→1 / 1→0 전이 검출 가능 여부

---

## 9-bis. 파일럿 ep0 판정 결과 → 재수집 전 조치 3건

2026-07-28 파일럿 1 에피소드(685프레임, 캔)를 실제로 변환·분석한 결과. **이 3개만 고치면 되고, 나머지 수집 형식은 그대로 유지.**

| # | 조치 | 근거 (실측) |
|---|---|---|
| 1 | **텔레옵 gain 하향** — 클램핑률 10% 미만까지 | `teleop_raw [-3.21, 3.94]` vs `twist_sent [-0.99, 1.00]`, **48% 프레임에서 클리핑**. 조작자가 시간의 절반을 ±1 한계에 부딪히며 조작 중 → 기록되는 궤적이 의도의 클리핑판 |
| 2 | **충돌에러 나면 그 에피소드 폐기하고 재수집** | grasp~release 사이에 `robot_mode==0`(위치제어 복구)이 3회. 궤적이 끊겨 학습 불가. 변환기가 자동 SKIP하고 발생률 리포트 |
| 3 | **앵커 1~9 골고루** | `dj4` range 0.104 vs median 1.90(1/18) — E6 j5 병리 재발 후보. j4 절대 span이 3.83°뿐이라 "이 궤적에서만 안 쓴 것"인지 구조적 저분산인지 앵커를 흩어야 갈림. 에피소드 수보다 앵커 분산이 중요 |

**바꾸지 않아도 되는 것** (실측으로 확인됨)
- **phase 버튼 불필요** — TCP 기하로 변환 시점에 자동 계산. `place_x/y`가 release 순간 실제 TCP와 정확히 일치함을 확인, release 직전 변위로 삽입축이 나옴
- **그리퍼 게이팅** — 실제 지연 2프레임(125ms)으로 0.4s 상한에 안 붙음. `min_send_interval_sec` 손대지 말 것
- 녹화 16Hz / 명령 32Hz / CSV 44컬럼 / 224 이미지 / 카메라 skew(p95 19ms)

**⚠️ retract 손실** — release 이후 텔레옵 retract(mode5, TCP +119mm)가 사이의 mode0 때문에 별개 구간이 되어 버려지고 있음. 이어붙이면 관절 궤적에 가짜 점프가 생김. **책장 삽입에서는 삽입축을 따라 빼는 동작이 필수 스킬**이므로, 가능하면 grasp부터 retract까지 mode 5를 끊김 없이 유지할 것. 변환기가 버려진 구간을 TCP 변위와 함께 리포트함.

**책 태스크 전환 시 `episode_meta.json` 추가 필드**
```json
"task_id":          "book_shelf_insert",   // "insert" 포함 시 변환기가 insertion 스키마 자동 선택
"object_category":  "sociology",
"prompt_object_name": "sociology book",    // 프롬프트에 그대로 들어감
"target_shelf":     "blue",                // 프롬프트에 그대로 들어감
"target_relation":  "left",
"rule_version":     "v01"                  // counterfactual 순열 식별자
```

---

## 10. 미결 사항

1. ~~관절공간 실행 경로~~ → **✅ 해소** (§1, ServoJ + joint velocity 둘 다 확인, ≈515Hz)
2. **그리퍼 이진 vs 연속** — 수집은 연속 원값으로 하고 계약은 변환 시 결정. 이진이면 E6 비교 변수가 embodiment 하나로 깔끔해지고, 연속이면 pi0 사전학습 관례(DROID/ALOHA)에 맞아 전이가 유리할 수 있으나 비교 변수가 둘로 늘어난다
3. **`delta_reference`** — `measured` vs `commanded`. 라벨 자기일관성은 `measured` 쪽. 파일럿에서 실측 비교 (§1.2)
4. **`convert_e7_to_lerobot.py`** 미작성 — 파일럿 스키마 확정 후
