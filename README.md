# Traffic RL — 최소 연료 경로 탐색

강화학습 기반 최소 연료 소모 경로 탐색 실험 프레임워크.

> **현행 환경**: 6×6 = 36 노드 테스트베드 (`data/6x6_topology.json`).
> 강남구 실데이터(1000+ 노드)도 동일 인터페이스로 즉시 확장 가능 — [강남구 전환](#강남구-데이터-전환) 참조.

---

## 프로젝트 목표

신호 정보와 실시간 교통 속도를 활용하여
**시간이 더 걸리더라도 연료 소모가 최소인 경로**를 탐색하는 DQN Agent 개발.

---

## 모델 5종 — 빠른 참조

`config/config.yaml` 의 `experiments.models` 키와 학습/시뮬레이션 CLI 인자에서 사용하는 이름이 동일.

| # | 모델 키 (CLI/config) | 종류 | 신호 인식 | 학습 모드 | 학습 스크립트 | 파일명 |
|---|---|---|---|---|---|---|
| ① | `shortest_dijkstra` | Dijkstra | — | 객체 저장만 | `train/01_train_shortest_dijkstra.py` | `models/model_shortest_dijkstra.pkl` |
| ② | `static_fuel_dijkstra` | Time-Dependent Dijkstra | 예상 대기 반영 | 객체 저장만 | `train/02_train_static_dijkstra.py` | `models/model_static_fuel_dijkstra.pkl` |
| ③ | `rl_base` | DQN (Dueling+Double) | ✗ State 마스킹 | `mode="base"` | `train/03_train_rl_base.py` | `models/model_rl_base.pth` |
| ④ | `rl_signal` | DQN (Dueling+Double) | ✓ flat 229d MLP | `mode="signal"` | `train/04_train_rl_signal.py` | `models/model_rl_signal.pth` |
| ⑤ | `rl_signal_attention` | DQN + Self-Attention | ✓ 13 노드 토큰 | `mode="attention"` | `train/05_train_rl_signal_attention.py` | `models/model_rl_signal_attention.pth` |

**CLI 호환 옵션** — `simulation.py` / `experiments/run_experiment.py` / `visualize.py`:
```bash
--models shortest_dijkstra            # ①
--models static_fuel_dijkstra         # ②
--models rl_base                      # ③
--models rl_signal                    # ④
--models rl_signal_attention          # ⑤
--models all                          # 전체
--models rl_signal rl_signal_attention   # 다중
```

학습된 모델의 학습일·환경·에피소드 등 메타데이터는 [models/MODEL_INFO.md](models/MODEL_INFO.md) 참조.

---

## 파일 구성

```
project_root/
├── data/
│   ├── 6x6_topology.json       # 현행 테스트베드 (36 노드)
│   ├── 6x6_speed_data.csv      # 링크별 속도 (07:00~08:55, 5분 단위)
│   ├── 10x10_topology.json     # 옛 환경 (호환 유지)
│   ├── 10x10_speed_data.csv
│   ├── gangnam_topology.json   # 강남구 실데이터 (전환 시 사용)
│   ├── gangnam_speed_data.csv
│   └── preprocessing/
│       └── signal_topology.py  # 강남구 토폴로지에 신호 추가
│
├── util/
│   ├── environment.py          # RoadNetworkEnv (State 229d, movement-aware step)
│   ├── fuel_calculate.py       # VT-Micro 다항회귀 + dt=0.1s 누적 적분
│   ├── reward.py               # 단일화 보상 (연료 + 도착만)
│   ├── model.py                # QNetworkBase / Signal / Attention
│   ├── agent.py                # DQNAgent (Double DQN + Dueling)
│   ├── dijkstra_models.py      # ShortestDijkstra / StaticFuelDijkstra
│   ├── generate_data.py        # 10x10 그리드 생성기 (옛 환경)
│   ├── generate_data_6x6.py    # 6x6 테스트베드 생성기 (코어·외곽 신호 분리)
│   └── viz_scale.py            # 시각화 자동 스케일 (36~1000+ 노드 호환)
│
├── train/
│   ├── _train_common.py        # 공통 학습 루프 (loss/이동분포/route별 도달률 로그)
│   ├── 01_train_shortest_dijkstra.py
│   ├── 02_train_static_dijkstra.py
│   ├── 03_train_rl_base.py
│   ├── 04_train_rl_signal.py
│   └── 05_train_rl_signal_attention.py
│
├── experiments/
│   ├── run_experiment.py       # 1,200 runs 일괄 실험
│   └── visualize.py            # 학습 곡선 + 경로 비교 이미지
│
├── config/
│   └── config.yaml             # 하이퍼파라미터 · 경로 · 시간대 중앙 관리
│
├── models/                     # 학습된 .pth / .pkl + MODEL_INFO.md
├── output/                     # 실험 결과 (타임스탬프별 폴더) + 학습 곡선
│
├── main.py                     # 통합 파이프라인 엔트리포인트
├── simulation.py               # 고주사율 애니메이션 시각화
├── visualize.py                # 단일 시뮬레이션 + 6개 그래프 저장
├── requirements.txt
└── README.md
```

---

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 전체 파이프라인 실행 (데이터 생성 → 학습 → 실험 → 시각화)
python main.py --step all

# 또는 단계별 실행
python main.py --step data          # 데이터 생성
python main.py --step train         # 전체 모델 학습
python main.py --step experiment    # 실험 수행
python main.py --step visualize     # 시각화

# 특정 모델만 학습
python main.py --step train --models rl_signal rl_signal_attention

# 개별 스크립트 실행
python train/04_train_rl_signal.py
python experiments/run_experiment.py
```

---

## 실험 설계

| 항목 | 내용 |
|---|---|
| 환경 (현행) | 6×6 = 36 노드 테스트베드 — 코어/외곽 신호 분리로 신호 효과 검증 |
| 환경 (확장) | 강남구 실데이터 1000+ 노드 — `config.yaml` 1줄로 전환 |
| 경로 | 단거리 2쌍 (short_01·02) + 장거리 2쌍 (long_01·02) = 4쌍 |
| 시간대 | 07:00 off_peak / 08:00 peak |
| 반복 | 경로당 30회 × 4 경로 × 2 시간대 = **240 runs / 모델** = **1,200 runs / 실험** |

### 6x6 신호 분류 (현행)

| 영역 | 노드 ID | cycle | green | 도로 속도 scale |
|---|---|---|---|---|
| 무신호 | 1, 6, 18, 31, 36 | — | — | — |
| **core_strong** | 8, 9, 10, 11, 14, 15, 16, 17 | 110s | **5s (4.5%)** | **0.35** |
| **core_weak** | 2, 3, 4, 5, 7, 12, 13 | 90s | 10s (11%) | 0.55 |
| outer | 19~30, 32~35 | 60s | **45s (75%)** | 1.00 |

→ 좌하단(1) → 우중단(18) 시나리오에서 **외곽 우회**가 코어 직진보다 fuel −57 mL + time −240s 유리 (peak 시간대).

### KPI

| 순위 | 지표 | 단위 |
|---|---|---|
| 1 | 연료 소모량 | mL (VT-Micro, dt=0.1s 적분) |
| 2 | 소요 시간 | 초 (주행 + 신호 대기) |
| 3 | 이동 거리 | m |
| 참고 | 신호 대기 시간 | 초 |
| 참고 | 좌/직/우 이동 분포 | 회 |
| 참고 | 목표 도달률 | % |

---

## 핵심 시스템 사양

### State 229d ([MDP 정의 노션](https://www.notion.so/MDP-35f3115cfcc78082b5bad061293b1d06))

| 그룹 | dim | 비고 |
|---|---|---|
| 위치 | 5 | cur xy + goal Δxy + dist |
| 시간 | 3 | sin/cos abs_sec + elapsed |
| 현재 신호 | 9 | cycle 절대 길이 + green/left 비율 + phase one-hot[3] + remain/cycle + sin/cos |
| 1-hop 노드 (K=4) | 44 | 노드당 11d (pos 2 + sig 9) |
| 1-hop 링크 (K=4) | 8 | 링크당 2d (len + 속도) |
| 2-hop 노드 (N=8) | 88 | 노드당 11d, BFS round-robin |
| 2-hop 링크 (L=12) | 72 | 링크당 6d (len + 속도 + parent_onehot[4]) |
| **합계** | **229** | 패딩 sentinel `pos=(-1,-1)` |

### 보상 함수 (단일화)

```
R = -α · fuel_mL  +  arrival_bonus · 𝟙_goal
```

- `α = 1.0`, `arrival_bonus = 500.0`, `penalty_timeout/dead = 0` (단일화)
- `shaping_weight = 0` (potential-based shaping 비활성화)
- 시간/대기/재방문 등 proxy reward 모두 제거 (Reward Hypothesis 준수)

### 연료 모형 — VT-Micro

```
ln(F) = Σᵢ₌₀..₃ Σⱼ₌₀..₃ Kᵢⱼ · aⁱ · sʲ
```

- Rakha-Ahn calibrated 32 계수 (가속/감속 부호별 분리)
- 입력 단위: 속도 km/h, 가속도 km/h/s
- 시간 적분: dt = 0.1s 누적 (비선형 항 bias 제거)
- 출력 L/s → mL 환산 (×1000)
- idle: `fc_rate(0, 0)` = 0.437 mL/s

### 신호 규칙 — Driver/Navigation 분리

- `environment.step()` 이 movement-aware 출발 대기를 단일 진실 공급원으로 처리
- `_phase_allows`: green→직진·우회전, left_turn→좌회전만, red/yellow→전부 정지
- `get_valid_actions()`: 좌회전 phase 없는 노드의 좌회전 액션 차단 (학습·시뮬 공통)
- `use_signal=False` (rl_base): **State에서만 신호 9d 마스킹**, dynamics는 항상 신호 준수

---

## simulation.py — 고주사율 애니메이션

```bash
# 2개 모델 비교
python simulation.py --models shortest_dijkstra rl_base --route long_01 --time_slot peak
python simulation.py --models shortest_dijkstra rl_signal_attention --route long_01 --time_slot peak

# 전체 모델
python simulation.py --models all --route short_01 --time_slot off_peak
```
rl_signal_attention / static_fuel_dijkstra / rl_base / rl_signal / shortest_dijkstra

### 속도 / 주사율 조절

| 옵션 | 설명 | 예시 |
|---|---|---|
| `--interval N` | 프레임 간격 ms (기본 33 = 30fps) | `--interval 16` → 60fps |
| `--speed N` | 애니메이션 배속 1~4 (기본 1배속) | `--speed 2` → 2배 빠르게 |
| `--save_gif` | GIF 저장 (Pillow 필요) | `output/gif/` |
| `--gif_only` | 창 없이 GIF만 (헤드리스) | CI/원격 환경용 |

### 신호 준수 규칙 (모든 모델 공통)

| 신호 | 색상 | 허용 동작 |
|---|---|---|
| 녹색 | ● 초록 | **직진 · 우회전만** 통과 |
| 파랑(좌회전) | ● 파랑 | **좌회전만** 통과 |
| 적색·황색 | ● 빨강 | **전체 정지** — 다음 허용 phase까지 대기 |
| 무신호 | ○ 회색(소) | 즉시 통과 |

rl_base도 시뮬레이션 중에는 동일하게 신호를 따릅니다. State에서만 신호 정보를 안 봄.

---

## visualize.py — 단일 시뮬레이션 + 6개 그래프

```bash
# 2개 모델 비교 → output/visualize/{YYYYMMDD}/
python visualize.py --models shortest_dijkstra rl_base --route long_01

# 전체 모델, 도달률 20회 계산
python visualize.py --models all --route short_01 --reach_trials 20
```

저장 파일:
| 파일 | 내용 |
|---|---|
| `01_route.png` | 경로 시각화 |
| `02_fuel.png` | 누적 연료 |
| `03_wait.png` | 누적 대기시간 |
| `04_speed.png` | 스텝별 속도 |
| `05_reward.png` | 누적 리워드 |
| `06_reach_rate.png` | 도달률 막대그래프 |

`experiments/visualize.py` 는 학습 곡선과 단일 경로 비교 이미지 생성:
```bash
python experiments/visualize.py --mode learning
python experiments/visualize.py --mode route --start 1 --goal 18
```

---

## 시각화 자동 스케일 ([util/viz_scale.py](util/viz_scale.py))

`density_scale = √(36 / n_nodes)` 기반 마커·도로·폰트 자동 조정:

| 환경 | n_nodes | density | marker_signal | link_lw |
|---|---|---|---|---|
| 6x6 testbed | 36 | 1.00 | 100 | 2.0 |
| 10x10 (옛) | 100 | 0.60 | 60 | 1.2 |
| 강남구 (전환 시) | 1000 | 0.19 | 20 (min) | 0.40 |

좌회전 ← 텍스트는 200 노드 초과 시 자동 생략 (시각적 잡음 방지).

---

## 강남구 데이터 전환

```yaml
# config/config.yaml 1줄 변경
data:
  topology: data/gangnam_topology.json
  speed:    data/gangnam_speed_data.csv

# routes를 강남구 실제 노드 ID 4쌍으로 업데이트 후
# (1,500~3,000 ep 권장)
```

```bash
python main.py --step train         # 6x6 학습된 모델은 자동 덮어쓰기 — 사전 백업 권장
python main.py --step experiment
```

상세 호환성 점검은 [0519 실험결과 노션](https://www.notion.so/3653115cfcc780b79c85f679a25980d8) "7. 강남구 데이터 전환 점검" 참조.

---

## 발전 방향

- [ ] 강남구 실데이터 본 학습 (1,500~3,000 ep)
- [ ] arrival_bonus 감소(500→200) — fuel 절감 인센티브 강화 실험
- [ ] LSTM 기반 교통 예측 모듈 통합
- [ ] 다중 출발지·목적지 동시 최적화
