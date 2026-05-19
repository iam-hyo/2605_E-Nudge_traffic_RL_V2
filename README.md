# Traffic RL — 최소 연료 경로 탐색

강화학습 기반 최소 연료 소모 경로 탐색 실험 프레임워크.

---

## 프로젝트 목표

신호 정보와 실시간 교통 속도를 활용하여  
**시간이 더 걸리더라도 연료 소모가 최소인 경로**를 탐색하는 DQN Agent 개발.

---

## 파일 구성

```
project_root/
├── data/
│   ├── 10x10_topology.json     # 10×10 합성 그리드 네트워크
│   └── speed_data.csv          # 링크별 속도 (07:00~08:55, 5분 단위)
│
├── util/
│   ├── generate_data.py        # topology + speed_data 자동 생성
│   ├── environment.py          # RoadNetworkEnv (State 229d, step, reset)
│   ├── fuel_calculate.py       # VT-Macro + 공회전 연료 계산
│   ├── reward.py               # 보상 체계 (연료 패널티 + 도착 보너스)
│   ├── model.py                # QNetworkBase / Signal / Attention
│   ├── agent.py                # DQNAgent (Double DQN + Dueling)
│   └── dijkstra_models.py      # ShortestDijkstra / StaticFuelDijkstra
│
├── config/
│   └── config.yaml             # 하이퍼파라미터 · 실험 경로 중앙 관리
│
├── train/
│   ├── 01_train_shortest_dijkstra.py
│   ├── 02_train_static_dijkstra.py
│   ├── 03_train_rl_base.py
│   ├── 04_train_rl_signal.py
│   └── 05_train_rl_signal_attention.py
│
├── experiments/
│   ├── run_experiment.py       # 전체 실험 수행 · 결과 저장
│   ├── evaluate.py             # KPI 통계 출력
│   └── visualize.py            # 학습 곡선 · 경로 비교 시각화
│
├── models/                     # 학습된 .pth / .pkl 저장
├── output/                     # 실험 결과 CSV (타임스탬프별 폴더)
├── main.py                     # 통합 실행 엔트리포인트
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
python experiments/evaluate.py
```

---

## 실험 설계

| 항목 | 내용 |
|---|---|
| 환경 A | 10×10 합성 그리드 (중간 발표) |
| 환경 B | 강남구 실제 데이터 (최종 발표, Data 교체만으로 확장) |
| 경로 | 단거리 2쌍 + 장거리 2쌍 = 총 4쌍 |
| 시간대 | 07:00 (한산) · 08:00 (병목) |
| 반복 | 경로당 30회 → 총 **240회 / 모델** |

### 대조군

| # | 모델 | 설명 |
|---|---|---|
| ① | Shortest Dijkstra | 링크 길이 최단 경로 |
| ② | Static Fuel Dijkstra | 예상 연료 최적 (Time-Dependent, 이론 기준선) |
| ③ | RL Base | 신호 State 미사용 |
| ④ | RL Signal | 신호 State 포함 |
| ⑤ | RL Signal + Attention | 전체 모델 (최종 제안) |

### KPI

| 순위 | 지표 | 단위 |
|---|---|---|
| 1 | 연료 소모량 | mL |
| 2 | 소요 시간 | 초 |
| 3 | 이동 거리 | m |
| 참고 | 신호 대기 시간 | 초 |
| 참고 | 목표 도달률 | % |

---

## 가정 사항

- 링크는 모두 양방향, 상행/하행 동일 평균 속도에서 가우시안 노이즈(±20%) 샘플링
- 가속도 고정: 2.5 m/s² (가속·감속 동일)
- 우회전 목표 속도: 20 km/h / 좌회전: 30 km/h
- 연료 모델: VT-Macro (주행) + Idle Fuel Rate 0.5 mL/s (신호 대기)
- 시간 패널티 없음 — 목표는 최소 연료 (시간 trade-off 허용)
- State 차원: 229 (위치 5 + 시간 3 + 현재신호 9 + 1-hop 노드 44 + 1-hop 링크 8 + 2-hop 노드 88 + 2-hop 링크 72)

---

## simulation.py

### 기본 실행

```bash
# 2개 모델 비교
python simulation.py --models shortest_dijkstra rl_base --route long_01 --time_slot peak

# 전체 모델
python simulation.py --models all --route short_01 --time_slot off_peak
```

### 속도 / 주사율 조절

| 옵션 | 설명 | 예시 |
|---|---|---|
| `--interval N` | 프레임 간격 ms (기본 33 = 30fps) | `--interval 16` → 60fps |
| `--speed N` | 애니메이션 배속 1~4 (기본 1배속) | `--speed 2` → 2배 빠르게 |

```bash
# 30fps · 1배속 (기본)
python simulation.py --models rl_signal --route long_01

# 60fps · 더 부드러운 움직임
python simulation.py --models rl_signal --route long_01 --interval 16

# 2배속으로 빠르게 관람
python simulation.py --models all --route short_01 --speed 2

# 고주사율 + 빠른 재생
python simulation.py --models rl_signal rl_signal_attention --interval 16 --speed 3
```

**배속별 링크 통과 시간:**

| `--speed` | frames/link | `--interval 33` 기준 | `--interval 16` 기준 |
|---|---|---|---|
| 1 (기본) | 25 | ~825 ms/link | ~400 ms/link |
| 2 | 12 | ~400 ms/link | ~192 ms/link |
| 3 | 8  | ~264 ms/link | ~128 ms/link |
| 4 | 6  | ~198 ms/link | ~96 ms/link  |

### 신호 준수 규칙

모든 모델(rl_base 포함)은 시뮬레이션 중 교통 법규를 준수합니다.
rl_base는 **경로 탐색 시에만** 신호 정보를 미사용하며, 운행 중에는 동일하게 신호를 따릅니다.

| 신호 | 색상 | 허용 동작 |
|---|---|---|
| 녹색 | ● 초록 | 직진 · 우회전 통과 |
| 파랑(← 좌회전) | ● 파랑 | 좌회전 · 우회전 통과 |
| 적색 | ● 빨강 | **전체 정지** (우회전 포함) — 녹색까지 대기 |
| 무신호 | ○ 회색(소) | 즉시 통과 |

### 화면 구성

- **좌측**: 모델별 정보 카드 (속도 / 누적연료 / 누적시간 / 스텝) + 실시간 연료·대기시간 그래프
- **우측**: 도로망 — 신호 색(녹/적/파랑) + 영구 경로 흔적(흰 테두리 + 모델 컬러) + 에이전트 이동 + 대기 링

## visualize.py

[2개 모델 비교 → output/visualize/20260514/ 에 저장]
python visualize.py --models shortest_dijkstra rl_base --route long_01

[전체 모델, 도달률 20회 계산]
python visualize.py --models all --route short_01 --reach_trials 20
저장 파일 6개:

파일	내용
01_route.png	경로 시각화
02_fuel.png	누적 연료 꺾은선
03_wait.png	누적 대기시간 꺾은선
04_speed.png	스텝별 속도 꺾은선
05_reward.png	누적 리워드 꺾은선
06_reach_rate.png	도달률 막대그래프

---

## 링크별 속도 상세

### 속도 데이터 구조 (`data/speed_data.csv`)

| 컬럼 | 내용 |
|---|---|
| `link_id` | 링크 식별자 (`end1_end2` 형식) |
| `t_0` ~ `t_23` | 07:00~08:55 구간 5분 단위 평균 속도 (km/h) |

**시간대별 기준 속도 프로파일:**

| 시간대 | arterial (주도로) | local (이면도로) |
|---|---|---|
| 07:00~07:20 (원활) | 44~48 km/h | 33~36 km/h |
| 07:25~08:10 (병목) | 18~33 km/h | 13~25 km/h |
| 08:15~08:55 (회복) | 40~49 km/h | 30~37 km/h |

**속도 변동 요인:**
1. **도로 타입**: arterial(주도로) × 1.0 / local(이면도로) × 0.75 — 두 타입 간 약 25% 차이
2. **링크 고유 오프셋**: 링크마다 ±3 km/h 고정 편차 (동일 타입이어도 링크별로 다름)
3. **시뮬레이션 노이즈**: 매 스텝마다 ±20% 가우시안 샘플링 (`NOISE_SIGMA = 0.20` in `environment.py`)

> **실제 속도 차이가 크지 않게 보이는 이유**: 합성 데이터는 링크별 편차가 ±3 km/h로 제한됨.  
> 강남구 실제 데이터로 교체 시 링크별 편차가 훨씬 크게 나타남.

### 속도 데이터 커스터마이징

```python
# util/generate_data.py 수정 포인트

# 1. 기준 속도 프로파일 변경 (07:00~08:55, 5분 단위 24슬롯)
BASE_SPEED_PROFILE = [48, 46, 44, ...]   # km/h

# 2. 도로 타입별 스케일 조정
ROAD_TYPE_SCALE = {"arterial": 1.0, "local": 0.75}

# 3. 링크별 편차 크기 조정 (현재 ±3 km/h)
link_offset = rng.gauss(0, 3.0)   # σ를 키울수록 링크간 속도 차이 증가

# 재생성
python util/generate_data.py
```

---

## 강남구 데이터 교체 방법

```bash
# 1. 강남구 topology 파일 준비
cp topology_GangNam.json data/10x10_topology.json   # 파일명 통일 또는

# 2. config.yaml 경로 변경
data:
  topology: "data/topology_GangNam.json"
  speed:    "data/speed_data_GangNam.csv"

# 3. 실험 경로(routes) 실제 노드 ID로 업데이트 후 실행
python main.py --step train
python main.py --step experiment
```

---

## 발전 방향

- [ ] 강남구 실제 데이터 적용
- [ ] VT-Macro 계수 실측 데이터로 캘리브레이션
- [ ] 다중 출발지·목적지 동시 최적화
- [ ] LSTM 기반 교통 예측 모듈 통합
