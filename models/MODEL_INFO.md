# 모델 명세

학습 환경: **6x6 테스트베드** (`data/6x6_topology.json`, `data/6x6_speed_data.csv`)
학습일: **2026-05-19**
보상 함수: `R = -α·fuel_mL + arrival_bonus(500)·𝟙_goal`
연료 모형: VT-Micro (Rakha-Ahn, dt=0.1s 누적 적분)

## 학습 완료 모델 (현행)

| 키 (config) | 모드 | 파일 | 에피소드 | 도달률 | 평균 fuel | 평균 wait | 비고 |
|---|---|---|---|---|---|---|---|
| `shortest_dijkstra` | — | `model_shortest_dijkstra.pkl` | — | — | — | — | Dijkstra 객체, 학습 없음 |
| `static_fuel_dijkstra` | — | `model_static_fuel_dijkstra.pkl` | — | — | — | — | Time-Dependent Dijkstra |
| `rl_base` | DQN base (신호 미사용) | `model_rl_base.pth` | 800 | 100% | 562 mL | 115 s | 신호 9d를 forward에서 마스킹 |
| `rl_signal` | DQN signal | `model_rl_signal.pth` | 800 | 100% | 582 mL | 132 s | flat 229d MLP |
| `rl_signal_attention` | DQN attention | `model_rl_signal_attention.pth` | 800 | 100% | 581 mL | 131 s | 13 노드 토큰 self-attention |

`*_best.pth` 는 학습 중 최고 도달률 시점 스냅샷.
`*_history.json` 은 에피소드별 reward/fuel/wait/steps/moves 기록.

## 추론용 파일 매핑 (코드 호환)

`experiments/run_experiment.py`, `simulation.py`, `visualize.py`, `experiments/visualize.py` 모두 아래 파일명을 하드코딩해서 로드한다.

```
model_shortest_dijkstra.pkl        ← shortest_dijkstra
model_static_fuel_dijkstra.pkl     ← static_fuel_dijkstra
model_rl_base.pth                  ← rl_base
model_rl_signal.pth                ← rl_signal
model_rl_signal_attention.pth      ← rl_signal_attention
```

## 아카이브

- `_archive_old_10x10/`: 옛 10x10 환경 + 옛 보상함수(VT-Macro / shaping 50) 시절 모델. 새 환경과 호환 안 됨 — 시뮬레이션·평가에 사용 금지.

## 재학습 방법

```bash
# 단일 모델
python train/03_train_rl_base.py
python train/04_train_rl_signal.py
python train/05_train_rl_signal_attention.py

# 전체 (Dijkstra 2 + RL 3)
python main.py --step train
```

학습 시간 기준: 800 ep / Base 38분 / Signal 22분 / Attention 25분 (CPU). 강남구 전환 시 1500~3000 ep 권장 (`config/config.yaml` 의 `episodes` 변경).
