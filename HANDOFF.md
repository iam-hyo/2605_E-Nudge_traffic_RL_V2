# 작업 핸드오프 — Traffic RL 프로젝트

> 이 문서는 머신 간(로컬 Windows ↔ 서버 Linux) 작업 인계용.
> Claude Code 대화 세션은 머신 로컬에만 저장되므로, 새 머신에서 `claude` 실행 후
> "HANDOFF.md 읽고 이어서 진행해줘" 라고 하면 맥락을 복원할 수 있다.

최종 업데이트: 2026-05-20

---

## 1. 프로젝트 개요

강화학습 기반 최소 연료 경로 탐색. 신호 정보 + 실시간 속도로 "느려도 연료 최소"
경로를 찾는 DQN 에이전트. 상세는 [README.md](README.md) 참조.

- State 229d, VT-Micro 연료 모형, 보상 단일화(`-α·fuel + arrival_bonus`)
- 모델 5종: shortest_dijkstra / static_fuel_dijkstra / rl_base / rl_signal / rl_signal_attention

## 2. 현재 진행 상황 (2026-05-20)

### 완료
- 6x6 격자 테스트베드 2회 실험 → **실패** (토폴로지 자기 모순: 외곽 link 가
  짧아 Shortest 가 우연히 우회 선택). 분석은 노션 "0520 6x6 실험결과 2".
- **6x6_cross 사선 토폴로지 신규 설계** (`data/6x6_cross_topology.json`):
  - B 사선(col 1, 2→32): 명시적 최단(3884m) + 신호 불리(green 7.5%, anti-sync)
  - C 사선(col 2, 3→33): 명시적 비최단(4200m) + 신호 우호(green 50%, green-wave)
  - 사전 검증: 양 시간대 모두 C 가 B 보다 fuel 우위 (off_peak −96mL, peak −38mL)
  - 의도: Shortest→B 선택, Signal-aware→C 선택 (신호 모델 우위 입증)

### 진행 중 / 다음 할 일
- [ ] **6x6_cross 환경 RL 3종 재학습** (1500 ep) — 로컬에서 백그라운드 학습 중이었음.
      서버에서는 `python train/03_train_rl_base.py` 등으로 재실행 (또는 학습된
      `models/model_rl_*.pth` 가 최신이면 그대로 사용).
- [ ] 재학습 후 `python experiments/run_experiment.py` 실험
- [ ] 결과 분석 → 노션 "0520 6x6 실험결과 2" 페이지에 로깅
      (https://www.notion.so/0520-6-6-2-3653115cfcc780e0af4beeda63907362)
- [ ] Shortest 가 B, RL Signal/Attention 이 C 를 선택하는지 확인이 핵심 KPI

## 3. 환경 설정 (config/config.yaml)

- topology: `data/6x6_cross_topology.json`, speed: `data/6x6_speed_data.csv`
- routes: cross_main (1→36) + 보조 3개
- arrival_bonus 200, episodes 1500, shaping_weight 0

## 4. 서버에서 학습 재개 방법

```bash
# 1. 의존성
pip install -r requirements.txt

# 2. 토폴로지 재생성 (필요 시 — 이미 data/ 에 커밋되어 있으면 생략 가능)
python util/generate_data_cross.py

# 3. 학습 (tmux 안에서 권장)
python train/01_train_shortest_dijkstra.py
python train/02_train_static_dijkstra.py
python train/03_train_rl_base.py
python train/04_train_rl_signal.py
python train/05_train_rl_signal_attention.py
#   또는: python main.py --step train

# 4. 실험
python experiments/run_experiment.py
```

## 5. 주의 사항

- 모델 파일명 규칙·학습 메타데이터는 `models/MODEL_INFO.md` 참조
- `models/_v1_0519/`, `_v2_0520_6x6/` 는 이전 버전 백업 (cross 실험과 무관)
- 학습 시간: 1모델 1500 ep ≈ CPU 2.5시간. GPU 서버면 훨씬 빠름
- Windows→Linux 이전 시 경로 구분자·인코딩 문제 없음 (모두 pathlib + utf-8 명시)
