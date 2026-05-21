# CLAUDE.md — Traffic RL 프로젝트 작업 지침

강화학습 기반 최소 연료 경로 탐색. 신호+실시간 속도 인식 DQN.
프로젝트 상세는 README.md, 현재 진행 상황은 HANDOFF.md 참조.

## 실험 산출물 규칙 (필수)

모든 실험(`experiments/run_experiment.py` 등)은 아래 규칙으로 산출물을 저장한다.

### 저장 위치
`output/{DD}_{HHMM}_{topology}_{detail}/`
  예) `output/20_2340_6x6cross_arrival200/`

  - `DD_HHMM` : 실행 시각 (일_시분)
  - `topology`: 6x6 / 6x6cross / 10x10 / gangnam
  - `detail`  : 실험 특징 한 단어 (arrival200, ep1500 …)

### 필수 산출물 4종
1. `results.csv`  — 전체 run raw 데이터
2. `summary.json` — 모델×경로×시간대 KPI 요약
3. `report.md`    — 결과 해설 (KPI 표 + 핵심 발견 + 결론 + 한계 및 문제점)
4. `{topology}_{model1}_{model2}.gif` — 핵심 모델 비교 시뮬레이션
     예) `6x6cross_short_rlattn.gif`

## 작업 방침
- 실험 위주 프로젝트 — 코드 수정·데이터 생성·학습·실험을 적극 진행한다.
- 임시 worktree/샌드박스를 만들지 않고 이 디렉터리에서 직접 작업한다.
- 큰 구조 변경(아키텍처·State 차원·보상 구조)만 사전 확인하고,
  그 외 실험·튜닝·재학습은 바로 진행한다.
- 실험 종료 시 산출물 4종을 빠짐없이 생성하고 노션에 로깅한다.

## 핵심 명령
- 데이터 생성:  `python main.py --step data`
- 학습:        `python main.py --step train`
- 실험:        `python experiments/run_experiment.py`
- 시뮬 GIF:    `python simulation.py --models <..> --gif_only`