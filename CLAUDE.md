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
3. `report.md`    — 실험 전반 보고서(실험 의도 + 실험 설계 + KPI 표 + 핵심 발견 + 결론 + 한계 및 문제점)
4. `{topology}_{model1}_{model2}.gif` — 핵심 모델 비교 시뮬레이션
     예) `6x6cross_short_rlattn.gif`

## 작업 방침
- 학습시 에포크를 넉넉하게 확보하여 성능우선적으로 시행하라.
- 실험 위주 프로젝트 — 코드 수정·데이터 생성·학습·실험을 적극 진행한다.
- 임시 worktree/샌드박스를 만들지 않고 이 디렉터리에서 직접 작업한다.
- 큰 구조 변경(아키텍처·State 차원·보상 구조)만 사전 확인하고,
  그 외 실험·튜닝·재학습은 바로 진행한다.
- 실험 종료 시 산출물 4종을 빠짐없이 생성하고 노션에 로깅한다.

## 한글 폰트 규칙 (필수)
한글이 들어가는 모든 시각화 산출물(그래프·GIF·PNG 등)은 **반드시 한글(CJK)
폰트를 matplotlib 에 설정**해 글자가 깨지지(□ tofu) 않게 한다.
- matplotlib import 직후, `pyplot` import 전에 `rcParams["font.family"]` 설정.
- `simulation.py` / `util/viz_12x12_map.py` 의 `_pick_korean_font()` 패턴 재사용
  (Malgun Gothic / AppleGothic / NanumGothic / **Noto Sans CJK** 순 탐색).
- `rcParams["axes.unicode_minus"] = False` 도 함께 설정 (마이너스 기호 깨짐 방지).
- 새 시각화 스크립트 작성 시 이 설정을 빠뜨리지 말 것.

## 핵심 명령
- 데이터 생성:  `python main.py --step data`
- 학습:        `python main.py --step train`
- 실험:        `python experiments/run_experiment.py`
- 시뮬 GIF:    `python simulation.py --models <..> --gif_only`