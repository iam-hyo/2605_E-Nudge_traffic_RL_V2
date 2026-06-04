# HANDOFF — 강남구 OD-2 재실험 (2026-06-04 진행 중)

> 이 작업을 이어받는 Claude 세션용 인계 메모. (`claude --continue` 로 맥락이
> 살아있으면 참고만, fresh 세션이면 이 문서로 전체 복원 가능)

## 목표
효준 OD-1 신논현 재실험과 동일 방법론으로 **OD-2 재실험**.
- 좌하단(SW)→우상단(NE) 경향 유지하며 '거리최단 ≠ 연료최단' 격차 큰 OD 선정
- 환경을 **gangnam_topology_add.json** (신호 231→614개 강화)으로 교체해 학습
- 산출물: 학습곡선 그래프, 시뮬레이션 GIF, 변경 학습파라미터 정리, 경로 PNG
- 보고서 → `output/{expdir}/report.md` + **노션 OD-2 페이지**
- OD-3 는 실험 안 함 (OD-1 은 일반화 참고용으로 유지)
- 루트 설명 + 결과 분석 자세히. RL 모델별 성능차 원인 분석 + 추가 실험 제안.

## 선정된 OD-2 (LOCKED)
- **출발 338934** (도곡역 일대, ~351m, **무신호 degree-3** — 정체 corridor 입구)
- **도착 341712** (영동대교 남단, ~277m)
- 직선 4.25km / 주행 5.8km
- 사전 Dijkstra(노이즈 제거) peak fuel TDD **−20.9%** (short 1396mL/대기1363s →
  fuelTDD 1104mL/대기673s), off-peak **−25.4%**
- 후보 스크리닝 산출물: `output/_od2_screening.json`, `output/_od2_rescreen.json`
  (스크립트: `util/screen_od2.py`, `util/rescreen_od2.py`)

## 설정 / 스크립트 (모두 디스크에 있음)
- config: **`config/config_gangnam_od2.yaml`** (topology=gangnam_topology_add,
  routes[0]=od2_dogok_yeongdong 338934→341712, routes[1]=od1_sinnonhyeon_suseo 참고,
  episodes 15000, train_max_steps 400, arrival_bonus 700, ε_decay 0.999975,
  checkpoint 1500, uturn_penalty 20, primary_boost 4, shaping_w 1500)
- 학습 런처: `util/train_od2_launch.py` (RL 3종 병렬, nohup 으로 띄워둠)
- supervisor: `util/supervisor_od2.sh` (nohup) — 학습완료 → 실험 → 학습곡선
  (`util/plot_learning_od2.py`) → 경로PNG (`util/gen_gangnam_hires.py`,
  env GANGNAM_CFG 로 config 지정) → 비교GIF → `.SUPERVISOR_DONE` 마커
- OD-1(clean) 학습모델 백업: `models/_archive_v4_0530_od1_clean/`

## 진행 상태 확인
```bash
cd /home/hjj/ms_ENudge/2605_E-Nudge_traffic_RL_V2
tail -3 output/train_logs/model_rl_base.log          # 학습 진행
tail -3 output/train_logs/model_rl_signal_attention.log
cat output/train_logs/.OD2_EXPDIR 2>/dev/null        # 실험 폴더명 (생기면)
ls output/*_dogok_yeongdong_od2/.SUPERVISOR_DONE 2>/dev/null  # 전체 완료 마커
```

## 남은 일 (학습+supervisor 완료 후 = Claude 가 해야 할 것)
1. `output/{expdir}/summary.json` + `results.csv` 읽어 KPI 표 작성
2. `output/{expdir}/report.md` 작성 — OD-1 report.md 구조 미러:
   실험의도 / OD선정근거 / 변경 학습파라미터 / 환경설정 / KPI표(OD-2 peak·off,
   OD-1 참고) / 핵심발견 / **RL 모델별 성능차 원인분석** / 결론 / 한계 /
   **검증·개선 추가실험 제안**
3. 노션 기록 — 페이지 **"강남구 OD-2 재실험"**
   URL: https://www.notion.so/OD-2-3723115cfcc7805db4aee88e41121627
   page id: `3723115cfcc7805db4aee88e41121627`
   (부모: 강남구 토폴로지 분석). 표·이미지 넣을 수 있게 작성.
   PNG/GIF 바이너리는 MCP 업로드 불가 → 사용자 수동 드래그&드롭 안내 (메모리 규칙).

## 참고: OD-1 재실험 산출물 (구조/문체 미러용)
`output/30_0101_sinnonhyeon_gangnam/report.md`, 노션 OD-1 페이지
https://www.notion.so/3703115cfcc7811eaa6fd3fbcb7aac04
