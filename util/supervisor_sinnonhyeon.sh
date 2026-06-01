#!/usr/bin/env bash
# 신논현 OD-1 재실험 supervisor
#   - RL 3종 학습 대기 → 실험 → 시각화 → 비교 GIF → 마커 파일
#   - 학습 launcher 가 종료될 때까지 대기 (단순/안정)
#   - 모든 출력은 output/train_logs/_supervisor.log 에 기록
set -u
cd /home/hjj/ms_ENudge/2605_E-Nudge_traffic_RL_V2
LOG="output/train_logs/_supervisor.log"
PY="venv/bin/python"
CFG="config/config_gangnam.yaml"

mkdir -p output/train_logs
echo "[supervisor] start $(date)" > "$LOG"

# ── 1. 학습 launcher PID 대기 ─────────────────────────────────────────────────
LAUNCH_PID="${1:-}"
if [ -z "$LAUNCH_PID" ]; then
  LAUNCH_PID=$(pgrep -f "train_gangnam_launch" | head -1)
fi
echo "[supervisor] waiting for launcher PID=$LAUNCH_PID" | tee -a "$LOG"
while kill -0 "$LAUNCH_PID" 2>/dev/null; do
  sleep 120
  echo "[supervisor] $(date +%H:%M)  training in progress (PID $LAUNCH_PID)" >> "$LOG"
done
echo "[supervisor] launcher done $(date)" | tee -a "$LOG"

# ── 2. 모델 파일 검증 ─────────────────────────────────────────────────────────
for m in model_rl_base model_rl_signal model_rl_signal_attention; do
  if [ ! -f "models/${m}.pth" ]; then
    echo "[ERROR] models/${m}.pth missing — abort" | tee -a "$LOG"
    exit 1
  fi
done
echo "[supervisor] all 3 RL models present" | tee -a "$LOG"

# ── 3. 실험 수행 (900 runs) ────────────────────────────────────────────────────
echo "[supervisor] running experiment $(date)" | tee -a "$LOG"
$PY -c "from experiments.run_experiment import main; main('$CFG')" >> "$LOG" 2>&1

# ── 4. 실험 출력 폴더를 descriptive name 으로 이동 ─────────────────────────────
# run_experiment.py 는 output/DD_HHMM/ 생성. 이번 세션 시작 후 가장 최신을 이동.
LATEST=$(ls -td output/[0-9][0-9]_[0-9][0-9][0-9][0-9] 2>/dev/null | head -1)
EXPDIR="output/$(date +%d_%H%M)_sinnonhyeon_gangnam"
if [ -n "$LATEST" ] && [ -d "$LATEST" ] && [ "$LATEST" != "$EXPDIR" ]; then
  mv "$LATEST" "$EXPDIR"
  echo "[supervisor] moved $LATEST → $EXPDIR" | tee -a "$LOG"
else
  EXPDIR="$LATEST"
fi
mkdir -p "$EXPDIR"

# ── 5. 고해상도 PNG / 카메라 GIF 생성 ──────────────────────────────────────────
echo "[supervisor] hires PNGs $(date)" | tee -a "$LOG"
$PY util/gen_gangnam_hires.py "$EXPDIR" >> "$LOG" 2>&1

# ── 6. 비교 GIF — shortest vs rl_signal_attention OD-1 peak ────────────────────
echo "[supervisor] comparison GIF $(date)" | tee -a "$LOG"
$PY simulation.py \
  --config "$CFG" \
  --models shortest_dijkstra rl_signal_attention \
  --route od1_sinnonhyeon_suseo --time_slot peak \
  --camera fit --gif_only \
  --gif_name "gangnam_short_rlattn_sinnonhyeon" >> "$LOG" 2>&1
# simulation.py 는 output/gif/ 에 저장 → EXPDIR 로 복사
if ls output/gif/*sinnonhyeon* >/dev/null 2>&1; then
  cp output/gif/*sinnonhyeon* "$EXPDIR/" 2>/dev/null
fi

# ── 7. 종료 마커 ──────────────────────────────────────────────────────────────
touch "$EXPDIR/.SUPERVISOR_DONE"
echo "[supervisor] ALL DONE $(date) → $EXPDIR" | tee -a "$LOG"
