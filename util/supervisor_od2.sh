#!/usr/bin/env bash
# 강남구 OD-2 재실험 supervisor — 학습 완료 → 실험 → 시각화 → GIF → 학습곡선 → 마커.
# nohup 으로 띄우면 VSCode/Claude 연결이 끊겨도 서버에서 끝까지 자동 완성.
# 로그: output/train_logs/_supervisor_od2.log
set -u
cd /home/hjj/ms_ENudge/2605_E-Nudge_traffic_RL_V2
LOG="output/train_logs/_supervisor_od2.log"
PY="venv/bin/python"
CFG="config/config_gangnam_od2.yaml"
export GANGNAM_CFG="$CFG"
mkdir -p output/train_logs
echo "[supervisor-od2] start $(date)" > "$LOG"

# ── 1. 학습 완료 대기 (.OD2_TRAIN_DONE 마커 또는 launcher PID) ──────────────────
DONE_MARKER="output/train_logs/.OD2_TRAIN_DONE"
echo "[supervisor-od2] waiting for training done marker $DONE_MARKER" | tee -a "$LOG"
while [ ! -f "$DONE_MARKER" ]; do
  sleep 120
  # launcher 프로세스가 죽었는데 마커가 없으면(크래시) 모델 존재로 판단
  if ! pgrep -f "train_od2_launch" >/dev/null 2>&1; then
    if [ -f models/model_rl_base.pth ] && [ -f models/model_rl_signal.pth ] \
       && [ -f models/model_rl_signal_attention.pth ]; then
      echo "[supervisor-od2] launcher gone + 3 models present → proceed" | tee -a "$LOG"
      break
    fi
  fi
  echo "[supervisor-od2] $(date +%H:%M) training in progress" >> "$LOG"
done
echo "[supervisor-od2] training done $(date)" | tee -a "$LOG"

# ── 2. 모델 검증 ───────────────────────────────────────────────────────────────
for m in model_rl_base model_rl_signal model_rl_signal_attention; do
  if [ ! -f "models/${m}.pth" ]; then
    echo "[ERROR] models/${m}.pth missing — abort" | tee -a "$LOG"; exit 1
  fi
done
echo "[supervisor-od2] all 3 RL models present" | tee -a "$LOG"

# ── 3. 실험 (600 runs: 2 OD × 2 slot × 30 × 5) ────────────────────────────────
echo "[supervisor-od2] running experiment $(date)" | tee -a "$LOG"
$PY -c "from experiments.run_experiment import main; main('$CFG')" >> "$LOG" 2>&1

# ── 4. 출력 폴더 descriptive name 으로 이동 ───────────────────────────────────
LATEST=$(ls -td output/[0-9][0-9]_[0-9][0-9][0-9][0-9] 2>/dev/null | head -1)
EXPDIR="output/$(date +%d_%H%M)_dogok_yeongdong_od2"
if [ -n "$LATEST" ] && [ -d "$LATEST" ] && [ "$LATEST" != "$EXPDIR" ]; then
  mv "$LATEST" "$EXPDIR"; echo "[supervisor-od2] moved $LATEST → $EXPDIR" | tee -a "$LOG"
else
  EXPDIR="$LATEST"
fi
mkdir -p "$EXPDIR"
echo "$EXPDIR" > output/train_logs/.OD2_EXPDIR

# ── 5. 학습곡선 그래프 ─────────────────────────────────────────────────────────
echo "[supervisor-od2] learning curves $(date)" | tee -a "$LOG"
$PY util/plot_learning_od2.py "$EXPDIR" >> "$LOG" 2>&1

# ── 6. 고해상도 경로 PNG (OD-2 + OD-1) ────────────────────────────────────────
echo "[supervisor-od2] hires PNGs $(date)" | tee -a "$LOG"
$PY util/gen_gangnam_hires.py "$EXPDIR" >> "$LOG" 2>&1

# ── 7. 비교 GIF — shortest vs rl_signal_attention, OD-2 peak (4배속) ──────────
echo "[supervisor-od2] comparison GIF $(date)" | tee -a "$LOG"
$PY simulation.py \
  --config "$CFG" \
  --models shortest_dijkstra rl_signal_attention \
  --route od2_dogok_yeongdong --time_slot peak \
  --camera fit --gif_only --speed 4 \
  --gif_name gangnam_short_rlattn_dogok_yeongdong >> "$LOG" 2>&1
if ls output/gif/*dogok_yeongdong* >/dev/null 2>&1; then
  cp output/gif/*dogok_yeongdong* "$EXPDIR/" 2>/dev/null
fi

# ── 8. 종료 마커 ───────────────────────────────────────────────────────────────
touch "$EXPDIR/.SUPERVISOR_DONE"
echo "[supervisor-od2] ALL DONE $(date) → $EXPDIR" | tee -a "$LOG"
