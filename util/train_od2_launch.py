"""강남구 OD-2 재실험 RL 3종 병렬 학습 런처 — config/config_gangnam_od2.yaml.

  venv/bin/python util/train_od2_launch.py

base / signal / attention 동시 학습 (서브프로세스). 각 프로세스 OMP 8스레드 제한
→ 24코어에서 3프로세스 과점유 없이. 로그: output/train_logs/<model>.log
"""
from __future__ import annotations
import os, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
LOGDIR = ROOT / "output" / "train_logs"
LOGDIR.mkdir(parents=True, exist_ok=True)
CFG = "config/config_gangnam_od2.yaml"

SPECS = [
    ("base",      "False", "model_rl_base"),
    ("signal",    "True",  "model_rl_signal"),
    ("attention", "True",  "model_rl_signal_attention"),
]

procs = []
for mode, sig, name in SPECS:
    env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               OPENBLAS_NUM_THREADS="8")
    code = ("from train._train_common import train_rl; "
            f"train_rl('{mode}', {sig}, '{CFG}', '{name}')")
    log = open(LOGDIR / f"{name}.log", "w", encoding="utf-8")
    p = subprocess.Popen(["venv/bin/python", "-u", "-c", code],
                         stdout=log, stderr=subprocess.STDOUT, env=env)
    procs.append((name, p, log))
    print(f"[launch] {name}  pid={p.pid}", flush=True)

t0 = time.time()
for name, p, log in procs:
    p.wait()
    log.close()
    print(f"[done]   {name}  rc={p.returncode}  t={time.time()-t0:.0f}s", flush=True)
print(f"[ALL DONE] {time.time()-t0:.0f}s", flush=True)
# 학습 완료 마커 → supervisor/모니터링용
(ROOT / "output" / "train_logs" / ".OD2_TRAIN_DONE").write_text(
    f"done {time.time():.0f}\n")
