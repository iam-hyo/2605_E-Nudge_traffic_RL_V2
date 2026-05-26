"""강남구 RL 3종 병렬 학습 런처 — config/config_gangnam.yaml 사용.

  venv/bin/python util/train_gangnam_launch.py

base / signal / attention 을 동시에(서브프로세스) 학습한다. 각 프로세스는
OMP 스레드를 8개로 제한해 24코어에서 3프로세스가 과점유 없이 돌게 한다.
로그: output/train_logs/<model>.log
"""
from __future__ import annotations
import os, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
LOGDIR = ROOT / "output" / "train_logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

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
            f"train_rl('{mode}', {sig}, 'config/config_gangnam.yaml', '{name}')")
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
