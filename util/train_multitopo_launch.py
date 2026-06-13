"""train_multitopo_launch.py — 다중 토폴로지 강건성 학습 런처.

config/config_multitopo.yaml 로 base/signal/attention 3종 병렬 학습.
attention 은 공정보정(under-fit 해소)으로 episodes 1.67× (25000) override.

  venv/bin/python util/train_multitopo_launch.py

로그: output/train_logs/mt_<model>.log
"""
from __future__ import annotations
import os, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
LOGDIR = ROOT / "output" / "train_logs"
LOGDIR.mkdir(parents=True, exist_ok=True)
CFG = "config/config_multitopo.yaml"

# (mode, use_signal, save_name, episodes_override)
SPECS = [
    ("base",      "False", "model_rl_base",            "None"),
    ("signal",    "True",  "model_rl_signal",          "None"),
    ("attention", "True",  "model_rl_signal_attention","25000"),  # 1.67× 공정보정
]

procs = []
for mode, sig, name, epov in SPECS:
    env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               OPENBLAS_NUM_THREADS="8")
    code = ("from train._train_common import train_rl; "
            f"train_rl('{mode}', {sig}, '{CFG}', '{name}', episodes_override={epov})")
    log = open(LOGDIR / f"mt_{name}.log", "w", encoding="utf-8")
    p = subprocess.Popen(["venv/bin/python", "-u", "-c", code],
                         stdout=log, stderr=subprocess.STDOUT, env=env)
    procs.append((name, p, log))
    print(f"[launch] {name}  pid={p.pid}  ep_override={epov}", flush=True)

t0 = time.time()
for name, p, log in procs:
    p.wait(); log.close()
    print(f"[done]   {name}  rc={p.returncode}  t={time.time()-t0:.0f}s", flush=True)
print(f"[ALL DONE] {time.time()-t0:.0f}s", flush=True)
