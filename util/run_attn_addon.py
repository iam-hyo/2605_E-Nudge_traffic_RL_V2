"""attention 특화모델 보강 — od1_attn·od2_attn 학습 후 5모델 전체로 OD별 재실험."""
import os, subprocess, time, shutil, pickle, yaml, io, contextlib, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent; os.chdir(ROOT)
LOG = ROOT / "output" / "train_logs"
OUT = ROOT / "output" / "11_2225_specialized"      # 기존 특화 산출물 폴더에 합류
MAIN_DONE = OUT / "DONE"

def log(m):
    print(m, flush=True); open(OUT / "_attn.log", "a").write(m + "\n")

log("[attn] start")

# 1) attention 특화 2모델 병렬 학습
SPECS = [("od1", "config/config_od1_attn.yaml"), ("od2", "config/config_od2_attn.yaml")]
procs = []
for od, cfg in SPECS:
    name = f"{od}_attn"
    env = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3")
    code = f"from train._train_common import train_rl; train_rl('attention', True, '{cfg}', '{name}', episodes_override=None)"
    f = open(LOG / f"sp_{name}.log", "w")
    p = subprocess.Popen(["venv/bin/python", "-u", "-c", code], stdout=f, stderr=subprocess.STDOUT, env=env)
    procs.append((name, p, f)); log(f"[train] {name} pid={p.pid}")
t0 = time.time()
for name, p, f in procs:
    p.wait(); f.close(); log(f"[train done] {name} rc={p.returncode} t={time.time()-t0:.0f}s")
for od in ["od1", "od2"]:
    b = ROOT / "models" / f"{od}_attn_best.pth"
    if b.exists(): shutil.copy(b, ROOT / "models" / f"{od}_attn.pth")
log("[best->final attn] done")

# 2) 메인 파이프라인 완료 대기 (base/signal final 모델 보장)
while not MAIN_DONE.exists():
    log("[wait] 메인 파이프라인 대기..."); time.sleep(20)
time.sleep(5)

import sys; sys.path.insert(0, str(ROOT))
from util.environment import RoadNetworkEnv
from util.dijkstra_models import ShortestDijkstra, StaticFuelDijkstra
from experiments.run_experiment import main as run_exp

# 3) OD별 5모델 전체 재실험
def run_od(od, cfg_path):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    e = RoadNetworkEnv(cfg['data']['topology'], cfg['data']['speed'], reward_cfg=cfg['reward'], use_signal=True)
    pickle.dump(ShortestDijkstra(e), open(ROOT / "models/model_shortest_dijkstra.pkl", "wb"))
    pickle.dump(StaticFuelDijkstra(e), open(ROOT / "models/model_static_fuel_dijkstra.pkl", "wb"))
    shutil.copy(ROOT / f"models/{od}_base.pth", ROOT / "models/model_rl_base.pth")
    shutil.copy(ROOT / f"models/{od}_signal.pth", ROOT / "models/model_rl_signal.pth")
    shutil.copy(ROOT / f"models/{od}_attn.pth", ROOT / "models/model_rl_signal_attention.pth")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_exp(cfg_path)
    out = buf.getvalue(); open(OUT / f"_exp_{od}_full.log", "w").write(out)
    m = re.findall(r"output/(\d{2}_\d{4})", out)
    if m:
        src = ROOT / "output" / m[-1]
        if (src / "results.csv").exists():
            shutil.copy(src / "results.csv", OUT / f"results_{od}_full.csv")
    log(f"[experiment-full] {od} done")

run_od("od1", "config/config_od1_attn.yaml")
run_od("od2", "config/config_od2_attn.yaml")
open(OUT / "DONE_ATTN", "w").write("ok")
log("[ATTN ADDON COMPLETE]")
