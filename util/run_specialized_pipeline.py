"""OD1·OD2 특화모델(단일 OD 학습) — 학습 → 실험 → PNG → 표 → DONE."""
import os, subprocess, time, shutil, pickle, yaml, csv, io, contextlib, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent; os.chdir(ROOT)
LOG = ROOT / "output" / "train_logs"; LOG.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "output" / (time.strftime("%d_%H%M") + "_specialized"); OUT.mkdir(parents=True, exist_ok=True)

def log(m):
    print(m, flush=True); open(OUT / "_pipeline.log", "a").write(m + "\n")

log(f"[pipeline] OUT={OUT}")

# 1) 4개 특화모델 병렬 학습 (od1/od2 × base/signal)
SPECS = [("base", "False", "od1", "config/config_od1.yaml"),
         ("signal", "True", "od1", "config/config_od1.yaml"),
         ("base", "False", "od2", "config/config_od2.yaml"),
         ("signal", "True", "od2", "config/config_od2.yaml")]
procs = []
for mode, sig, od, cfg in SPECS:
    name = f"{od}_{mode}"
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    code = f"from train._train_common import train_rl; train_rl('{mode}', {sig}, '{cfg}', '{name}', episodes_override=None)"
    f = open(LOG / f"sp_{name}.log", "w")
    p = subprocess.Popen(["venv/bin/python", "-u", "-c", code], stdout=f, stderr=subprocess.STDOUT, env=env)
    procs.append((name, p, f)); log(f"[train] {name} pid={p.pid}")
t0 = time.time()
for name, p, f in procs:
    p.wait(); f.close(); log(f"[train done] {name} rc={p.returncode} t={time.time()-t0:.0f}s")

# best → final (특화 이름)
for od in ["od1", "od2"]:
    for mode in ["base", "signal"]:
        b = ROOT / "models" / f"{od}_{mode}_best.pth"
        if b.exists():
            shutil.copy(b, ROOT / "models" / f"{od}_{mode}.pth")
log("[best->final] done")

import sys; sys.path.insert(0, str(ROOT))
from util.environment import RoadNetworkEnv
from util.dijkstra_models import ShortestDijkstra, StaticFuelDijkstra
from experiments.run_experiment import main as run_exp

# 2) OD별 실험 (특화모델을 model_rl_base/signal 로 스왑 후 실행)
def run_od(od, cfg_path):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    e = RoadNetworkEnv(cfg['data']['topology'], cfg['data']['speed'], reward_cfg=cfg['reward'], use_signal=True)
    pickle.dump(ShortestDijkstra(e), open(ROOT / "models/model_shortest_dijkstra.pkl", "wb"))
    pickle.dump(StaticFuelDijkstra(e), open(ROOT / "models/model_static_fuel_dijkstra.pkl", "wb"))
    shutil.copy(ROOT / f"models/{od}_base.pth", ROOT / "models/model_rl_base.pth")
    shutil.copy(ROOT / f"models/{od}_signal.pth", ROOT / "models/model_rl_signal.pth")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_exp(cfg_path)
    out = buf.getvalue(); open(OUT / f"_exp_{od}.log", "w").write(out)
    m = re.findall(r"output/(\d{2}_\d{4})", out)
    if m:
        src = ROOT / "output" / m[-1]
        if (src / "results.csv").exists():
            shutil.copy(src / "results.csv", OUT / f"results_{od}.csv")
    log(f"[experiment] {od} done")
    # PNG
    penv = dict(os.environ, GANGNAM_CFG=cfg_path)
    subprocess.run(["venv/bin/python", "util/gen_gangnam_hires.py", str(OUT / f"routes_{od}")],
                   env=penv, stdout=open(OUT / f"_png_{od}.log", "w"), stderr=subprocess.STDOUT)
    log(f"[png] {od} done")

run_od("od1", "config/config_od1.yaml")
run_od("od2", "config/config_od2.yaml")

# 3) 표 (연료·시간, vs① %)
def tbl():
    lab = {"shortest_dijkstra": "① 거리최단", "static_fuel_dijkstra": "② 연료최소(오라클)",
           "rl_base": "③ rl_base(특화)", "rl_signal": "④ rl_signal(특화)"}
    order = ["shortest_dijkstra", "static_fuel_dijkstra", "rl_base", "rl_signal"]
    lines = ["# OD1·OD2 특화모델 실험 (v3 환경 258d, peak 30회)", "",
             "단일 OD 학습 특화모델. 시간=주행+신호대기, 괄호=① 거리최단 대비 %.", ""]
    out_rows = {}
    for od, label in [("od1", "OD1 신논현→수서"), ("od2", "OD2 양재→영동")]:
        rc = OUT / f"results_{od}.csv"
        if not rc.exists(): continue
        rows = list(csv.DictReader(open(rc)))
        def fn(r, k):
            try: return float(r[k])
            except: return float('nan')
        def tot(r): return fn(r, "travel_time") + fn(r, "wait_time")
        by = defaultdict(list)
        for r in rows: by[r["model"]].append(r)
        def ms(xs):
            xs = [x for x in xs if x == x]; m = sum(xs) / len(xs)
            return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** .5
        bf = ms([fn(r, "fuel_total") for r in by["shortest_dijkstra"]])[0]
        bt = ms([tot(r) for r in by["shortest_dijkstra"]])[0]
        out_rows[od] = {}
        for m in order:
            if not by.get(m): continue
            fm, fs = ms([fn(r, "fuel_total") for r in by[m]])
            tm, _ = ms([tot(r) for r in by[m]])
            out_rows[od][m] = (fm, fs, (fm-bf)/bf*100, tm, (tm-bt)/bt*100)
    lines.append("| 모델 | OD1 연료(mL) | OD1 시간(s) | OD2 연료(mL) | OD2 시간(s) |")
    lines.append("|---|---|---|---|---|")
    for m in order:
        c = [lab[m]]
        for od in ["od1", "od2"]:
            r = out_rows.get(od, {}).get(m)
            if r:
                fm, fs, vf, tm, vt = r
                vfs = "—" if abs(vf) < 0.05 else f"{vf:+.1f}%"
                vts = "—" if abs(vt) < 0.05 else f"{vt:+.1f}%"
                c += [f"{fm:.0f}±{fs:.0f} ({vfs})", f"{tm:.0f} ({vts})"]
            else:
                c += ["—", "—"]
        lines.append("| " + " | ".join(c) + " |")
    return "\n".join(lines)

open(OUT / "report.md", "w", encoding="utf-8").write(tbl())
log(f"[report] {OUT/'report.md'}")
open(OUT / "DONE", "w").write("ok")
log("[PIPELINE COMPLETE]")
