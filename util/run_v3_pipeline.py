"""v3 경로트리 확률 State + POMDP 환경 — 원큐 파이프라인.
학습(base·signal) → best→final → dijkstra 재생성 → 실험 → 보고서 → DONE.
"""
import os, subprocess, time, shutil, pickle, yaml, json, csv, io, contextlib, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent; os.chdir(ROOT)
CFG = "config/config_gangnam_v3.yaml"
LOG = ROOT / "output" / "train_logs"; LOG.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "output" / (time.strftime("%d_%H%M") + "_v3"); OUT.mkdir(parents=True, exist_ok=True)

def log(m):
    print(m, flush=True)
    with open(OUT / "_pipeline.log", "a") as f:
        f.write(m + "\n")

log(f"[pipeline] OUT={OUT}")

# 1) 학습 (base·signal 병렬)
SPECS = [("base", "False", "model_rl_base"), ("signal", "True", "model_rl_signal"),
         ("attention", "True", "model_rl_signal_attention")]
procs = []
for mode, sig, name in SPECS:
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    code = (f"from train._train_common import train_rl; "
            f"train_rl('{mode}', {sig}, '{CFG}', '{name}', episodes_override=None)")
    f = open(LOG / f"v3_{name}.log", "w")
    p = subprocess.Popen(["venv/bin/python", "-u", "-c", code],
                         stdout=f, stderr=subprocess.STDOUT, env=env)
    procs.append((name, p, f)); log(f"[train] {name} pid={p.pid}")

t0 = time.time()
for name, p, f in procs:
    p.wait(); f.close(); log(f"[train done] {name} rc={p.returncode} t={time.time()-t0:.0f}s")

# 2) best → final
for m in ["rl_base", "rl_signal", "rl_signal_attention"]:
    b = ROOT / "models" / f"model_{m}_best.pth"
    if b.exists():
        shutil.copy(b, ROOT / "models" / f"model_{m}.pth")
log("[best->final] done")

# 3) dijkstra 재생성 (정지연료 모델 반영된 오라클)
import sys; sys.path.insert(0, str(ROOT))
from util.environment import RoadNetworkEnv
from util.dijkstra_models import ShortestDijkstra, StaticFuelDijkstra
cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
e = RoadNetworkEnv(cfg['data']['topology'], cfg['data']['speed'],
                   reward_cfg=cfg['reward'], use_signal=True)
pickle.dump(ShortestDijkstra(e), open(ROOT / "models/model_shortest_dijkstra.pkl", "wb"))
pickle.dump(StaticFuelDijkstra(e), open(ROOT / "models/model_static_fuel_dijkstra.pkl", "wb"))
log("[dijkstra] saved")

# 4) 실험
from experiments.run_experiment import main as run_exp
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    run_exp(CFG)
expout = buf.getvalue(); open(OUT / "_experiment.log", "w").write(expout)
m = re.findall(r"output/(\d{2}_\d{4})", expout)
src = None
if m:
    src = ROOT / "output" / m[-1]
    for fn in ["results.csv", "summary.json"]:
        if (src / fn).exists():
            shutil.copy(src / fn, OUT / fn)
log(f"[experiment] done (src={src})")

# 5) 보고서 (per-route KPI)
def build_report():
    rows = []
    rcsv = OUT / "results.csv"
    if not rcsv.exists():
        return "results.csv 없음"
    with open(rcsv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # group by (route, slot, model)
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r.get("route", "?"), r.get("time_slot", r.get("slot", "?")))
        agg[key][r.get("model", "?")].append(r)
    def fnum(r, k):
        try: return float(r.get(k, "nan"))
        except: return float("nan")
    def mean(xs):
        xs = [x for x in xs if x == x]
        return sum(xs) / len(xs) if xs else float("nan")
    order = ["shortest_dijkstra", "static_fuel_dijkstra", "rl_base", "rl_signal", "rl_signal_attention"]
    lines = ["# v3 경로트리 확률 State + POMDP 환경 — 실험 보고서", "",
             "## 환경 변경 (필수 기입)",
             "1. **State 254d 경로트리 확률**: 1·2·3-hop = 4·8·16 경로(분기 cap 2), "
             "엣지별 8피처(valid·mv_l/r·통과확률·평균속도·길이·예상연료·목표접근) + 시계열 parent.",
             "2. **POMDP**: State는 speed.csv 평균속도만 관측(실주행속도 은닉) → 도착시각 불확실 "
             "→ 신호 통과가 확률(pass_prob). 도착분포 N(μ,σ), σ=20%·이동시간 누적.",
             "3. **신호 정지 모델**: 대기(t_wait>0) 발생 시 감속(→0)+공회전+재가속 정지연료 부과. "
             "정지 1회≈17mL(공회전 47초) ≫ 대기시간 → 정지 회피가 연료 핵심.",
             "4. **비보호 우회전 허용**: 우회전은 적신호에도 통과(pass_prob≈1).",
             "5. base = use_signal False(통과확률1·정지연료 미인지) / signal = 전체 관측. "
             "fuel_TDD = 전지적 오라클(전체정보·결정론) 유지.", "",
             "## 경로별 KPI (30회 평균, 연료 mL)", ""]
    for (route, slot), models in sorted(agg.items()):
        base_fuel = mean([fnum(r, "fuel_total") for r in models.get("shortest_dijkstra", [])])
        lines.append(f"### {route} · {slot}")
        lines.append("| 모델 | 연료(mL) | vs최단 | 도달률 | 거리(m) | 시간(s) |")
        lines.append("|---|---|---|---|---|---|")
        for mdl in order:
            rs = models.get(mdl, [])
            if not rs: continue
            fu = mean([fnum(r, "fuel_total") for r in rs])
            rc = mean([1.0 if str(r.get("reached_goal", r.get("reached", ""))).lower() in ("true","1","1.0") else 0.0 for r in rs])
            di = mean([fnum(r, "distance") for r in rs])
            ti = mean([fnum(r, "total_time") for r in rs])
            vs = f"{(fu-base_fuel)/base_fuel*100:+.1f}%" if base_fuel == base_fuel and base_fuel else "—"
            lines.append(f"| {mdl} | {fu:.0f} | {vs} | {rc*100:.0f}% | {di:.0f} | {ti:.0f} |")
        lines.append("")
    return "\n".join(lines)

open(OUT / "report.md", "w", encoding="utf-8").write(build_report())
log(f"[report] {OUT/'report.md'}")
open(OUT / "DONE", "w").write("ok")
log("[PIPELINE COMPLETE]")
