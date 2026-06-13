import os, subprocess, time, shutil, pickle, yaml
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; os.chdir(ROOT)
CFG="config/config_gangnam_fuelshape.yaml"
LOG=ROOT/"output"/"train_logs"; LOG.mkdir(parents=True,exist_ok=True)
OUT=ROOT/"output"/(time.strftime("%d_%H%M")+"_fuelshape"); OUT.mkdir(parents=True,exist_ok=True)
def log(m): 
    print(m,flush=True); open(OUT/"_pipeline.log","a").write(m+"\n")
log(f"[pipeline] OUT={OUT}")
# 1) 학습 3종 병렬
SPECS=[("base","False","model_rl_base","None"),("signal","True","model_rl_signal","None"),
       ("attention","True","model_rl_signal_attention","23000")]
procs=[]
for mode,sig,name,ep in SPECS:
    env=dict(os.environ,OMP_NUM_THREADS="8",MKL_NUM_THREADS="8",OPENBLAS_NUM_THREADS="8")
    code=f"from train._train_common import train_rl; train_rl('{mode}',{sig},'{CFG}','{name}',episodes_override={ep})"
    f=open(LOG/f"fs_{name}.log","w"); p=subprocess.Popen(["venv/bin/python","-u","-c",code],stdout=f,stderr=subprocess.STDOUT,env=env)
    procs.append((name,p,f)); log(f"[train] {name} pid={p.pid}")
t0=time.time()
for name,p,f in procs: p.wait(); f.close(); log(f"[train done] {name} rc={p.returncode} t={time.time()-t0:.0f}s")
# 2) best→final
for m in ["rl_base","rl_signal","rl_signal_attention"]:
    b=ROOT/"models"/f"model_{m}_best.pth"
    if b.exists(): shutil.copy(b, ROOT/"models"/f"model_{m}.pth")
log("[best→final] done")
# 3) dijkstra pkl (clean)
import sys; sys.path.insert(0,str(ROOT))
from util.environment import RoadNetworkEnv
from util.dijkstra_models import ShortestDijkstra, StaticFuelDijkstra
cfg=yaml.safe_load(open(CFG,encoding="utf-8"))
e=RoadNetworkEnv(cfg['data']['topology'],cfg['data']['speed'],reward_cfg=cfg['reward'],use_signal=True)
pickle.dump(ShortestDijkstra(e),open(ROOT/"models/model_shortest_dijkstra.pkl","wb"))
pickle.dump(StaticFuelDijkstra(e),open(ROOT/"models/model_static_fuel_dijkstra.pkl","wb"))
log("[dijkstra] saved")
# 4) 실험
from experiments.run_experiment import main as run_exp
import io, contextlib
buf=io.StringIO()
with contextlib.redirect_stdout(buf): run_exp(CFG)
expout=buf.getvalue(); open(OUT/"_experiment.log","w").write(expout)
# run_experiment 자체 폴더 결과 복사
import re, glob
m=re.findall(r"output/(\d{2}_\d{4})", expout)
if m:
    rd=ROOT/"output"/m[-1]
    for fn in ["results.csv","summary.json"]:
        if (rd/fn).exists(): shutil.copy(rd/fn, OUT/fn)
log(f"[experiment] done → {OUT}")
open(OUT/"DONE","w").write("ok")
log("[PIPELINE COMPLETE]")
