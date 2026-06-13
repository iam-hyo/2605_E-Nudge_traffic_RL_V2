import os, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; os.chdir(ROOT)
LOG=ROOT/"output"/"train_logs"; LOG.mkdir(parents=True,exist_ok=True)
CFG="config/config_gangnam_urban.yaml"
SPECS=[("base","False","model_rl_base","None"),
       ("signal","True","model_rl_signal","None"),
       ("attention","True","model_rl_signal_attention","23000")]  # attn 1.5x
P=[]
for mode,sig,name,ep in SPECS:
    env=dict(os.environ,OMP_NUM_THREADS="8",MKL_NUM_THREADS="8",OPENBLAS_NUM_THREADS="8")
    code=f"from train._train_common import train_rl; train_rl('{mode}',{sig},'{CFG}','{name}',episodes_override={ep})"
    f=open(LOG/f"urb_{name}.log","w"); p=subprocess.Popen(["venv/bin/python","-u","-c",code],stdout=f,stderr=subprocess.STDOUT,env=env)
    P.append((name,p,f)); print(f"[launch] {name} pid={p.pid} ep={ep}",flush=True)
t0=time.time()
for name,p,f in P: p.wait(); f.close(); print(f"[done] {name} rc={p.returncode} t={time.time()-t0:.0f}s",flush=True)
print(f"[ALL DONE] {time.time()-t0:.0f}s",flush=True)
