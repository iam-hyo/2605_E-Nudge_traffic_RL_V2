"""강남 100% 단독 학습 런처 (멀티토포 weight 18% 대조군).
base/signal/attention 3개를 config_gangnam_100.yaml 로 학습, *_gn100 으로 저장.
실행: python train/_launch_gn100.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train._train_common import train_rl

CFG = "config/config_gangnam_100.yaml"
JOBS = [
    ("base",      False, "model_rl_base_gn100"),
    ("signal",    True,  "model_rl_signal_gn100"),
    ("attention", True,  "model_rl_signal_attention_gn100"),
]

if __name__ == "__main__":
    for mode, use_signal, name in JOBS:
        t0 = time.time()
        print(f"\n{'='*60}\n[GN100] 학습 시작: {mode} → {name}\n{'='*60}", flush=True)
        train_rl(mode=mode, use_signal=use_signal, cfg_path=CFG, save_name=name)
        print(f"[GN100] {mode} 완료 ({(time.time()-t0)/60:.1f}분)", flush=True)
    print("\n[GN100] 전체 완료", flush=True)
