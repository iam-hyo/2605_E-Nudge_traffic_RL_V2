"""
05_train_rl_signal_attention.py
-------------------------------
RL Signal + Attention — 전체 모델.
단독 실행: python train/05_train_rl_signal_attention.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train._train_common import train_rl

if __name__ == "__main__":
    train_rl(mode="attention", use_signal=True, save_name="model_rl_signal_attention")
