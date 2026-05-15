"""
04_train_rl_signal.py
---------------------
RL Signal — 신호 State 포함, Attention 없음.
단독 실행: python train/04_train_rl_signal.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train._train_common import train_rl

if __name__ == "__main__":
    train_rl(mode="signal", use_signal=True, save_name="model_rl_signal")
