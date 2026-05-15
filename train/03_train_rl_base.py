"""
03_train_rl_base.py
-------------------
RL Base — 신호 State 미사용.
단독 실행: python train/03_train_rl_base.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train._train_common import train_rl

if __name__ == "__main__":
    train_rl(mode="base", use_signal=False, save_name="model_rl_base")
