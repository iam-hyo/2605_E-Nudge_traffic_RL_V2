"""
02_train_static_dijkstra.py
---------------------------
정적 연료 최적 Dijkstra — Time-Dependent, 신호 대기 반영.
단독 실행: python train/02_train_static_dijkstra.py
"""
import pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from util.environment import RoadNetworkEnv
from util.dijkstra_models import StaticFuelDijkstra

def main(cfg_path="config/config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    env = RoadNetworkEnv(cfg["data"]["topology"], cfg["data"]["speed"],
                         reward_cfg=cfg["reward"])
    model = StaticFuelDijkstra(env)
    out = Path(cfg["output"]["model_dir"])
    out.mkdir(exist_ok=True)
    with open(out / "model_static_fuel_dijkstra.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Static Fuel Dijkstra 저장 완료.")
    return model

if __name__ == "__main__":
    main()
