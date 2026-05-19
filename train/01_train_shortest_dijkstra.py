"""
01_train_shortest_dijkstra.py
-----------------------------
최단 거리 Dijkstra — 학습 없음, 모델 객체만 생성·저장.
단독 실행: python train/01_train_shortest_dijkstra.py
"""
import pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from util.environment import RoadNetworkEnv
from util.dijkstra_models import ShortestDijkstra

def main(cfg_path="config/config.yaml"):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    env = RoadNetworkEnv(cfg["data"]["topology"], cfg["data"]["speed"],
                         reward_cfg=cfg["reward"])
    model = ShortestDijkstra(env)
    out = Path(cfg["output"]["model_dir"])
    out.mkdir(exist_ok=True)
    with open(out / "model_shortest_dijkstra.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Shortest Dijkstra 저장 완료.")
    return model

if __name__ == "__main__":
    main()
