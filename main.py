"""
main.py
-------
전체 파이프라인 통합 실행.

  python main.py --step all          # 데이터 생성 → 학습 → 실험 → 시각화
  python main.py --step data         # 데이터 생성만
  python main.py --step train        # 학습만
  python main.py --step experiment   # 실험만
  python main.py --step visualize    # 시각화만

모델 선택:
  python main.py --step train --models rl_signal rl_signal_attention
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def step_data(cfg: dict):
    print("\n[1/4] 데이터 생성")
    from util.generate_data import generate_topology, generate_speed_csv
    topo = generate_topology(cfg["data"]["topology"])
    generate_speed_csv(topo, cfg["data"]["speed"])

    # config의 routes PLACEHOLDER를 실제 노드 ID로 자동 업데이트
    import json, math
    with open(cfg["data"]["topology"]) as f:
        t = json.load(f)
    meta = t["metadata"]
    grid = meta["grid_size"]
    nodes_sorted = sorted(t["nodes"], key=lambda n: n["id"])

    # 단거리: (0,0)→(1,2), (2,0)→(0,2)
    # 장거리: (0,0)→(9,9), (0,9)→(9,0)
    def nid(r, c):
        return str(100101 + r * grid + c)

    routes_update = [
        {"name":"short_01","start":nid(0,0),"goal":nid(1,2),"type":"short"},
        {"name":"short_02","start":nid(2,0),"goal":nid(0,2),"type":"short"},
        {"name":"long_01", "start":nid(0,0),"goal":nid(9,9),"type":"long"},
        {"name":"long_02", "start":nid(0,9),"goal":nid(9,0),"type":"long"},
    ]
    cfg["experiments"]["routes"] = routes_update
    with open("config/config.yaml", "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    print("  config.yaml routes 자동 업데이트 완료.")


def step_train(cfg: dict, models: list[str] | None = None):
    print("\n[2/4] 모델 학습")
    flags = cfg["experiments"].get("models", {})

    all_models = [
        ("shortest_dijkstra",    _train_dijkstra_shortest),
        ("static_fuel_dijkstra", _train_dijkstra_fuel),
        ("rl_base",              lambda c: _train_rl("base",      False, c)),
        ("rl_signal",            lambda c: _train_rl("signal",    True,  c)),
        ("rl_signal_attention",  lambda c: _train_rl("attention", True,  c)),
    ]

    for name, fn in all_models:
        if models and name not in models:
            continue
        if not flags.get(name, True):
            print(f"  [SKIP] {name}")
            continue
        print(f"\n  ▶ {name}")
        fn(cfg)


def _train_dijkstra_shortest(cfg):
    import pickle
    from util.environment import RoadNetworkEnv
    from util.dijkstra_models import ShortestDijkstra
    env = RoadNetworkEnv(cfg["data"]["topology"], cfg["data"]["speed"],
                         reward_cfg=cfg["reward"])
    m = ShortestDijkstra(env)
    out = Path(cfg["output"]["model_dir"])
    out.mkdir(exist_ok=True)
    with open(out / "model_shortest_dijkstra.pkl", "wb") as f:
        pickle.dump(m, f)
    print("    저장 완료.")


def _train_dijkstra_fuel(cfg):
    import pickle
    from util.environment import RoadNetworkEnv
    from util.dijkstra_models import StaticFuelDijkstra
    env = RoadNetworkEnv(cfg["data"]["topology"], cfg["data"]["speed"],
                         reward_cfg=cfg["reward"])
    m = StaticFuelDijkstra(env)
    out = Path(cfg["output"]["model_dir"])
    out.mkdir(exist_ok=True)
    with open(out / "model_static_fuel_dijkstra.pkl", "wb") as f:
        pickle.dump(m, f)
    print("    저장 완료.")


def _train_rl(mode: str, use_signal: bool, cfg: dict):
    from train._train_common import train_rl
    name_map = {"base": "model_rl_base",
                "signal": "model_rl_signal",
                "attention": "model_rl_signal_attention"}
    train_rl(mode=mode, use_signal=use_signal,
             save_name=name_map[mode])


def step_experiment(cfg: dict):
    print("\n[3/4] 실험 수행")
    from experiments.run_experiment import main as run_exp
    run_exp()


def step_visualize(cfg: dict):
    print("\n[4/4] 시각화")
    from experiments.visualize import plot_learning_curves, plot_routes
    plot_learning_curves()
    routes = cfg["experiments"]["routes"]
    if routes:
        r = routes[-1]   # 장거리 경로로 시각화
        if "PLACEHOLDER" not in r["start"]:
            plot_routes(cfg["data"]["topology"],
                        start=r["start"], goal=r["goal"])


def main():
    parser = argparse.ArgumentParser(description="Traffic RL Pipeline")
    parser.add_argument("--step", default="all",
                        choices=["all", "data", "train", "experiment", "visualize"])
    parser.add_argument("--models", nargs="*", default=None,
                        help="학습할 모델 이름 (공백 구분). 미지정 시 config 기준.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8", encoding="utf-8"))

    steps = {
        "data":       [step_data],
        "train":      [step_train],
        "experiment": [step_experiment],
        "visualize":  [step_visualize],
        "all":        [step_data, step_train, step_experiment, step_visualize],
    }[args.step]

    for fn in steps:
        if fn == step_train:
            fn(cfg, models=args.models)
        else:
            fn(cfg)

    print("\n✓ 완료")


if __name__ == "__main__":
    main()
