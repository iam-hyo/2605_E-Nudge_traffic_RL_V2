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
    """
    config.data.topology 경로명으로 환경 자동 분기.
      data/6x6_cross_* → util.generate_data_cross  (사선 토폴로지, 속도 재사용)
      data/6x6_*       → util.generate_data_6x6    (격자 테스트베드)
      data/10x10_*     → util.generate_data        (옛 10x10, 호환 유지)
      data/gangnam* / 그 외 → 외부 데이터 가정, 생성 스킵
    routes는 config.yaml에서 사용자가 직접 관리 (자동 덮어쓰기 안 함).
    """
    print("\n[1/4] 데이터 생성")
    topo_path = cfg["data"]["topology"]
    name = Path(topo_path).name.lower()
    if name.startswith("6x6_cross"):
        # 사선 토폴로지 — 속도 CSV 는 6x6 격자와 link_id 호환되어 재사용
        from util.generate_data_cross import generate_topology
        generate_topology(topo_path)
        print(f"  [info] 속도 파일은 재생성 없이 재사용: {cfg['data']['speed']}")
        return
    if name.startswith("6x6"):
        from util.generate_data_6x6 import generate_topology, generate_speed_csv
    elif name.startswith("10x10"):
        from util.generate_data import generate_topology, generate_speed_csv
    else:
        print(f"  [skip] {topo_path} 는 외부 데이터 — 생성 안 함")
        return
    topo = generate_topology(topo_path)
    generate_speed_csv(topo, cfg["data"]["speed"])


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

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

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
