"""
experiments/visualize.py
------------------------
학습 곡선 + 경로 시각화.

단독 실행:
  python experiments/visualize.py --mode learning   # 학습 곡선
  python experiments/visualize.py --mode route      # 경로 비교
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def plot_learning_curves(model_dir: str = "models"):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")   # 서버 환경 대응

    model_dir = Path(model_dir)
    histories = list(model_dir.glob("*_history.json"))
    if not histories:
        print("학습 이력 파일 없음.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for h_path in sorted(histories):
        with open(h_path, encoding="utf-8") as f:
            data = json.load(f)
        # 새 포맷: {"metadata":..., "history":[...]}  / 구 포맷: [...] 직접
        hist = data["history"] if isinstance(data, dict) and "history" in data else data
        name = h_path.stem.replace("_history", "")
        eps  = [h["episode"] for h in hist]
        rews = [h["reward"]  for h in hist]
        fuls = [h["fuel"]    for h in hist]

        # 이동 평균 (window=50)
        w = 50
        def ma(x):
            return [sum(x[max(0,i-w):i+1])/len(x[max(0,i-w):i+1])
                    for i in range(len(x))]

        axes[0].plot(eps, ma(rews), label=name, linewidth=1.5)
        axes[1].plot(eps, ma(fuls), label=name, linewidth=1.5)

    axes[0].set(title="Episode Reward (MA-50)", xlabel="Episode", ylabel="Reward")
    axes[1].set(title="Episode Fuel (MA-50)",   xlabel="Episode", ylabel="Fuel (mL)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    out = Path("output") / "learning_curves.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"학습 곡선 저장 → {out}")


def plot_routes(topology_path: str, model_dir: str = "models",
                start: str | None = None, goal: str | None = None,
                cfg_path: str = "config/config.yaml",
                filter_models: list[str] | None = None):
    """
    filter_models: 표시할 모델 키 목록 (None이면 전체).
                   키: shortest, static_fuel, rl_base, rl_signal, rl_attn
    """
    import matplotlib.pyplot as plt
    import matplotlib
    import yaml
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    from util.environment import RoadNetworkEnv
    from util.agent import DQNAgent
    from util.dijkstra_models import ShortestDijkstra, StaticFuelDijkstra
    from util.viz_scale import viz_params

    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    env_all  = RoadNetworkEnv(topology_path, cfg["data"]["speed"],
                              reward_cfg=cfg["reward"], use_signal=True)
    env_base = RoadNetworkEnv(topology_path, cfg["data"]["speed"],
                              reward_cfg=cfg["reward"], use_signal=False)

    if not start:
        start = env_all.default_start
    if not goal:
        goal  = env_all.default_goals[0]

    def get_path(model, env, s, g):
        env.reset(start_node=s, goal_nodes=[g])
        path = [s]
        state = env._get_state()
        for _ in range(env.max_steps):
            valid = env.get_valid_actions()
            if not valid: break
            action = model.act(state, valid)
            state, _, done, _ = env.step(action)
            path.append(env.current_node)
            if done: break
        return path

    node_list = sorted(env_all.nodes.keys())
    tc = cfg["train"]

    ALL_META = {
        "shortest":    ("① Shortest Dijkstra",    ShortestDijkstra(env_all),  env_all,  "#2176e8"),
        "static_fuel": ("② StaticFuel Dijkstra",  StaticFuelDijkstra(env_all),env_all,  "#e87a21"),
        "rl_base":     ("③ RL Base",              None,                        env_base, "#27a85e"),
        "rl_signal":   ("④ RL Signal",            None,                        env_all,  "#d43535"),
        "rl_attn":     ("⑤ RL Signal+Attn",       None,                        env_all,  "#8e4fcf"),
    }
    rl_cfg = [
        ("rl_base",   "base",      env_base, "model_rl_base.pth"),
        ("rl_signal", "signal",    env_all,  "model_rl_signal.pth"),
        ("rl_attn",   "attention", env_all,  "model_rl_signal_attention.pth"),
    ]
    for key, mode, env_r, pth_name in rl_cfg:
        agent = DQNAgent(mode=mode, epsilon=0.0, epsilon_decay=1.0, lr=tc["lr"])
        pth = Path(model_dir) / pth_name
        if pth.exists():
            agent.load(str(pth))
        ALL_META[key] = (ALL_META[key][0], agent, env_r, ALL_META[key][3])

    keys = filter_models if filter_models else list(ALL_META.keys())
    suffix = "_".join(keys) if filter_models else "all"

    # 자동 스케일 — 36 노드 ~ 1000+ 노드 호환
    vp = viz_params(env_all.N, env_all.map_diag)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#f4f6fb")
    ax.set_facecolor("#eef0f5")

    # 도로망
    for lk in env_all.links.values():
        p1 = env_all.nodes[lk["end1"]]["pos"]
        p2 = env_all.nodes[lk["end2"]]["pos"]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color="#d0d3e0", lw=vp["link_lw"], zorder=1)

    # 신호 노드
    for nid, nd in env_all.nodes.items():
        if nd.get("signal"):
            has_lt = any(p["type"] in ("left_turn", "left") for p in nd["signal"]["phases"])
            c  = "#1e90ff" if has_lt else "#16c45e"
            s_ = vp["node_size_lt"] * 0.55 if has_lt else vp["node_size_signal"] * 0.55
            ax.scatter(*nd["pos"], c=c, s=s_, zorder=2, alpha=0.6)
        else:
            ax.scatter(*nd["pos"], c="#c0c4d6", s=vp["node_size_nosig"], zorder=2)

    # 경로 (굵은 흰 테두리 + 모델 컬러)
    for key in keys:
        label, model, env_r, color = ALL_META[key]
        if model is None:
            continue
        path = get_path(model, env_r, start, goal)
        xs = [env_all.nodes[n]["pos"][0] for n in path if n in env_all.nodes]
        ys = [env_all.nodes[n]["pos"][1] for n in path if n in env_all.nodes]
        ax.plot(xs, ys, lw=vp["path_bg_lw"] + 1.5, color="white", alpha=0.9, zorder=3,
                solid_capstyle="round", solid_joinstyle="round")
        ax.plot(xs, ys, lw=vp["path_lw"] + 0.7, color=color, alpha=0.85, zorder=4,
                solid_capstyle="round", solid_joinstyle="round", label=label)

    # 출발/도착 마커
    sp = env_all.nodes.get(start, {}).get("pos", [0, 0])
    gp = env_all.nodes.get(goal,  {}).get("pos", [0, 0])
    star_s = max(vp["star_ms"], 200)
    ax.scatter(*sp, c="#16c45e", s=star_s, marker="*", zorder=9, label=f"출발 {start}")
    ax.scatter(*gp, c="#f5a623", s=star_s, marker="*", zorder=9, label=f"도착 {goal}")

    ax.legend(loc="upper left", fontsize=10, facecolor="white",
              edgecolor="#dde0ea", framealpha=0.95)
    ax.set_title(f"경로 비교  {start} → {goal}", fontsize=13, fontweight="bold",
                 color="#22242a", pad=12)
    ax.set_aspect("equal")
    ax.axis("off")

    out = Path("output") / f"route_{start}_{goal}_{suffix}.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"경로 비교 저장 → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",     default="learning",
                        choices=["learning", "route"])
    parser.add_argument("--topology", default="data/6x6_topology.json")
    parser.add_argument("--start",    default=None)
    parser.add_argument("--goal",     default=None)
    parser.add_argument("--models",   nargs="+", default=None,
                        help="표시할 모델 키 (shortest static_fuel rl_base rl_signal rl_attn)")
    args = parser.parse_args()

    if args.mode == "learning":
        plot_learning_curves()
    else:
        plot_routes(args.topology, start=args.start, goal=args.goal,
                    filter_models=args.models)
