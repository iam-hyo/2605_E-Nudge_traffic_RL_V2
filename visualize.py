"""
visualize.py
------------
단일/복수 모델 선택 → 1회 시뮬레이션 → 꺾은선 그래프 + 경로 이미지 저장.

저장 위치: output/visualize/{YYYYMMDD}/

저장 파일:
  01_route.png       경로 시각화
  02_fuel.png        누적 연료 (mL)
  03_wait.png        누적 대기시간 (s)
  04_speed.png       스텝별 속도 (km/h)
  05_reward.png      누적 리워드
  06_reach_rate.png  도달률 막대 그래프

사용법:
  python visualize.py --models rl_base shortest_dijkstra --route long_01 --time_slot peak
  python visualize.py --models all --route short_01 --reach_trials 20
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import matplotlib
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from util.environment import RoadNetworkEnv
from util.agent import DQNAgent
from util.dijkstra_models import ShortestDijkstra, StaticFuelDijkstra

# ── 상수 ─────────────────────────────────────────────────────────────────────
MODEL_COLORS: dict[str, str] = {
    "shortest_dijkstra":    "#1f77b4",
    "static_fuel_dijkstra": "#ff7f0e",
    "rl_base":              "#2ca02c",
    "rl_signal":            "#d62728",
    "rl_signal_attention":  "#9467bd",
}
ALL_MODELS = list(MODEL_COLORS.keys())


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def _load_model(name: str, cfg: dict, env: RoadNetworkEnv):
    model_dir = Path(cfg["output"]["model_dir"])
    if name == "shortest_dijkstra":
        return ShortestDijkstra(env)
    if name == "static_fuel_dijkstra":
        return StaticFuelDijkstra(env)

    pth_map = {
        "rl_base":             "model_rl_base.pth",
        "rl_signal":           "model_rl_signal.pth",
        "rl_signal_attention": "model_rl_signal_attention.pth",
    }
    mode_map = {
        "rl_base": "base", "rl_signal": "signal",
        "rl_signal_attention": "attention",
    }
    tc    = cfg["train"]
    agent = DQNAgent(
        mode          = mode_map[name],
        gamma         = tc["gamma"],
        epsilon_min   = tc["epsilon_min"],
        epsilon       = tc["epsilon_min"],
        epsilon_decay = 1.0,
        lr            = tc["lr"],
    )
    pth = model_dir / pth_map[name]
    if pth.exists():
        agent.load(str(pth))
    else:
        print(f"  [경고] {pth.name} 없음 — 미학습 모델")
    return agent


def _make_env(cfg: dict, use_signal: bool) -> RoadNetworkEnv:
    return RoadNetworkEnv(
        cfg["data"]["topology"],
        cfg["data"]["speed"],
        reward_cfg = cfg["reward"],
        use_signal = use_signal,
    )


# ── 1회 에피소드 실행 & 트레이스 수집 ─────────────────────────────────────────
def run_episode_trace(
    model, env: RoadNetworkEnv,
    start: str, goal: str, start_hour: float,
) -> dict:
    """에피소드를 1회 실행하며 스텝별 지표를 수집."""
    state = env.reset(start_node=start, goal_nodes=[goal], start_hour=start_hour)
    trace: dict = {
        "steps":      [],
        "nodes":      [start],
        "speed_kmh":  [],
        "fuel_step":  [],
        "cum_fuel":   [],
        "wait_step":  [],
        "cum_wait":   [],
        "reward":     [],
        "cum_reward": [],
        "dist_step":  [],
        "cum_dist":   [],
        "reached":    False,
    }
    cum_fuel = cum_wait = cum_reward = cum_dist = 0.0
    done = False

    while not done:
        valid = env.get_valid_actions()
        if not valid:
            break
        action = model.act(state, valid)
        state, reward, done, info = env.step(action)

        step = len(trace["steps"]) + 1
        cum_fuel    += info.get("fuel_total",   0.0)
        cum_wait    += info.get("wait_time",    0.0)
        cum_reward  += reward
        cum_dist    += info.get("distance",     0.0)

        trace["steps"].append(step)
        trace["nodes"].append(env.current_node)
        trace["speed_kmh"].append(info.get("speed_kmh",   0.0))
        trace["fuel_step"].append(info.get("fuel_total",   0.0))
        trace["cum_fuel"].append(cum_fuel)
        trace["wait_step"].append(info.get("wait_time",   0.0))
        trace["cum_wait"].append(cum_wait)
        trace["reward"].append(reward)
        trace["cum_reward"].append(cum_reward)
        trace["dist_step"].append(info.get("distance",    0.0))
        trace["cum_dist"].append(cum_dist)
        trace["reached"] = info.get("reached_goal", False)

    return trace


# ── 경로 시각화 ───────────────────────────────────────────────────────────────
def plot_route(
    traces: dict[str, dict],
    env: RoadNetworkEnv,
    start: str, goal: str,
    outdir: Path,
):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    # 도로망
    for lk in env.links.values():
        p1 = env.nodes[str(lk["end1"])]["pos"]
        p2 = env.nodes[str(lk["end2"])]["pos"]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color="#333355", lw=1.5, zorder=1, solid_capstyle="round")

    # 노드 (신호 여부)
    for nd in env.nodes.values():
        col = "#ff6666" if nd.get("signal") else "#555577"
        ax.scatter(*nd["pos"], c=col, s=22, zorder=2, alpha=0.7, edgecolors="none")

    # 각 모델 경로
    for name, trace in traces.items():
        color  = MODEL_COLORS.get(name, "#ffffff")
        path   = [n for n in trace["nodes"] if n in env.nodes]
        xs = [env.nodes[n]["pos"][0] for n in path]
        ys = [env.nodes[n]["pos"][1] for n in path]
        label = f"{name}  ({'✓ 도달' if trace['reached'] else '✗ 미도달'})"
        ax.plot(xs, ys, lw=2.8, color=color, alpha=0.85, label=label, zorder=3)
        # 시작점 화살표
        if len(xs) >= 2:
            ax.annotate("", xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                        zorder=4)

    # 출발 / 도착
    sp = env.nodes.get(start, {}).get("pos", [0, 0])
    gp = env.nodes.get(goal,  {}).get("pos", [0, 0])
    ax.scatter(*sp, c="#00ff88", s=300, marker="*", zorder=7, label="출발")
    ax.scatter(*gp, c="#ffd700", s=300, marker="*", zorder=7, label="도착")

    ax.set_aspect("equal")
    ax.axis("off")
    leg = ax.legend(loc="upper left", fontsize=9, facecolor="#1a1a2e",
                    labelcolor="white", edgecolor="#555577", framealpha=0.9)
    ax.set_title(f"경로 비교  ({start} → {goal})", fontsize=14,
                 color="white", pad=10)

    _save(fig, outdir / "01_route.png")


# ── 꺾은선 그래프 ─────────────────────────────────────────────────────────────
def plot_line_charts(traces: dict[str, dict], outdir: Path):
    charts = [
        ("cum_fuel",   "누적 연료 (mL)",      "02_fuel.png"),
        ("cum_wait",   "누적 대기시간 (s)",    "03_wait.png"),
        ("speed_kmh",  "스텝별 속도 (km/h)",   "04_speed.png"),
        ("cum_reward", "누적 리워드",           "05_reward.png"),
    ]
    for key, ylabel, fname in charts:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#0d1117")

        has_data = False
        for name, trace in traces.items():
            if not trace["steps"]:
                continue
            color  = MODEL_COLORS.get(name, "#ffffff")
            label  = f"{name} ({'✓' if trace['reached'] else '✗'})"
            ax.plot(trace["steps"], trace[key],
                    color=color, lw=2, label=label,
                    marker=".", markersize=3, alpha=0.9)
            has_data = True

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xlabel("스텝", color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.set_title(ylabel, color="white", fontsize=13)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444466")
        ax.grid(True, alpha=0.2, color="#555577")
        leg = ax.legend(fontsize=9, facecolor="#1a1a2e",
                        labelcolor="white", edgecolor="#555577")

        _save(fig, outdir / fname)


# ── 도달률 막대 그래프 ────────────────────────────────────────────────────────
def plot_reach_rate(
    model_names: list[str],
    cfg: dict,
    route_name: str,
    time_slot: str,
    outdir: Path,
    n_trials: int = 10,
):
    routes_map = {r["name"]: r for r in cfg["experiments"]["routes"]}
    tslots_map = {t["label"]: t for t in cfg["experiments"]["time_slots"]}
    route  = routes_map[route_name]
    tslot  = tslots_map[time_slot]
    start, goal = route["start"], route["goal"]
    start_hour  = tslot["start_hour"]

    print(f"  도달률 계산 ({n_trials}회 반복)...")
    reach_rates: dict[str, float] = {}
    for name in model_names:
        env   = _make_env(cfg, use_signal=(name != "rl_base"))
        model = _load_model(name, cfg, env)
        cnt   = 0
        for _ in range(n_trials):
            state = env.reset(start_node=start, goal_nodes=[goal],
                              start_hour=start_hour)
            done = reached = False
            while not done:
                valid = env.get_valid_actions()
                if not valid:
                    break
                action = model.act(state, valid)
                state, _, done, info = env.step(action)
                reached = info.get("reached_goal", False)
            if reached:
                cnt += 1
        reach_rates[name] = cnt / n_trials
        print(f"    {name}: {cnt}/{n_trials} = {cnt/n_trials:.0%}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    names  = list(reach_rates.keys())
    rates  = [reach_rates[n] for n in names]
    colors = [MODEL_COLORS.get(n, "#ffffff") for n in names]
    x      = np.arange(len(names))

    bars = ax.bar(x, rates, color=colors, alpha=0.85,
                  edgecolor="#222244", linewidth=1.2)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9, color="white")
    ax.set_ylabel("도달률", color="white")
    ax.set_title(
        f"도달률 비교  ({route_name} | {time_slot} | {n_trials}회 평균)",
        color="white", fontsize=13,
    )
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444466")
    ax.grid(axis="y", alpha=0.2, color="#555577")

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{rate:.0%}",
                ha="center", va="bottom", fontsize=11,
                fontweight="bold", color="white")

    _save(fig, outdir / "06_reach_rate.png")


# ── 공통 저장 ─────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, path: Path):
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  저장: {path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="교통 RL 결과 시각화 (이미지 저장)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", default=["shortest_dijkstra", "rl_base"],
        help=(
            "모델 선택 (공백 구분, 'all' 로 전체)\n"
            f"  선택지: {ALL_MODELS + ['all']}"
        ),
    )
    parser.add_argument(
        "--route", default="long_01",
        choices=["short_01", "short_02", "long_01", "long_02"],
    )
    parser.add_argument(
        "--time_slot", default="off_peak",
        choices=["off_peak", "peak"],
    )
    parser.add_argument("--config",       default="config/config.yaml")
    parser.add_argument("--reach_trials", type=int, default=10,
                        help="도달률 계산 반복 횟수 (기본: 10)")
    args = parser.parse_args()

    model_names = ALL_MODELS if "all" in args.models else args.models
    invalid = [m for m in model_names if m not in ALL_MODELS]
    if invalid:
        parser.error(f"알 수 없는 모델: {invalid}")

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    # 출력 폴더: output/visualize/YYYYMMDD/
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    outdir   = Path(cfg["output"]["result_dir"]) / "visualize" / date_str
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n모델    : {model_names}")
    print(f"경로    : {args.route}  |  시간대: {args.time_slot}")
    print(f"저장위치: {outdir}\n")

    routes_map = {r["name"]: r for r in cfg["experiments"]["routes"]}
    tslots_map = {t["label"]: t for t in cfg["experiments"]["time_slots"]}
    route  = routes_map[args.route]
    tslot  = tslots_map[args.time_slot]
    start, goal = route["start"], route["goal"]
    start_hour  = tslot["start_hour"]

    # ── 1회 에피소드 트레이스 수집 ────────────────────────────────────────────
    print("1회 시뮬레이션 실행 중...")
    traces:  dict[str, dict] = {}
    ref_env: RoadNetworkEnv | None = None

    for name in model_names:
        print(f"  [{name}]")
        env   = _make_env(cfg, use_signal=(name != "rl_base"))
        if ref_env is None:
            ref_env = env
        model = _load_model(name, cfg, env)
        trace = run_episode_trace(model, env, start, goal, start_hour)
        traces[name] = trace

        if trace["steps"]:
            status = "도달" if trace["reached"] else "미도달"
            print(f"    → {status} | 스텝={len(trace['steps'])} "
                  f"| 연료={trace['cum_fuel'][-1]:.1f}mL "
                  f"| 시간={trace['cum_wait'][-1]+sum(trace['dist_step']):.0f}s")
        else:
            print("    → 첫 스텝 없음 (유효 액션 없음)")

    # ── 그래프 저장 ───────────────────────────────────────────────────────────
    print("\n그래프 저장 중...")
    plot_route(traces, ref_env, start, goal, outdir)
    plot_line_charts(traces, outdir)

    print(f"\n도달률 계산 ({args.reach_trials}회)...")
    plot_reach_rate(model_names, cfg, args.route, args.time_slot,
                    outdir, n_trials=args.reach_trials)

    print(f"\n완료! 저장 위치: {outdir}")


if __name__ == "__main__":
    main()
