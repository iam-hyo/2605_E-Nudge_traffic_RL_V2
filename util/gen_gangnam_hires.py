"""강남구 고해상도 산출물 생성기 — config_gangnam.yaml 의 OD 3쌍에 대해
정적 PNG(전체도 + 줌 인셋) + 카메라 팬 GIF 생성.

방법론:
  · 미터 투영 (위경도 → 등거리 평면 m) — `gangnam_hires_viz.GangnamMap`
  · 자유 확대 — `render_window(center, half_width)` 매 영역 풀해상도 렌더
  · 카메라 자동 이동 GIF — Dijkstra 경로 따라 반경 ~240m 추적

회전제한 인지(원인 ④ 수정 후) ShortestDijkstra 와 StaticFuelDijkstra 두
경로를 함께 표시해 "거리최단 vs 연료최단" 분기를 한눈에.

사용: venv/bin/python util/gen_gangnam_hires.py [out_dir]
"""
from __future__ import annotations
import sys, json, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import yaml

from util.environment import RoadNetworkEnv
from util.dijkstra_models import ShortestDijkstra, StaticFuelDijkstra
from util.gangnam_hires_viz import (GangnamMap, render_window,
                                     figure_fullmap, camera_gif,
                                     _PIL)
import matplotlib.pyplot as plt


def _planner_path(env: RoadNetworkEnv, start: str, goal: str, hour: float,
                  planner_cls) -> list[str]:
    """planner 가 따라가는 노드 경로(현실 주행 가능 경로)."""
    env.max_steps = 400
    state = env.reset(start_node=start, goal_nodes=[goal], start_hour=hour)
    M = planner_cls(env)
    path = [start]
    done = False; steps = 0
    while not done and steps < 400:
        v = env.get_valid_actions()
        if not v:
            break
        a = M.act(state, v)
        state, _, done, _ = env.step(a)
        path.append(env.current_node)
        steps += 1
    return path


def figure_od_fullmap(gm: GangnamMap, route_short, route_fuel,
                      start, goal, title, out_path):
    """OD 한 쌍의 전체도 — 거리최단(파랑) + 연료최단(주황) 경로 동시 표시."""
    fig = plt.figure(figsize=(13, 13), dpi=200)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.92])
    xs = [p[0] for p in gm.xy.values()]
    ys = [p[1] for p in gm.xy.values()]
    cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
    half = max(max(xs)-min(xs), max(ys)-min(ys)) / 2 + 200

    # 베이스 그림 — render_window 가 링크/start/goal 까지
    render_window(ax, gm, cx, cy, half, route=route_short,
                  start=start, goal=goal, title=title)
    # fuel 경로 추가 (주황 점선)
    fx = [gm.xy[n][0] for n in route_fuel if n in gm.xy]
    fy = [gm.xy[n][1] for n in route_fuel if n in gm.xy]
    ax.plot(fx, fy, "--", color="#e8821f", lw=2.4, zorder=7,
            dashes=(4, 2), label="신호준수 연료최단 (StaticFuelDijkstra)")
    # 범례 다시 그리기 — render_window 의 라벨 + fuel 경로
    handles = [
        plt.Line2D([], [], color="#2176e8", lw=3.2,
                   label="신호준수 거리최단 (ShortestDijkstra)"),
        plt.Line2D([], [], color="#e8821f", lw=2.4, linestyle="--",
                   label="신호준수 연료최단 (StaticFuelDijkstra)"),
        plt.Line2D([], [], marker="*", color="#16a858", ms=14, ls="none",
                   label="출발"),
        plt.Line2D([], [], marker="*", color="#e4a017", ms=14, ls="none",
                   label="도착"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.95)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def figure_od_zoom3(gm: GangnamMap, route_short, route_fuel,
                    start, goal, title, out_path):
    """경로 시작/중간/종료 3구간 줌 인셋 — 분기 위치를 선명히."""
    pts_s = [gm.xy[n] for n in route_short if n in gm.xy]
    pts_f = [gm.xy[n] for n in route_fuel if n in gm.xy]
    centers = [pts_s[0], pts_s[len(pts_s)//2], pts_s[-1]]
    labels  = ["출발 부근", "경로 중간", "도착 부근"]
    fig = plt.figure(figsize=(15, 5.4), dpi=200)
    for k, ((cx, cy), lbl) in enumerate(zip(centers, labels)):
        ax = fig.add_subplot(1, 3, k + 1)
        render_window(ax, gm, cx, cy, 280, route=route_short,
                      start=start, goal=goal)
        fx = [p[0] for p in pts_f]; fy = [p[1] for p in pts_f]
        ax.plot(fx, fy, "--", color="#e8821f", lw=2.4, zorder=7, dashes=(4, 2))
        ax.set_title(f"{lbl}", fontsize=11, fontweight="bold", color="#22242a",
                     pad=6)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def main(out_dir: str):
    cfg = yaml.safe_load(open(ROOT / "config" / "config_gangnam.yaml",
                              encoding="utf-8"))
    routes = cfg["experiments"]["routes"]
    env = RoadNetworkEnv(cfg["data"]["topology"], cfg["data"]["speed"],
                         reward_cfg=cfg["reward"], use_signal=True)
    gm = GangnamMap(str(ROOT / cfg["data"]["topology"]))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    for r in routes:
        s, g, name = r["start"], r["goal"], r["name"]
        title = f"강남구 OD — {name}: {s} → {g}"
        print(f"\n[{name}] {s} → {g}")
        rs = _planner_path(env, s, g, 8.0, ShortestDijkstra)
        rf = _planner_path(env, s, g, 8.0, StaticFuelDijkstra)
        print(f"  shortest path len={len(rs)} ; fuelTDD path len={len(rf)}")
        figure_od_fullmap(gm, rs, rf, s, g, title,
                          str(out / f"gn_{name}_fullmap.png"))
        figure_od_zoom3(gm, rs, rf, s, g, title,
                        str(out / f"gn_{name}_zoom3.png"))
        # 카메라 팬 GIF (경로 따라가는 풀해상도 카메라) — 무거우니 OD-1 만
        if name == routes[0]["name"]:
            print(f"  [camera GIF] {name} …")
            camera_gif(gm, rs, str(out / f"gn_{name}_camera.gif"),
                       agent_path=rf, start=s, goal=g, px=900, dpi=110, fps=20)

    # 방법론 데모 — pixel compare (기존 72dpi vs 본 방법론)
    from util.gangnam_hires_viz import figure_pixel_compare
    r0 = routes[0]; rs0 = _planner_path(env, r0["start"], r0["goal"], 8.0,
                                         ShortestDijkstra)
    figure_pixel_compare(gm, rs0, r0["start"], r0["goal"],
                         str(out / "gn_pixel_compare.png"))


if __name__ == "__main__":
    od = sys.argv[1] if len(sys.argv) > 1 else \
         str(ROOT / "output" / (datetime.datetime.now().strftime("%d_%H%M")
                                + "_gn_hires"))
    main(od)
