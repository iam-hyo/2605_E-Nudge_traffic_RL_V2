"""
verify_12x12.py
---------------
12x12 토폴로지 사전 검증 — 설계 의도가 성립하는지 확인.

확인 항목
  1. 연결성 — n1 에서 전 노드 도달 가능, max degree ≤ 4
  2. 최단경로 — ShortestDijkstra 가 대각 코리도(n1→n144)를 택하는가
  3. 신호 회피 혜택 — 대각 코리도(최단·신호지옥) vs 간선 우회로의 연료 비교
     → peak 에서 우회로 연료가 코리도보다 확실히 낮아야 함.

사용: python util/verify_12x12.py
"""
from __future__ import annotations
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from util.environment import RoadNetworkEnv
from util.dijkstra_models import ShortestDijkstra

TOPO  = str(ROOT / "data" / "12x12_topology.json")
SPEED = str(ROOT / "data" / "12x12_speed_data.csv")

CORRIDOR = [13 * k + 1 for k in range(12)]                 # 주 대각 코리도
PERIM    = list(range(1, 13)) + [12 * r + 12 for r in range(1, 12)]  # 외곽 간선


def rollout(env, path, start_hour):
    env.reset(start_node=str(path[0]), goal_nodes=[str(path[-1])],
              start_hour=start_hour)
    tot = {"fuel": 0.0, "time": 0.0, "dist": 0.0, "wait": 0.0, "stops": 0}
    for nxt in path[1:]:
        _, _, done, info = env.step(str(nxt))
        tot["fuel"] += info["fuel_total"]
        tot["time"] += info["travel_time"] + info["wait_time"]
        tot["dist"] += info["distance"]
        tot["wait"] += info["wait_time"]
        if info["wait_time"] > 0.5:
            tot["stops"] += 1
    return tot


def main(n_runs: int = 40):
    env = RoadNetworkEnv(TOPO, SPEED, use_signal=True)

    # 1. 연결성
    seen, dq = {"1"}, deque(["1"])
    while dq:
        u = dq.popleft()
        for v, _ in env.adj.get(u, []):
            if v not in seen:
                seen.add(v); dq.append(v)
    deg = max(len(v) for v in env.adj.values())
    print(f"[연결성] 도달 노드 {len(seen)}/{env.N}  |  max degree {deg}")

    # 2. 최단경로
    env.reset(start_node="1", goal_nodes=["144"], start_hour=8.0)
    sd = ShortestDijkstra(env)
    prev = sd._run("1", {"144"})
    sp = sd._reconstruct(prev, "1", "144")
    sp_cost = sd._path_cost(prev, "1", "144")
    is_diag = [int(x) for x in sp] == CORRIDOR
    print(f"[최단경로] {len(sp)-1} 링크 · {sp_cost:.0f} m · "
          f"대각 코리도와 일치? {'✅' if is_diag else '❌  ' + '→'.join(sp[:6])+'...'}")

    # 3. 신호 회피 혜택
    for label, sh in (("off_peak 07:00", 7.0), ("peak 08:00", 8.0)):
        print(f"\n{'='*64}\n {label}\n{'='*64}")
        print(f"  {'경로':<22}{'거리(m)':<10}{'연료(mL)':<12}"
              f"{'시간(s)':<10}{'대기(s)':<10}{'정지'}")
        res = {}
        for name, path in (("대각 코리도(최단)", CORRIDOR),
                           ("외곽 간선 우회로", PERIM)):
            acc = {"fuel": 0.0, "time": 0.0, "dist": 0.0, "wait": 0.0, "stops": 0.0}
            for _ in range(n_runs):
                t = rollout(env, path, sh)
                for k in acc:
                    acc[k] += t[k]
            for k in acc:
                acc[k] /= n_runs
            res[name] = acc
            print(f"  {name:<22}{acc['dist']:<10.0f}{acc['fuel']:<12.1f}"
                  f"{acc['time']:<10.0f}{acc['wait']:<10.0f}{acc['stops']:.1f}")
        d = res["대각 코리도(최단)"]["fuel"] - res["외곽 간선 우회로"]["fuel"]
        pct = 100.0 * d / res["대각 코리도(최단)"]["fuel"]
        verdict = "✅ 우회로 우위" if d > 0 else "❌ 코리도가 더 저연료"
        print(f"  → 우회로가 코리도 대비 연료 {d:+.1f} mL ({pct:+.1f}%)  {verdict}")


if __name__ == "__main__":
    main()
