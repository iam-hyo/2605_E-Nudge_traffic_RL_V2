"""OD-2 재실험 후보 스크리닝 — gangnam_topology_add.json (신호 강화 토폴로지).

좌하단(SW) → 우상단(NE) 방향을 유지하면서 '거리최단 ≠ 연료최단' 격차가 큰
OD 쌍을 사전 Dijkstra 로 탐색. OD-1 재실험과 동일 방법론.

각 후보쌍 (origin∈SW, dest∈NE) 에 대해 peak(08:00) 기준:
  - ShortestDijkstra 경로를 env 로 시뮬 → short_fuel/short_wait/dist/reach
  - StaticFuelDijkstra 경로를 env 로 시뮬 → fuel_tdd_fuel/wait
  - savings = (tdd-short)/short
거리최단이 신호밀집 corridor 직진 → wait/fuel 폭증, fuel TDD 우회 시 절감 큰
쌍을 선별 (RL 우위 발현 잠재력 큼).

사용: venv/bin/python util/screen_od2.py
"""
from __future__ import annotations
import sys, math, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from util.environment import RoadNetworkEnv
from util.dijkstra_models import ShortestDijkstra, StaticFuelDijkstra

TOPO  = "data/gangnam_topology_add.json"
SPEED = "data/gangnam_clean_speed_data.csv"
REWARD = {"alpha": 1.0, "arrival_bonus": 700.0, "penalty_dead": 0.0,
          "penalty_timeout": 0.0}


def simulate(env, planner, start, goal, hour, max_steps=400):
    state = env.reset(start_node=start, goal_nodes=[goal], start_hour=hour)
    M = planner(env)
    fuel = wait = dist = 0.0
    steps = 0; reached = False; path = [start]
    done = False
    while not done and steps < max_steps:
        v = env.get_valid_actions()
        if not v:
            break
        a = M.act(state, v)
        state, _, done, info = env.step(a)
        fuel += info.get("fuel_total", 0.0)
        wait += info.get("wait_time", 0.0)
        dist += info.get("distance", 0.0)
        reached = info.get("reached_goal", False)
        path.append(env.current_node)
        steps += 1
    return dict(fuel=fuel, wait=wait, dist=dist, steps=steps,
               reached=reached, path=path)


def main():
    env = RoadNetworkEnv(TOPO, SPEED, reward_cfg=REWARD, use_signal=True)
    nodes = env.nodes
    deg = {n: len(env.adj.get(n, [])) for n in nodes}

    xs = [nodes[n]["pos"][0] for n in nodes]
    ys = [nodes[n]["pos"][1] for n in nodes]
    lon0, lon1 = min(xs), max(xs)
    lat0, lat1 = min(ys), max(ys)
    # 지도가 반대각선(북=서, 동=남)이라 절대 NE 코너는 비어있음.
    # → '상대적' SW→NE: origin 은 좌하단 1/3 영역, dest 는 origin 기준 북동(Δ>0).
    # SW 영역 (좌하단, 양재·도곡 일대): lon 하위 35%, lat 하위 45%
    sw_lon = lon0 + 0.35 * (lon1 - lon0)
    sw_lat = lat0 + 0.45 * (lat1 - lat0)
    # 북동 populated 영역 (삼성·청담·영동·대치북): lon 중앙 이상 & lat 상위 35%
    ne_lon_min = lon0 + 0.30 * (lon1 - lon0)
    ne_lat_min = lat0 + 0.55 * (lat1 - lat0)

    def signal_neighbors(n):
        return sum(1 for nb, _ in env.adj.get(n, []) if nodes[nb].get("signal"))

    sw = [n for n in nodes if deg[n] >= 3
          and nodes[n]["pos"][0] <= sw_lon and nodes[n]["pos"][1] <= sw_lat]
    ne = [n for n in nodes if deg[n] >= 3
          and nodes[n]["pos"][0] >= ne_lon_min and nodes[n]["pos"][1] >= ne_lat_min]
    print(f"SW candidates={len(sw)}  NE candidates={len(ne)}")

    # 후보 과다 → 신호 인접 수 높은 노드 우선 (corridor 진입부) 상위 N
    sw.sort(key=lambda n: -signal_neighbors(n))
    ne.sort(key=lambda n: -signal_neighbors(n))
    sw = sw[:16]; ne = ne[:18]

    def haversine_km(a, b):
        (x1, y1), (x2, y2) = nodes[a]["pos"], nodes[b]["pos"]
        R = 6371.0
        p1, p2 = math.radians(y1), math.radians(y2)
        dphi = math.radians(y2 - y1); dl = math.radians(x2 - x1)
        h = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
        return 2*R*math.asin(math.sqrt(h))

    results = []
    for o in sw:
        for d in ne:
            # 상대적 북동 방향 강제 (좌하단→우상단 경향성 유지)
            dlon = nodes[d]["pos"][0] - nodes[o]["pos"][0]
            dlat = nodes[d]["pos"][1] - nodes[o]["pos"][1]
            if dlon <= 0.005 or dlat <= 0.010:
                continue
            straight = haversine_km(o, d)
            if straight < 3.0 or straight > 7.5:   # 양재→영동 4.7km 수준 유지
                continue
            s = simulate(env, ShortestDijkstra, o, d, 8.0)
            if not s["reached"]:
                continue
            f = simulate(env, StaticFuelDijkstra, o, d, 8.0)
            if not f["reached"]:
                continue
            sav = (f["fuel"] - s["fuel"]) / s["fuel"] * 100.0
            results.append(dict(
                o=o, d=d, straight_km=round(straight, 2),
                short_fuel=round(s["fuel"], 1), short_wait=round(s["wait"], 1),
                short_dist=round(s["dist"]), short_steps=s["steps"],
                tdd_fuel=round(f["fuel"], 1), tdd_wait=round(f["wait"], 1),
                tdd_dist=round(f["dist"]), savings_pct=round(sav, 1),
                short_signal=sum(1 for n in s["path"] if nodes[n].get("signal")),
                tdd_signal=sum(1 for n in f["path"] if nodes[n].get("signal")),
                path_diverge=(s["path"] != f["path"]),
            ))
    results.sort(key=lambda r: r["savings_pct"])  # 가장 음수(절감 큼)가 위
    print(f"\n총 {len(results)} 후보쌍 (도달+거리 필터 통과)\n")
    hdr = ("origin    dest      straight short_fuel short_wait  tdd_fuel tdd_wait"
           "  sav%   sigS/sigT")
    print(hdr); print("-"*len(hdr))
    for r in results[:25]:
        print(f"{r['o']:<9} {r['d']:<9} {r['straight_km']:>6}km "
              f"{r['short_fuel']:>9} {r['short_wait']:>9} "
              f"{r['tdd_fuel']:>9} {r['tdd_wait']:>8} "
              f"{r['savings_pct']:>6} {r['short_signal']:>4}/{r['tdd_signal']}")

    Path(ROOT/"output").mkdir(exist_ok=True)
    with open(ROOT/"output"/"_od2_screening.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=1)
    print("\nsaved → output/_od2_screening.json")


if __name__ == "__main__":
    main()
