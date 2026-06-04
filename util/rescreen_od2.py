"""OD-2 후보 재평가 — 노이즈 제거(deterministic_speed) + 다중 repeat 평균.

1차 스크리닝(util/screen_od2.py)은 속도 가우시안 노이즈로 단일 run 분산이 커서
savings 추정이 불안정. 본 스크립트는 output/_od2_screening.json 상위 후보를
  (a) deterministic_speed=True 단일 run (노이즈 0, 기대 비교)
  (b) 확률적 12 repeat 평균 (실험과 동일 조건)
양쪽으로 재평가해 안정적 savings 로 재랭크.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from util.environment import RoadNetworkEnv
from util.dijkstra_models import ShortestDijkstra, StaticFuelDijkstra

TOPO  = "data/gangnam_topology_add.json"
SPEED = "data/gangnam_clean_speed_data.csv"
REWARD = {"alpha": 1.0, "arrival_bonus": 700.0}


def sim(env, planner, o, d, hour, max_steps=400):
    s = env.reset(start_node=o, goal_nodes=[d], start_hour=hour)
    M = planner(env)
    fuel = wait = dist = 0.0; steps = 0; reached = False; path = [o]; done = False
    while not done and steps < max_steps:
        v = env.get_valid_actions()
        if not v: break
        a = M.act(s, v); s, _, done, info = env.step(a)
        fuel += info.get("fuel_total", 0); wait += info.get("wait_time", 0)
        dist += info.get("distance", 0); reached = info.get("reached_goal", False)
        path.append(env.current_node); steps += 1
    return dict(fuel=fuel, wait=wait, dist=dist, steps=steps, reached=reached, path=path)


def main():
    cands = json.load(open(ROOT/"output"/"_od2_screening.json", encoding="utf-8"))
    top = cands[:20]   # 이미 savings 오름차순(절감 큰 순)

    env_det = RoadNetworkEnv(TOPO, SPEED, reward_cfg=REWARD, use_signal=True,
                             deterministic_speed=True)
    env_sto = RoadNetworkEnv(TOPO, SPEED, reward_cfg=REWARD, use_signal=True,
                             deterministic_speed=False)
    nodes = env_det.nodes

    rows = []
    for c in top:
        o, d = c["o"], c["d"]
        # (a) 결정론적 단일 run
        sd = sim(env_det, ShortestDijkstra, o, d, 8.0)
        fd = sim(env_det, StaticFuelDijkstra, o, d, 8.0)
        if not (sd["reached"] and fd["reached"]):
            continue
        det_sav = (fd["fuel"] - sd["fuel"]) / sd["fuel"] * 100
        # (b) 확률적 12 repeat 평균
        N = 12
        sf = [sim(env_sto, ShortestDijkstra, o, d, 8.0) for _ in range(N)]
        ff = [sim(env_sto, StaticFuelDijkstra, o, d, 8.0) for _ in range(N)]
        sf_m = sum(r["fuel"] for r in sf)/N; sw_m = sum(r["wait"] for r in sf)/N
        ff_m = sum(r["fuel"] for r in ff)/N; fw_m = sum(r["wait"] for r in ff)/N
        sto_sav = (ff_m - sf_m)/sf_m*100
        sig_s = sum(1 for n in sd["path"] if nodes[n].get("signal"))
        sig_f = sum(1 for n in fd["path"] if nodes[n].get("signal"))
        rows.append(dict(o=o, d=d, straight=c["straight_km"],
                         det_short=round(sd["fuel"]), det_tdd=round(fd["fuel"]),
                         det_sav=round(det_sav,1),
                         sto_short=round(sf_m), sto_short_wait=round(sw_m),
                         sto_tdd=round(ff_m), sto_tdd_wait=round(fw_m),
                         sto_sav=round(sto_sav,1),
                         short_dist=round(sd["dist"]), tdd_dist=round(fd["dist"]),
                         sig_s=sig_s, sig_f=sig_f, diverge=(sd["path"]!=fd["path"])))
        r = rows[-1]
        print(f"{o}->{d} {r['straight']}km | DET sav={r['det_sav']:>5}% "
              f"({r['det_short']}->{r['det_tdd']}) | STO sav={r['sto_sav']:>5}% "
              f"short={r['sto_short']}mL/w{r['sto_short_wait']}s "
              f"tdd={r['sto_tdd']}mL/w{r['sto_tdd_wait']}s | sig {r['sig_s']}->{r['sig_f']}",
              flush=True)

    rows.sort(key=lambda r: r["sto_sav"])
    json.dump(rows, open(ROOT/"output"/"_od2_rescreen.json","w",encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n=== TOP by stochastic 12-rep savings ===")
    for r in rows[:10]:
        print(f"{r['o']}->{r['d']}  STO {r['sto_sav']}%  DET {r['det_sav']}%  "
              f"short {r['sto_short']}mL/{r['sto_short_wait']}s  "
              f"tdd {r['sto_tdd']}mL/{r['sto_tdd_wait']}s  {r['straight']}km")
    print("saved → output/_od2_rescreen.json")


if __name__ == "__main__":
    main()
