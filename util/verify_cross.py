"""
verify_cross.py
---------------
6x6_cross 토폴로지 사전 검증 — 6개 세로 도로(col) 각각으로 n1→n36 을
강제 주행시켜 평균 연료/시간/거리를 비교한다.

목표: peak 시간대에 col 2 (n3→n33, 명시적 최장이지만 신호 우호) 가
      연료 최소가 되어야 — 신호 학습 모델이 채택할 근거가 성립.

사용: python util/verify_cross.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from util.environment import RoadNetworkEnv

GRID = 6
TOPO  = str(ROOT / "data" / "6x6_cross_topology.json")
SPEED = str(ROOT / "data" / "6x6_cross_speed_data.csv")


def nid(r: int, c: int) -> str:
    return str(r * GRID + c + 1)


def col_route(c: int) -> list[str]:
    """col c 로 n1→n36 가는 경로 노드열."""
    path = [nid(0, cc) for cc in range(0, c + 1)]      # 하단 가로선
    path += [nid(r, c) for r in range(1, GRID)]        # col c 세로 (사선)
    path += [nid(GRID - 1, cc) for cc in range(c + 1, GRID)]  # 상단 가로선
    return path


def rollout(env: RoadNetworkEnv, path: list[str], start_hour: float) -> dict:
    env.reset(start_node=path[0], goal_nodes=[path[-1]], start_hour=start_hour)
    tot = {"fuel": 0.0, "time": 0.0, "dist": 0.0, "wait": 0.0, "waits": 0}
    for nxt in path[1:]:
        _, _, done, info = env.step(nxt)
        tot["fuel"] += info["fuel_total"]
        tot["time"] += info["travel_time"] + info["wait_time"]
        tot["dist"] += info["distance"]
        tot["wait"] += info["wait_time"]
        if info["wait_time"] > 0.5:
            tot["waits"] += 1
    return tot


def main(n_runs: int = 60):
    env = RoadNetworkEnv(TOPO, SPEED, use_signal=True)
    for label, sh in (("off_peak 07:00", 7.0), ("peak 08:00", 8.0)):
        print(f"\n{'='*72}\n {label}\n{'='*72}")
        print(f"  {'col':<5}{'tag':<11}{'fuel(mL)':<12}{'time(s)':<10}"
              f"{'dist(m)':<10}{'wait(s)':<10}{'정지횟수'}")
        rows = []
        for c in range(GRID):
            path = col_route(c)
            acc = {"fuel": 0.0, "time": 0.0, "dist": 0.0, "wait": 0.0, "waits": 0.0}
            for _ in range(n_runs):
                t = rollout(env, path, sh)
                for k in acc:
                    acc[k] += t[k]
            for k in acc:
                acc[k] /= n_runs
            from util.generate_data_cross import PERSONA
            rows.append((c, PERSONA[c]["tag"], acc))
            print(f"  {c:<5}{PERSONA[c]['tag']:<11}{acc['fuel']:<12.1f}"
                  f"{acc['time']:<10.1f}{acc['dist']:<10.0f}"
                  f"{acc['wait']:<10.1f}{acc['waits']:.1f}")
        best = min(rows, key=lambda x: x[2]["fuel"])
        print(f"  → 최소 연료 경로: col {best[0]} ({best[1]})")
        if "peak" in label:
            ok = best[0] == 2
            print(f"  → [목표 col 2] {'✅ 달성' if ok else '❌ 미달성'}")


if __name__ == "__main__":
    main()
