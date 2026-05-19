"""
experiments/run_experiment.py
------------------------------
실험 수행 스크립트.

- config.yaml 에서 경로·시간대·반복 횟수 로드
- 5개 모델 순차 실행
- output/{timestamp}/results.csv 저장

단독 실행: python experiments/run_experiment.py
"""

from __future__ import annotations

import csv
import datetime
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.environment import RoadNetworkEnv
from util.agent import DQNAgent
from util.dijkstra_models import ShortestDijkstra, StaticFuelDijkstra


# ── 모델 로더 ─────────────────────────────────────────────────────────────────
def _load_model(name: str, cfg: dict, env: RoadNetworkEnv):
    model_dir = Path(cfg["output"]["model_dir"])

    if name == "shortest_dijkstra":
        p = model_dir / "model_shortest_dijkstra.pkl"
        if p.exists():
            with open(p, "rb") as f:
                m = pickle.load(f)
            m.env = env   # env 교체 (경로·시간대별 리셋 반영)
            return m
        return ShortestDijkstra(env)

    if name == "static_fuel_dijkstra":
        p = model_dir / "model_static_fuel_dijkstra.pkl"
        if p.exists():
            with open(p, "rb") as f:
                m = pickle.load(f)
            m.env = env
            return m
        return StaticFuelDijkstra(env)

    # RL 모델
    mode_map = {
        "rl_base":             ("base",      False),
        "rl_signal":           ("signal",    True),
        "rl_signal_attention": ("attention", True),
    }
    mode, use_sig = mode_map[name]
    node_list = sorted(env.nodes.keys())
    tc = cfg["train"]
    agent = DQNAgent(
        action_size   = env.action_size,
        node_list     = node_list,
        mode          = mode,
        gamma         = tc["gamma"],
        epsilon_min   = tc["epsilon_min"],
        epsilon       = tc["epsilon_min"],   # 실험 시 탐험 없음
        epsilon_decay = 1.0,
        lr            = tc["lr"],
    )

    pth_name = {
        "rl_base":             "model_rl_base.pth",
        "rl_signal":           "model_rl_signal.pth",
        "rl_signal_attention": "model_rl_signal_attention.pth",
    }[name]
    pth = model_dir / pth_name
    if pth.exists():
        agent.load(str(pth))
        print(f"  [로드] {pth}")
    else:
        print(f"  [경고] {pth} 없음 — 미학습 모델로 실험 진행")
    return agent


# ── 단일 에피소드 실행 ────────────────────────────────────────────────────────
def run_episode(model, env: RoadNetworkEnv,
                start: str, goal: str, start_hour: float) -> dict:
    """
    환경의 step() 이 신호 규칙(movement-aware 대기 + fuel_idle)을 단일 진실
    공급원으로 처리. 모든 모델 동일 규칙 — 사후 보정 불필요.

    return: 에피소드 KPI dict
      fuel_total/drive/idle, travel_time, wait_time, distance,
      wait_count, reached, steps, n_left, n_right, n_straight, path
    """
    state = env.reset(start_node=start, goal_nodes=[goal],
                      start_hour=start_hour)
    done = False
    ep = {
        "fuel_total": 0.0, "fuel_drive": 0.0, "fuel_idle": 0.0,
        "travel_time": 0.0, "wait_time": 0.0,
        "distance": 0.0, "wait_count": 0,
        "reached": False, "steps": 0,
        "n_left": 0, "n_right": 0, "n_straight": 0,
        "path": [start],
    }

    while not done:
        valid = env.get_valid_actions()
        if not valid:
            break

        action = model.act(state, valid)
        state, reward, done, info = env.step(action)

        ep["fuel_total"]  += info.get("fuel_total",  0.0)
        ep["fuel_drive"]  += info.get("fuel_drive",  0.0)
        ep["fuel_idle"]   += info.get("fuel_idle",   0.0)
        ep["travel_time"] += info.get("travel_time", 0.0)
        wt = info.get("wait_time", 0.0)
        ep["wait_time"]   += wt
        ep["distance"]    += info.get("distance",    0.0)
        if wt > 0.5:
            ep["wait_count"] += 1
        ep["steps"]   += 1
        ep["reached"]  = info.get("reached_goal", False)
        ep["path"].append(env.current_node)

        mv = info.get("movement", "straight")
        if mv == "left":
            ep["n_left"] += 1
        elif mv == "right":
            ep["n_right"] += 1
        else:
            ep["n_straight"] += 1

    # CSV 친화적 정리 — path는 string 으로
    ep["path"] = "->".join(ep["path"])
    return {k: round(v, 4) if isinstance(v, float) else v
            for k, v in ep.items()}


# ── 메인 실험 루프 ────────────────────────────────────────────────────────────
def main(cfg_path: str = "config/config.yaml"):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))

    # 출력 폴더
    ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(cfg["output"]["result_dir"]) / ts
    outdir.mkdir(parents=True, exist_ok=True)

    # 환경 생성 (모델별로 use_signal 다름 → 모델 로드 시 별도 env 생성)
    def make_env(use_signal: bool) -> RoadNetworkEnv:
        return RoadNetworkEnv(
            cfg["data"]["topology"], cfg["data"]["speed"],
            reward_cfg=cfg["reward"], use_signal=use_signal,
        )

    model_flags = cfg["experiments"].get("models", {})
    model_names = [
        "shortest_dijkstra",
        "static_fuel_dijkstra",
        "rl_base",
        "rl_signal",
        "rl_signal_attention",
    ]
    use_signal_map = {
        "shortest_dijkstra":    True,
        "static_fuel_dijkstra": True,
        "rl_base":              False,
        "rl_signal":            True,
        "rl_signal_attention":  True,
    }

    routes    = cfg["experiments"]["routes"]
    time_slots = cfg["experiments"]["time_slots"]
    n_repeat  = cfg["experiments"]["repeat"]

    all_rows = []
    total    = (sum(1 for m in model_names if model_flags.get(m, True))
                * len(routes) * len(time_slots) * n_repeat)
    done_cnt = 0
    t_start  = time.time()

    for m_name in model_names:
        if not model_flags.get(m_name, True):
            print(f"[SKIP] {m_name}")
            continue

        env   = make_env(use_signal_map[m_name])
        model = _load_model(m_name, cfg, env)
        print(f"\n{'─'*50}")
        print(f" 모델: {m_name}")

        for route in routes:
            start = route["start"]
            goal  = route["goal"]

            # PLACEHOLDER 체크
            if "PLACEHOLDER" in start or "PLACEHOLDER" in goal:
                print(f"  [경고] {route['name']}: 노드 ID가 PLACEHOLDER — 건너뜀")
                print(f"         config/config.yaml 의 routes 섹션에 실제 노드 ID를 입력하세요.")
                continue

            for tslot in time_slots:
                label = tslot["label"]
                sh    = tslot["start_hour"]
                reach_cnt = 0

                for rep in range(1, n_repeat + 1):
                    result = run_episode(model, env, start, goal, sh)
                    reach_cnt += int(result["reached"])
                    row = {
                        "model":      m_name,
                        "route":      route["name"],
                        "route_type": route["type"],
                        "time_slot":  label,
                        "start_hour": sh,
                        "rep":        rep,
                        **result,
                    }
                    all_rows.append(row)
                    done_cnt += 1

                elapsed = time.time() - t_start
                eta = elapsed / done_cnt * (total - done_cnt) if done_cnt else 0
                print(f"  {route['name']} | {label} | "
                      f"reach={reach_cnt}/{n_repeat} | "
                      f"진행 {done_cnt}/{total} | ETA {eta:.0f}s")

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    if not all_rows:
        print("\n[경고] 실험 결과 없음. config.yaml의 routes를 확인하세요.")
        return

    csv_path = outdir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    # 요약 통계
    summary = _summarize(all_rows)
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f" 실험 완료 → {outdir}")
    print(f" 전체 실행 수: {len(all_rows)}")
    print(f" 소요 시간: {time.time()-t_start:.1f}s")
    _print_summary(summary)


def _summarize(rows: list[dict]) -> dict:
    from collections import defaultdict, Counter
    grouped = defaultdict(list)
    for r in rows:
        key = (r["model"], r["route"], r["time_slot"])
        grouped[key].append(r)

    summary = {}
    for (model, route, ts), reps in grouped.items():
        k = f"{model}|{route}|{ts}"
        fuels  = [r["fuel_total"]  for r in reps]
        times  = [r["travel_time"] + r["wait_time"] for r in reps]
        dists  = [r["distance"]    for r in reps]
        waits  = [r["wait_time"]   for r in reps]
        steps  = [r["steps"]       for r in reps]
        reached = [r["reached"]    for r in reps]
        # 가장 빈도 높은 경로 (모델이 안정적인지 확인)
        path_freq = Counter(r["path"] for r in reps).most_common(1)
        top_path, top_cnt = path_freq[0] if path_freq else ("", 0)
        summary[k] = {
            "model": model, "route": route, "time_slot": ts,
            "n": len(reps),
            "fuel_mean":  round(float(np.mean(fuels)), 2),
            "fuel_std":   round(float(np.std(fuels)),  2),
            "time_mean":  round(float(np.mean(times)), 2),
            "time_std":   round(float(np.std(times)),  2),
            "dist_mean":  round(float(np.mean(dists)), 1),
            "wait_mean":  round(float(np.mean(waits)), 2),
            "wait_count": round(float(np.mean([r["wait_count"] for r in reps])), 2),
            "steps_mean": round(float(np.mean(steps)), 1),
            "n_left":     round(float(np.mean([r["n_left"]     for r in reps])), 2),
            "n_right":    round(float(np.mean([r["n_right"]    for r in reps])), 2),
            "n_straight": round(float(np.mean([r["n_straight"] for r in reps])), 2),
            "reach_rate": round(float(np.mean(reached)), 3),
            "top_path":   top_path,
            "top_path_ratio": round(top_cnt / len(reps), 2),
        }
    return summary


def _print_summary(summary: dict):
    header = (f"{'모델':<26} {'경로':<10} {'시간대':<10} "
              f"{'연료mL':<14} {'시간s':<10} {'대기s':<8} "
              f"{'스텝':<5} {'도달':<5} {'좌/직/우'}")
    print(f"\n{header}")
    print("─" * len(header))
    for v in summary.values():
        mv = f"{v['n_left']:.0f}/{v['n_straight']:.0f}/{v['n_right']:.0f}"
        print(f"{v['model']:<26} {v['route']:<10} {v['time_slot']:<10} "
              f"{v['fuel_mean']:>7.1f}±{v['fuel_std']:<5.1f} "
              f"{v['time_mean']:>7.1f}  "
              f"{v['wait_mean']:>5.1f}  "
              f"{v['steps_mean']:>4.1f}  "
              f"{v['reach_rate']:>4.0%}  "
              f"{mv}")


if __name__ == "__main__":
    main()
