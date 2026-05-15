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


# ── 신호 강제 유틸 (simulation.py 와 동일한 로직) ────────────────────────────
def _calc_signal_wait(node: dict, arrive_sec: float) -> float:
    """적색/황색 → 다음 녹색(또는 좌회전) 페이즈까지 대기. 모든 모델에 공통 적용."""
    sig = node.get("signal")
    if sig is None:
        return 0.0
    local_t = (arrive_sec + sig["offset"]) % sig["cycle_length"]
    elapsed = 0.0
    for ph in sig["phases"]:
        if elapsed <= local_t < elapsed + ph["duration"]:
            if ph["type"] in ("red", "yellow"):
                return elapsed + ph["duration"] - local_t
            return 0.0
        elapsed += ph["duration"]
    return 0.0


def _calc_left_turn_wait(from_node: dict, prev_pos: list, from_pos: list,
                         to_pos: list, current_sec: float) -> float:
    """좌회전 시 left_turn 페이즈까지 대기. 스크린 좌표 외적 < 0 = 좌회전."""
    sig = from_node.get("signal")
    if sig is None:
        return 0.0
    if not any(p["type"] == "left_turn" for p in sig["phases"]):
        return 0.0
    dx1 = from_pos[0] - prev_pos[0]; dy1 = from_pos[1] - prev_pos[1]
    if dx1 == 0 and dy1 == 0:
        return 0.0
    dx2 = to_pos[0] - from_pos[0];  dy2 = to_pos[1] - from_pos[1]
    if dx1 * dy2 - dy1 * dx2 >= 0:
        return 0.0  # 직진 또는 우회전
    local_t = (current_sec + sig["offset"]) % sig["cycle_length"]
    elapsed = 0.0
    for i, ph in enumerate(sig["phases"]):
        if elapsed <= local_t < elapsed + ph["duration"]:
            if ph["type"] == "left_turn":
                return 0.0
            wait = elapsed + ph["duration"] - local_t
            n = len(sig["phases"])
            for j in range(1, n):
                nxt = sig["phases"][(i + j) % n]
                if nxt["type"] == "left_turn":
                    return wait
                wait += nxt["duration"]
            return 0.0
        elapsed += ph["duration"]
    return 0.0


# ── 단일 에피소드 실행 ────────────────────────────────────────────────────────
def run_episode(model, env: RoadNetworkEnv,
                start: str, goal: str, start_hour: float,
                idle_fc: float = 0.5) -> dict:
    """
    idle_fc: 공회전 연료 소모율 (mL/s). 신호 강제 대기 시간에 적용.
    모든 모델에 동일한 신호 규칙 적용 (use_signal 설정과 무관).
      - 적색/황색 도착 → _calc_signal_wait() 로 대기 강제
      - 좌회전 페이즈 있는 노드에서 좌회전 → _calc_left_turn_wait() 로 대기 강제
    """
    state = env.reset(start_node=start, goal_nodes=[goal],
                      start_hour=start_hour)
    done = False
    ep = {
        "fuel_total": 0.0, "fuel_drive": 0.0, "fuel_idle": 0.0,
        "travel_time": 0.0, "wait_time": 0.0,
        "distance": 0.0, "wait_count": 0,
        "reached": False, "steps": 0,
    }

    while not done:
        valid = env.get_valid_actions()
        if not valid:
            break

        from_node = env.current_node
        from_pos  = list(env.nodes[from_node]["pos"])
        prev_node = env.previous_node

        action = model.act(state, valid)

        # ① 좌회전 신호 대기 (출발 교차로)
        lt_wait = 0.0
        if prev_node != from_node:
            prev_pos   = list(env.nodes[prev_node]["pos"])
            to_pos_tmp = list(env.nodes[action]["pos"])
            cur_sec    = env.start_time_sec + env.current_time
            lt_wait = _calc_left_turn_wait(
                env.nodes[from_node], prev_pos, from_pos, to_pos_tmp, cur_sec
            )
            if lt_wait > 0:
                env.current_time += lt_wait

        state, reward, done, info = env.step(action)

        to_node = env.current_node
        info_wt = info.get("wait_time", 0.0)

        # ② 적색 신호 대기 (도착 교차로)
        arrive_sec = env.start_time_sec + env.current_time - info_wt
        sim_wt     = _calc_signal_wait(env.nodes[to_node], arrive_sec)
        extra_wait = sim_wt - info_wt
        if extra_wait > 0:
            env.current_time += extra_wait

        # 이 스텝의 총 신호 대기 (공정 비교를 위해 모든 모델 동일 기준)
        total_wt = lt_wait + sim_wt

        ep["fuel_total"]  += (info.get("fuel_total", 0)
                              + idle_fc * lt_wait + idle_fc * extra_wait)
        ep["fuel_drive"]  += info.get("fuel_drive", 0)
        ep["fuel_idle"]   += (info.get("fuel_idle",  0)
                              + idle_fc * lt_wait + idle_fc * extra_wait)
        ep["travel_time"] += info.get("travel_time", 0)
        ep["wait_time"]   += total_wt
        ep["distance"]    += info.get("distance", 0)
        if total_wt > 0:
            ep["wait_count"] += 1
        ep["steps"] += 1
        ep["reached"] = info.get("reached_goal", False)

    return {k: round(v, 4) if isinstance(v, float) else v
            for k, v in ep.items()}


# ── 메인 실험 루프 ────────────────────────────────────────────────────────────
def main(cfg_path: str = "config/config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))

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

                idle_fc = cfg.get("physics", {}).get("ifc_ml_s", 0.5)
                for rep in range(1, n_repeat + 1):
                    result = run_episode(model, env, start, goal, sh,
                                         idle_fc=idle_fc)
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
    from collections import defaultdict
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
        reached = [r["reached"]    for r in reps]
        summary[k] = {
            "model": model, "route": route, "time_slot": ts,
            "n": len(reps),
            "fuel_mean":  round(float(np.mean(fuels)), 2),
            "fuel_std":   round(float(np.std(fuels)),  2),
            "time_mean":  round(float(np.mean(times)), 2),
            "time_std":   round(float(np.std(times)),  2),
            "dist_mean":  round(float(np.mean(dists)), 1),
            "wait_mean":  round(float(np.mean(waits)), 2),
            "reach_rate": round(float(np.mean(reached)), 3),
        }
    return summary


def _print_summary(summary: dict):
    print(f"\n{'모델':<28} {'경로':<12} {'시간대':<10} "
          f"{'연료(mL)':<12} {'시간(s)':<10} {'도달률'}")
    print("─" * 80)
    for v in summary.values():
        print(f"{v['model']:<28} {v['route']:<12} {v['time_slot']:<10} "
              f"{v['fuel_mean']:>8.1f}±{v['fuel_std']:<5.1f} "
              f"{v['time_mean']:>8.1f}  "
              f"{v['reach_rate']:.0%}")


if __name__ == "__main__":
    main()
