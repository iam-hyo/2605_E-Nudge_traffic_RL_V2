"""
_train_common.py
----------------
RL 학습 공통 루프. train/ 스크립트에서 import하여 사용.

주요 개선사항:
  - multi_route: 매 에피소드 랜덤 start/goal + time_slot 사용 (일반화 핵심)
  - shaping_weight: 목표 거리 기반 보조 보상으로 방향 학습 가속
  - warmup_steps: 메모리 충분히 채운 후 replay 시작 (초기 고분산 방지)
  - checkpoint_every: 중간 체크포인트 저장
  - 최고 도달률 모델 자동 저장
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import yaml

from util.environment import RoadNetworkEnv
from util.agent import DQNAgent


def load_cfg(cfg_path: str = "config/config.yaml") -> dict:
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env(cfg: dict, use_signal: bool = True) -> RoadNetworkEnv:
    return RoadNetworkEnv(
        topology_path = cfg["data"]["topology"],
        speed_path    = cfg["data"]["speed"],
        reward_cfg    = cfg["reward"],
        use_signal    = use_signal,
    )


def _dist_to_goal(env: RoadNetworkEnv) -> float:
    """현재 노드에서 목표 중심까지 유클리드 거리."""
    pos  = env.nodes[env.current_node]["pos"]
    goal = env._goal_center
    return math.hypot(pos[0] - goal[0], pos[1] - goal[1])


def train_rl(mode: str, use_signal: bool, cfg_path: str = "config/config.yaml",
             save_name: str | None = None):
    cfg = load_cfg(cfg_path)
    tc  = cfg["train"]
    env = build_env(cfg, use_signal=use_signal)

    node_list = sorted(env.nodes.keys())
    agent = DQNAgent(
        action_size   = env.action_size,
        node_list     = node_list,
        mode          = mode,
        gamma         = tc["gamma"],
        epsilon       = tc["epsilon_start"],
        epsilon_min   = tc["epsilon_min"],
        epsilon_decay = tc["epsilon_decay"],
        lr            = tc["lr"],
        memory_size   = tc["memory_size"],
        batch_size    = tc["batch_size"],
        target_update = tc["target_update"],
    )

    log_interval     = tc.get("log_interval", 100)
    checkpoint_every = tc.get("checkpoint_every", 500)
    warmup_steps     = tc.get("warmup_steps", 3000)
    multi_route      = tc.get("multi_route", True)
    shaping_w        = tc.get("shaping_weight", 0.0)
    map_diag         = env.map_diag

    model_dir = Path(cfg["output"]["model_dir"])
    model_dir.mkdir(exist_ok=True)

    # 경로 / 시간대 목록
    routes     = cfg["experiments"]["routes"]
    time_slots = cfg["experiments"]["time_slots"]

    # 경로 타입별 도달률 추적
    route_reach: dict[str, list[bool]] = defaultdict(list)
    best_reach_rate = -1.0

    name = save_name or f"model_{mode}{'_signal' if use_signal else ''}"

    def log(*args, **kwargs):
        print(*args, **kwargs, flush=True)

    log(f"\n{'='*60}")
    log(f" 학습 시작 | mode={mode} | use_signal={use_signal}")
    log(f" episodes={tc['episodes']} | batch={tc['batch_size']}")
    log(f" memory={tc['memory_size']} | warmup={warmup_steps} steps")
    log(f" multi_route={multi_route} | shaping={shaping_w}")
    log(f" ε: {tc['epsilon_start']} → {tc['epsilon_min']} "
        f"(decay={tc['epsilon_decay']}, ~{_ep_to_min(tc):.0f}ep)")
    log(f"{'='*60}")

    history     = []
    total_steps = 0
    loss_buf: list[float] = []
    t0 = time.time()

    for ep in range(1, tc["episodes"] + 1):
        # ── 경로 / 시간대 선택 ────────────────────────────────────────────────
        if multi_route:
            route = random.choice(routes)
            tslot = random.choice(time_slots)
            state = env.reset(
                start_node  = route["start"],
                goal_nodes  = [route["goal"]],
                start_hour  = tslot["start_hour"],
            )
            route_key = f"{route['type']}"
        else:
            state     = env.reset()
            route_key = "default"

        ep_reward = 0.0
        ep_fuel   = 0.0
        ep_wait   = 0.0
        ep_steps  = 0
        ep_info   = {}
        move_counts = {"straight": 0, "left": 0, "right": 0, "uturn": 0}

        while True:
            valid = env.get_valid_actions()
            if not valid:
                break

            # 보조 보상용 진행 전 거리
            d_before = _dist_to_goal(env) if shaping_w > 0 else 0.0

            action                         = agent.act(state, valid)
            next_state, reward, done, info = env.step(action)
            next_valid                     = env.get_valid_actions()

            # Potential-based 거리 shaping (목표에 가까워지면 양수)
            if shaping_w > 0:
                d_after  = _dist_to_goal(env)
                shaping  = shaping_w * (d_before - d_after) / map_diag
                reward  += shaping

            total_steps += 1

            # warmup 이전에는 메모리만 채우고 replay 생략
            agent.remember(state, action, reward, next_state, done, next_valid)
            if total_steps >= warmup_steps:
                loss = agent.replay()
                if loss is not None:
                    loss_buf.append(loss)

            state      = next_state
            ep_reward += reward
            ep_fuel   += info.get("fuel_total", 0.0)
            ep_wait   += info.get("wait_time",  0.0)
            ep_steps  += 1
            ep_info    = info
            mv = info.get("movement", "straight")
            if mv in move_counts:
                move_counts[mv] += 1

            if done:
                break

        reached = ep_info.get("reached_goal", False)
        agent.end_episode()
        route_reach[route_key].append(reached)

        history.append({
            "episode": ep,
            "reward":  round(ep_reward, 3),
            "fuel":    round(ep_fuel, 3),
            "wait":    round(ep_wait, 1),
            "steps":   ep_steps,
            "epsilon": round(agent.epsilon, 4),
            "reached": reached,
            "route":   route_key,
            "moves":   move_counts.copy(),
        })

        # ── 주기 로그 ─────────────────────────────────────────────────────────
        if ep % log_interval == 0:
            recent     = history[-log_interval:]
            avg_r      = sum(h["reward"]  for h in recent) / len(recent)
            avg_f      = sum(h["fuel"]    for h in recent) / len(recent)
            avg_w      = sum(h["wait"]    for h in recent) / len(recent)
            avg_steps  = sum(h["steps"]   for h in recent) / len(recent)
            reach_r    = sum(h["reached"] for h in recent) / len(recent)
            elapsed    = time.time() - t0
            warmup_tag = "" if total_steps >= warmup_steps else " [WARMUP]"

            # 이동 분포 (좌/우/직진)
            total_moves = {"straight": 0, "left": 0, "right": 0}
            for h in recent:
                for k in total_moves:
                    total_moves[k] += h["moves"].get(k, 0)
            tot = sum(total_moves.values()) or 1
            mv_str = (f"straight={total_moves['straight']/tot:.0%}  "
                      f"left={total_moves['left']/tot:.0%}  "
                      f"right={total_moves['right']/tot:.0%}")

            # 손실 평균 (최근 1000 replay 호출 기준)
            loss_str = ""
            if loss_buf:
                recent_loss = loss_buf[-1000:]
                loss_str = f" | Loss={sum(recent_loss)/len(recent_loss):.3f}"

            # 경로 타입별 도달률
            rt_str = "  ".join(
                f"{k}={sum(v[-50:])/max(len(v[-50:]),1):.0%}"
                for k, v in sorted(route_reach.items())
            )
            log(f"Ep {ep:4d}/{tc['episodes']} | "
                f"R={avg_r:7.1f} | Fuel={avg_f:5.1f}mL | Wait={avg_w:4.0f}s | "
                f"Steps={avg_steps:4.1f} | Reach={reach_r:.0%} | "
                f"ε={agent.epsilon:.3f}{loss_str} | t={elapsed:.0f}s{warmup_tag}")
            if rt_str:
                log(f"          ├ 경로 도달률(최근50): {rt_str}")
            log(f"          └ 이동 분포: {mv_str}")

            # 최고 도달률 모델 저장
            if reach_r > best_reach_rate and total_steps >= warmup_steps:
                best_reach_rate = reach_r
                agent.save(str(model_dir / f"{name}_best.pth"))

        # ── 중간 체크포인트 ────────────────────────────────────────────────────
        if ep % checkpoint_every == 0:
            ckpt_path = model_dir / f"{name}_ep{ep}.pth"
            agent.save(str(ckpt_path))
            log(f"  [체크포인트] {ckpt_path.name}")

    # ── 최종 저장 ─────────────────────────────────────────────────────────────
    agent.save(str(model_dir / f"{name}.pth"))

    # history.json 에 학습 metadata 헤더 포함 (가시성 / 호환성 검증용)
    train_meta = {
        "model_name":    name,
        "mode":          mode,
        "use_signal":    use_signal,
        "topology":      cfg["data"]["topology"],
        "speed":         cfg["data"]["speed"],
        "episodes":      tc["episodes"],
        "reward_cfg":    cfg.get("reward", {}),
        "shaping_w":     shaping_w,
        "trained_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec":   round(time.time() - t0, 1),
        "n_nodes":       env.N,
        "n_links":       len(env.links),
    }
    out_payload = {"metadata": train_meta, "history": history}
    with open(model_dir / f"{name}_history.json", "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    final_reach = sum(h["reached"] for h in history[-200:]) / min(200, len(history))
    log(f"\n{'='*60}")
    log(f" 학습 완료 | 최종 도달률(최근200): {final_reach:.1%}")
    log(f" 최고 도달률: {best_reach_rate:.1%}  → {name}_best.pth")
    log(f" 모델 저장 → {model_dir}/{name}.pth")
    log(f"{'='*60}\n")
    return agent, history


def _ep_to_min(tc: dict) -> float:
    """epsilon이 epsilon_min에 도달하는 에피소드 수 (근사치)."""
    if tc["epsilon_decay"] >= 1.0:
        return float("inf")
    return math.log(tc["epsilon_min"] / tc["epsilon_start"]) / math.log(tc["epsilon_decay"])
