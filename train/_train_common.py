"""
_train_common.py
----------------
RL 학습 공통 루프. train/ 스크립트에서 import하여 사용.

주요 개선사항:
  - 엣지-상대적 행동 공간: 모델 출력이 슬롯 Q값(K_HOP1개)이라 토폴로지와 무관.
    → 여러 토폴로지를 하나의 모델로 동시 학습 가능 (multi-topology).
  - multi_topology: 매 에피소드 가중치 기반으로 토폴로지를 고른 뒤 그 안에서
    랜덤 start/goal + time_slot 사용 (미지 토폴로지 일반화의 핵심).
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
from collections import defaultdict, deque, Counter
from pathlib import Path

import yaml

from util.environment import RoadNetworkEnv, _phase_category
from util.agent import DQNAgent
from util.fuel_calculate import SpeedProfile, fuel_idle


def _fuel_to_go(env: RoadNetworkEnv, goal: str, hour: float) -> dict:
    """목표에서 역방향 연료-Dijkstra → 각 노드의 '목적지까지 예상 연료(mL)'.

    potential-based shaping 용. 링크 비용 = 순항 주행연료 + 통과노드 기대 신호대기 idle.
    회전·시간의존은 무시(정적 근사) — potential 은 근사여도 최적정책 불변(Ng 1999).
    """
    import heapq
    slot = max(0, min(23, int((hour * 3600 - 7 * 3600) // 300)))

    def exp_wait(nid: str) -> float:
        sg = env.nodes[nid].get("signal")
        if not sg:
            return 0.0
        c = sg["cycle_length"]
        g = sum(p["duration"] for p in sg["phases"]
                if _phase_category(p["type"]) == "green")
        red = c - g
        return (red * red) / (2 * c) if c else 0.0   # 균등 도착 기대 대기

    dist = {goal: 0.0}
    pq = [(0.0, goal)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, 1e18):
            continue
        for v, lid in env.adj.get(u, []):
            lk = env.links[lid]
            v_kh = env.speed_db.get(lid, [35.0] * 24)[slot]
            v_ms = max(5.0, v_kh) / 3.6
            drive = SpeedProfile(v_ms, v_ms, v_ms, lk["len"]).total_fuel() * 1000.0
            idle  = fuel_idle(exp_wait(u)) * 1000.0
            nd = d + drive + idle
            if nd < dist.get(v, 1e18):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


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


# ── 다중 토폴로지 학습 환경 구성 ──────────────────────────────────────────────
def _component_of(env: RoadNetworkEnv, src: str) -> list[str]:
    """src 가 속한 연결 요소(노드 ID 리스트) — 랜덤 경로가 도달 가능하도록."""
    seen = {src}
    dq   = deque([src])
    while dq:
        u = dq.popleft()
        for v, _ in env.adj.get(u, []):
            if v not in seen and v in env.nodes:
                seen.add(v)
                dq.append(v)
    return sorted(seen)


def _random_routes(env: RoadNetworkEnv, n: int = 12,
                    min_hops: int = 4) -> list[dict]:
    """연결 요소 안에서 (start, goal) 쌍 n개 무작위 생성 — 도달 가능 보장."""
    comp = _component_of(env, env.default_start)
    if len(comp) < 2:
        return [{"start": env.default_start,
                 "goal":  env.default_goals[0], "name": "default"}]
    routes = []
    for i in range(n):
        s = random.choice(comp)
        g = random.choice(comp)
        tries = 0
        while g == s and tries < 10:
            g = random.choice(comp)
            tries += 1
        routes.append({"start": s, "goal": g, "name": f"rnd{i}"})
    return routes


def _build_train_envs(cfg: dict, use_signal: bool) -> list[dict]:
    """
    config.train.topologies 가 있으면 다중 토폴로지 학습 환경 리스트를 만든다.
    각 항목: {name, env, routes, weight}.
    없으면 config.data 단일 토폴로지로 폴백.
    """
    tc        = cfg["train"]
    topos     = tc.get("topologies")
    train_max = tc.get("train_max_steps")

    if not topos:
        env = build_env(cfg, use_signal=use_signal)
        if train_max:
            env.max_steps = train_max
        return [{"name": Path(cfg["data"]["topology"]).stem,
                 "env": env, "routes": cfg["experiments"]["routes"],
                 "weight": 1.0}]

    out = []
    for spec in topos:
        env = RoadNetworkEnv(
            topology_path = spec["topology"],
            speed_path    = spec["speed"],
            reward_cfg    = cfg["reward"],
            use_signal    = use_signal,
        )
        if train_max:
            env.max_steps = train_max
        if spec.get("use_config_routes"):
            routes = list(cfg["experiments"]["routes"])
            # 정적 primary_boost (첫 경로 복제) — dynamic_od_boost 미사용 시에만.
            # dynamic_od_boost=True 면 학습 루프에서 도달률 역가중으로 동적 선택하므로
            # 정적 복제는 생략(중복 방지).
            if not cfg["train"].get("dynamic_od_boost"):
                routes += [cfg["experiments"]["routes"][0]] * spec.get("primary_boost", 0)
        else:
            routes = _random_routes(env, n=spec.get("n_random_routes", 12))
        out.append({
            "name":   Path(spec["topology"]).stem,
            "env":    env,
            "routes": routes,
            "weight": float(spec.get("weight", 1.0)),
        })
    return out


def _dist_to_goal(env: RoadNetworkEnv) -> float:
    """현재 노드→목표 중심 유클리드 거리. 위경도면 경도축 cos(lat) 보정(미터 비율 정렬)."""
    pos  = env.nodes[env.current_node]["pos"]
    goal = env._goal_center
    xs = getattr(env, "lon_scale", 1.0)
    return math.hypot((pos[0] - goal[0]) * xs, pos[1] - goal[1])


def train_rl(mode: str, use_signal: bool, cfg_path: str = "config/config.yaml",
             save_name: str | None = None, episodes_override: int | None = None):
    cfg = load_cfg(cfg_path)
    tc  = cfg["train"]
    # attention 등 모델별 epoch 보정 — launcher가 episodes_override 전달.
    if episodes_override:
        tc["episodes"] = int(episodes_override)

    train_envs = _build_train_envs(cfg, use_signal)
    env_weights = [te["weight"] for te in train_envs]

    agent = DQNAgent(
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
    shaping_w        = tc.get("shaping_weight", 0.0)
    fuel_shaping_w   = tc.get("fuel_shaping_weight", 0.0)   # 연료 potential shaping
    _f2g_cache: dict = {}
    uturn_penalty    = float(cfg.get("reward", {}).get("uturn_penalty", 0.0))
    revisit_penalty  = float(cfg.get("reward", {}).get("revisit_penalty", 0.0))
    step_penalty     = float(cfg.get("reward", {}).get("step_penalty", 0.0))
    dynamic_od_boost = bool(tc.get("dynamic_od_boost", False))
    # 동적 OD 부스트: 최근 도달률이 낮은 OD 를 더 자주 샘플링 (역가중).
    #   weight_i = max(boost_floor, 1 - reach_rate_i)  → reach 0 이면 1.0, reach 1 이면 floor.
    od_boost_floor = float(tc.get("od_boost_floor", 0.15))
    od_reach: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
    # ── 보상 anneal — uturn/revisit 페널티를 학습 후반 감쇠해 연료최적성 복원 ──
    #   pen_scale(ep) = max(floor, 1 - ep/(episodes*frac)) : 1.0 → floor 선형 감쇠 후 유지.
    #   frac=0 이면 anneal 미적용(상수 1.0). 초반 페널티로 루프회피 학습 → 후반 fuel 신호 지배.
    penalty_anneal_frac = float(tc.get("penalty_anneal_frac", 0.0))
    penalty_floor       = float(tc.get("penalty_floor", 0.15))
    def _pen_scale(ep):
        if penalty_anneal_frac <= 0:
            return 1.0
        return max(penalty_floor, 1.0 - ep / (tc["episodes"] * penalty_anneal_frac))

    model_dir = Path(cfg["output"]["model_dir"])
    model_dir.mkdir(exist_ok=True)

    time_slots = cfg["experiments"]["time_slots"]

    # 토폴로지별 도달률 추적
    route_reach: dict[str, list[bool]] = defaultdict(list)
    best_reach_rate = -1.0

    name = save_name or f"model_{mode}{'_signal' if use_signal else ''}"

    def log(*args, **kwargs):
        print(*args, **kwargs, flush=True)

    log(f"\n{'='*60}")
    log(f" 학습 시작 | mode={mode} | use_signal={use_signal}")
    log(f" episodes={tc['episodes']} | batch={tc['batch_size']}")
    log(f" memory={tc['memory_size']} | warmup={warmup_steps} steps")
    log(f" 학습 토폴로지 ({len(train_envs)}개):")
    for te in train_envs:
        log(f"   - {te['name']:24s} w={te['weight']:.2f} "
            f"nodes={te['env'].N} routes={len(te['routes'])} "
            f"max_steps={te['env'].max_steps}")
    log(f" ε: {tc['epsilon_start']} → {tc['epsilon_min']} "
        f"(decay={tc['epsilon_decay']}, ~{_ep_to_min(tc):.0f}ep)")
    log(f" shaping_w={shaping_w}  uturn_penalty={uturn_penalty}  "
        f"revisit_penalty={revisit_penalty}  "
        f"arrival_bonus={cfg.get('reward', {}).get('arrival_bonus', 500.0)}")
    log(f" dynamic_od_boost={dynamic_od_boost} (floor={od_boost_floor})")
    log(f" penalty_anneal_frac={penalty_anneal_frac} floor={penalty_floor}")
    log(f" fuel_shaping_w={fuel_shaping_w}")
    log(f"{'='*60}")

    history     = []
    total_steps = 0
    loss_buf: list[float] = []
    t0 = time.time()

    for ep in range(1, tc["episodes"] + 1):
        # ── 토폴로지 / 경로 / 시간대 선택 ─────────────────────────────────────
        te        = random.choices(train_envs, weights=env_weights, k=1)[0]
        env       = te["env"]
        # ── 경로 선택 — 동적 OD 부스트 (도달률 역가중) 또는 균등 ──────────────
        if dynamic_od_boost and len(te["routes"]) > 1:
            rweights = []
            for r in te["routes"]:
                dq = od_reach[r["name"]]
                rr = (sum(dq) / len(dq)) if dq else 0.0   # 데이터 없으면 reach 0 → 많이 샘플
                rweights.append(max(od_boost_floor, 1.0 - rr))
            route = random.choices(te["routes"], weights=rweights, k=1)[0]
        else:
            route = random.choice(te["routes"])
        tslot     = random.choice(time_slots)
        route_key = te["name"]
        map_diag  = env.map_diag

        state = env.reset(
            start_node  = route["start"],
            goal_nodes  = [route["goal"]],
            start_hour  = tslot["start_hour"],
        )

        # ── 연료 potential shaping: 목적지까지 예상 잔여연료표 (에피소드당 1회, 캐시) ──
        f2g = None; f2g_norm = 1.0
        if fuel_shaping_w > 0:
            ck = (route_key, route["goal"], tslot["start_hour"])
            if ck not in _f2g_cache:
                _f2g_cache[ck] = _fuel_to_go(env, route["goal"], tslot["start_hour"])
            f2g = _f2g_cache[ck]
            f2g_norm = f2g.get(route["start"], 0.0) or 1.0

        ep_reward = 0.0
        ep_fuel   = 0.0
        ep_wait   = 0.0
        ep_steps  = 0
        ep_info   = {}
        move_counts = {"straight": 0, "left": 0, "right": 0, "uturn": 0}
        visited   = {route["start"]: 1}   # 재방문 패널티용 노드 방문 카운트

        while True:
            valid = env.get_valid_actions()
            if not valid:
                break

            d_before = _dist_to_goal(env) if shaping_w > 0 else 0.0

            action = agent.act(state, valid)
            slot   = valid.index(action)               # 엣지-상대적 슬롯 인덱스
            next_state, reward, done, info = env.step(action)
            next_valid = env.get_valid_actions()

            if shaping_w > 0:
                d_after  = _dist_to_goal(env)
                reward  += shaping_w * (d_before - d_after) / map_diag

            # ── 연료 potential shaping: Δ(목적지까지 예상 잔여연료) ──
            #   reward += w · (fuel2go[이동前] - fuel2go[이동後]) / fuel2go[출발]
            #   신호 적은 쪽으로 갈수록 잔여연료 더↓ → 보상↑. 합=상수라 최적정책 불변.
            if f2g is not None:
                pv = f2g.get(env.previous_node)
                cv = f2g.get(env.current_node)
                if pv is not None and cv is not None:
                    reward += fuel_shaping_w * (pv - cv) / f2g_norm

            pscale = _pen_scale(ep)
            if uturn_penalty > 0 and info.get("movement") == "uturn":
                reward -= uturn_penalty * pscale

            # 재방문 패널티 — 이미 방문한 노드로 다시 진입하면 부과 (루프 억제)
            nxt_node = env.current_node
            if revisit_penalty > 0 and visited.get(nxt_node, 0) > 0:
                reward -= revisit_penalty * pscale
            visited[nxt_node] = visited.get(nxt_node, 0) + 1

            # 스텝 패널티 — 매 스텝 소액 비용 (장거리 배회 억제, 비-potential)
            if step_penalty > 0:
                reward -= step_penalty

            total_steps += 1

            agent.remember(state, slot, reward, next_state, done,
                           len(next_valid))
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
        od_reach[route["name"]].append(reached)   # per-OD 도달률 (동적 부스트 + 로깅)

        history.append({
            "episode": ep,
            "reward":  round(ep_reward, 3),
            "fuel":    round(ep_fuel, 3),
            "wait":    round(ep_wait, 1),
            "steps":   ep_steps,
            "epsilon": round(agent.epsilon, 4),
            "reached": reached,
            "route":   route_key,
            "od":      route["name"],
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

            total_moves = {"straight": 0, "left": 0, "right": 0}
            for h in recent:
                for k in total_moves:
                    total_moves[k] += h["moves"].get(k, 0)
            tot = sum(total_moves.values()) or 1
            mv_str = (f"straight={total_moves['straight']/tot:.0%}  "
                      f"left={total_moves['left']/tot:.0%}  "
                      f"right={total_moves['right']/tot:.0%}")

            loss_str = ""
            if loss_buf:
                recent_loss = loss_buf[-1000:]
                loss_str = f" | Loss={sum(recent_loss)/len(recent_loss):.3f}"

            rt_str = "  ".join(
                f"{k}={sum(v[-50:])/max(len(v[-50:]),1):.0%}"
                for k, v in sorted(route_reach.items())
            )
            log(f"Ep {ep:4d}/{tc['episodes']} | "
                f"R={avg_r:7.1f} | Fuel={avg_f:5.1f}mL | Wait={avg_w:4.0f}s | "
                f"Steps={avg_steps:4.1f} | Reach={reach_r:.0%} | "
                f"ε={agent.epsilon:.3f}{loss_str} | t={elapsed:.0f}s{warmup_tag}")
            if rt_str:
                log(f"          ├ 토폴로지 도달률(최근50): {rt_str}")
            # per-OD 도달률(최근200) — OD-3 학습곡선 직접 확인용
            od_str = "  ".join(
                f"{k.split('_')[0]}={sum(v)/max(len(v),1):.0%}(n{len(v)})"
                for k, v in sorted(od_reach.items())
            )
            if od_str:
                log(f"          ├ OD별 도달률(최근200): {od_str}")
            log(f"          └ 이동 분포: {mv_str}")

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

    train_meta = {
        "model_name":    name,
        "mode":          mode,
        "use_signal":    use_signal,
        "action_space":  "edge-relative (slot, K_HOP1)",
        "topologies":    [{"name": te["name"], "weight": te["weight"],
                           "n_nodes": te["env"].N} for te in train_envs],
        "episodes":      tc["episodes"],
        "reward_cfg":    cfg.get("reward", {}),
        "shaping_w":     shaping_w,
        "trained_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec":   round(time.time() - t0, 1),
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
