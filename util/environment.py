"""
environment.py
--------------
RoadNetworkEnv — OpenAI Gym 유사 인터페이스.

State 벡터 (59차원):
  A. 현재 위치      s[0–6]   (7d)
  B. 시간           s[7–9]   (3d)
  C. 현재 신호      s[10–14] (5d)
  D. 1-hop × K=4   s[15–46] (32d)
  E. 2-hop × M=4   s[47–58] (12d)

Action: 인접 노드 ID (문자열) → node_to_idx 로 정수 변환
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np

from util.fuel_calculate import SpeedProfile, fuel_idle
from util.reward import RewardCalculator

# ── 전역 상수 ─────────────────────────────────────────────────────────────────
K_HOP1      = 4
M_HOP2      = 4
STATE_SIZE  = 59        # 7+3+5+32+12
SPEED_MAX   = 80.0      # km/h — 정규화 기준
NOISE_SIGMA = 0.20      # 속도 노이즈 비율 (±20%)
SPEED_MIN   = 5.0       # km/h 하한
V_TURN_RIGHT = 20 / 3.6 # m/s
V_TURN_LEFT  = 30 / 3.6 # m/s
ACCEL_MS2    = 2.5       # m/s²

# road_type별 노이즈 σ (km/h)
ROAD_SIGMA = {"arterial": 2.0, "local": 3.5}


class RoadNetworkEnv:
    """
    매개변수
    --------
    topology_path : topology JSON 경로
    speed_path    : speed_data CSV 경로
    reward_cfg    : RewardCalculator 키워드 인자 dict
    use_signal    : False 이면 신호 State를 모두 0으로 반환 (RL Base용)
    """

    def __init__(
        self,
        topology_path: str,
        speed_path:    str,
        reward_cfg:    Optional[dict] = None,
        use_signal:    bool = True,
    ):
        self.use_signal = use_signal
        self.reward_calc = RewardCalculator(**(reward_cfg or {}))

        # ── 토폴로지 로드 ─────────────────────────────────────────────────────
        with open(topology_path, encoding="utf-8") as f:
            topo = json.load(f)

        meta = topo["metadata"]
        self.default_start = str(meta["start_node"])
        self.default_goals = [str(g) for g in meta["goal_nodes"]]
        self.max_steps     = meta["max_steps"]
        self.default_start_hour = float(meta.get("start_hour", 7.0))

        self.nodes: dict[str, dict] = {str(n["id"]): n for n in topo["nodes"]}
        self.node_ids   = sorted(self.nodes.keys())
        self.node_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}
        self.N = len(self.node_ids)

        # 양방향 adj: {node_id: [(neighbor_id, link_id), ...]}
        self.links: dict[str, dict] = {}
        self.adj:   dict[str, list[tuple[str, str]]] = {n: [] for n in self.node_ids}

        max_len = 1.0
        for lk in topo["links"]:
            lid, e1, e2 = str(lk["id"]), str(lk["end1"]), str(lk["end2"])
            self.links[lid] = lk
            self.adj[e1].append((e2, lid))
            self.adj[e2].append((e1, lid))
            if lk["len"] > max_len:
                max_len = lk["len"]
        self.max_link_len = max_len

        # 지도 범위 (좌표 정규화용)
        all_pos = [n["pos"] for n in self.nodes.values()]
        xs = [p[0] for p in all_pos]
        ys = [p[1] for p in all_pos]
        self.map_x_min, self.map_x_max = min(xs), max(xs) or 1.0
        self.map_y_min, self.map_y_max = min(ys), max(ys) or 1.0
        self.map_w    = self.map_x_max - self.map_x_min or 1.0
        self.map_h    = self.map_y_max - self.map_y_min or 1.0
        self.map_diag = math.hypot(self.map_w, self.map_h) or 1.0

        # ── 속도 DB 로드 ──────────────────────────────────────────────────────
        # {link_id: [t_0, t_1, ..., t_23]}  (km/h)
        self.speed_db: dict[str, list[float]] = {}
        self._load_speed_csv(speed_path)

        self.state_size  = STATE_SIZE
        self.action_size = self.N

        # 에피소드 상태 (reset으로 초기화)
        self.current_node  = self.default_start
        self.previous_node = self.default_start
        self.goal_nodes    = self.default_goals
        self.current_time  = 0.0
        self.start_time_sec = int(self.default_start_hour * 3600)
        self.steps         = 0
        self._goal_center  = self._calc_goal_center()

    # ── 내부 유틸 ─────────────────────────────────────────────────────────────
    def _load_speed_csv(self, path: str):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lid = row["link_id"]
                speeds = [float(row[f"t_{i}"]) for i in range(24)]
                self.speed_db[lid] = speeds

    def _calc_goal_center(self) -> list[float]:
        gps = [self.nodes[g]["pos"] for g in self.goal_nodes if g in self.nodes]
        if not gps:
            return [0.0, 0.0]
        return [sum(p[0] for p in gps) / len(gps),
                sum(p[1] for p in gps) / len(gps)]

    def _time_slot(self, abs_sec: float) -> int:
        """절대 초 → 5분 슬롯 인덱스 (0~23)."""
        slot = int((abs_sec - 7 * 3600) // 300)
        return max(0, min(23, slot))

    def _get_link_speed_ms(self, link_id: str, abs_sec: float) -> float:
        """
        링크 속도 샘플링 (m/s).
        양방향 동일 속도. 노이즈 20% 가우시안.
        """
        slot    = self._time_slot(abs_sec)
        base_kh = self.speed_db.get(link_id, [35.0] * 24)[slot]
        sigma   = base_kh * NOISE_SIGMA
        v_kh    = random.gauss(base_kh, sigma)
        v_kh    = max(SPEED_MIN, v_kh)
        return v_kh / 3.6   # → m/s

    def _get_signal(self, node_id: str, at_sec: float) -> dict:
        """
        신호 정보 반환.
        {exists, sin, cos, remain, has_lt}
        """
        if not self.use_signal:
            return {"exists": 0.0, "sin": 0.0, "cos": 0.0,
                    "remain": 0.0, "has_lt": 0.0}

        sig = self.nodes[node_id].get("signal")
        if sig is None:
            return {"exists": 0.0, "sin": 0.0, "cos": 0.0,
                    "remain": 0.0, "has_lt": 0.0}

        cycle   = sig["cycle_length"]
        offset  = sig["offset"]
        local_t = (at_sec + offset) % cycle

        s_sin = math.sin(2 * math.pi * local_t / cycle)
        s_cos = math.cos(2 * math.pi * local_t / cycle)

        # 현재 페이즈 잔여 비율
        elapsed, remain = 0.0, 0.0
        for ph in sig["phases"]:
            if elapsed <= local_t < elapsed + ph["duration"]:
                remain = (elapsed + ph["duration"] - local_t) / ph["duration"]
                break
            elapsed += ph["duration"]

        has_lt = 1.0 if any(p["type"] == "left_turn"
                            for p in sig["phases"]) else 0.0
        return {"exists": 1.0, "sin": s_sin, "cos": s_cos,
                "remain": remain, "has_lt": has_lt}

    def _calc_wait(self, node_id: str, arrive_sec: float) -> float:
        """신호 도달 시각 기준 대기 시간(초). 비신호=0."""
        sig = self.nodes[node_id].get("signal")
        if sig is None or not self.use_signal:
            return 0.0

        cycle   = sig["cycle_length"]
        offset  = sig["offset"]
        local_t = (arrive_sec + offset) % cycle

        elapsed = 0.0
        for ph in sig["phases"]:
            if elapsed <= local_t < elapsed + ph["duration"]:
                if ph["type"] in ("red", "yellow"):
                    return elapsed + ph["duration"] - local_t
                return 0.0
            elapsed += ph["duration"]
        return 0.0

    # ── Public API ────────────────────────────────────────────────────────────
    def reset(
        self,
        start_node:  Optional[str] = None,
        goal_nodes:  Optional[list[str]] = None,
        start_hour:  Optional[float] = None,
    ) -> np.ndarray:
        self.current_node  = start_node or self.default_start
        self.previous_node = self.current_node
        self.goal_nodes    = goal_nodes  or self.default_goals
        self.start_time_sec = int((start_hour or self.default_start_hour) * 3600)
        self.current_time  = 0.0
        self.steps         = 0
        self._goal_center  = self._calc_goal_center()
        return self._get_state()

    def get_valid_actions(self) -> list[str]:
        """
        U턴(previous_node) 제외한 인접 노드 목록.
        정렬로 순서 고정 → State 패딩 일관성 유지.
        """
        return sorted([
            nb for nb, _ in self.adj.get(self.current_node, [])
            if nb != self.previous_node
        ])

    def step(self, action: str) -> tuple[np.ndarray, float, bool, dict]:
        # 링크 탐색
        link_id = None
        for nb, lid in self.adj.get(self.current_node, []):
            if nb == action:
                link_id = lid
                break

        if link_id is None:
            state = self._get_state()
            r = self.reward_calc.terminal_reward(False, False, True)
            return state, r, True, {"msg": "invalid_action", "reached_goal": False}

        lk       = self.links[link_id]
        abs_now  = self.start_time_sec + self.current_time
        v_ms     = self._get_link_speed_ms(link_id, abs_now)

        # 진출 목표 속도 결정 (회전 여부는 next 노드 신호로 판단)
        next_sig  = self.nodes[action].get("signal")
        has_lt    = next_sig and any(p["type"] == "left_turn"
                                     for p in next_sig["phases"])
        # 간단히 left_turn 페이즈 있으면 좌회전 가능 교차로로 가정 (30 km/h)
        # 신호 없는 교차로 / 직진은 순항 속도 유지
        v_exit = V_TURN_LEFT if has_lt else V_TURN_RIGHT if (
            next_sig and not has_lt) else v_ms

        profile    = SpeedProfile(
            v_cruise  = v_ms,
            v_entry   = v_ms * 0.7,   # 이전 링크 진출 속도 근사
            v_exit    = min(v_exit, v_ms),
            link_len  = lk["len"],
        )
        t_travel   = profile.total_time()
        arrive_sec = abs_now + t_travel
        t_wait     = self._calc_wait(action, arrive_sec)

        # 연료
        fuel_drive = profile.total_fuel()
        fuel_wait  = fuel_idle(t_wait)
        fuel_total = fuel_drive + fuel_wait

        # 상태 전이
        self.current_time  += t_travel + t_wait
        self.previous_node  = self.current_node
        self.current_node   = action
        self.steps         += 1

        reached  = self.current_node in self.goal_nodes
        timeout  = self.steps >= self.max_steps

        r_step     = self.reward_calc.step_reward(fuel_total)
        r_terminal = self.reward_calc.terminal_reward(reached, timeout, False)
        reward     = r_step + r_terminal

        done = reached or timeout
        info = {
            "travel_time":  t_travel,
            "wait_time":    t_wait,
            "fuel_drive":   fuel_drive,
            "fuel_idle":    fuel_wait,
            "fuel_total":   fuel_total,
            "distance":     lk["len"],
            "speed_kmh":    v_ms * 3.6,
            "reached_goal": reached,
            "is_timeout":   timeout,
        }
        return self._get_state(), reward, done, info

    # ── State 벡터 ────────────────────────────────────────────────────────────
    def _get_state(self) -> np.ndarray:
        cur   = self.current_node
        prev  = self.previous_node
        abs_t = self.start_time_sec + self.current_time

        cx, cy = self.nodes[cur]["pos"]
        px, py = self.nodes[prev]["pos"]
        gx, gy = self._goal_center

        # A. 위치 (7d)
        s_pos = [
            (cx - self.map_x_min) / self.map_w,
            (cy - self.map_y_min) / self.map_h,
            (px - self.map_x_min) / self.map_w,
            (py - self.map_y_min) / self.map_h,
            (gx - cx) / self.map_w,
            (gy - cy) / self.map_h,
            math.hypot(gx - cx, gy - cy) / self.map_diag,
        ]

        # B. 시간 (3d)
        t_ratio = abs_t / 86400.0
        s_time = [
            math.sin(2 * math.pi * t_ratio),
            math.cos(2 * math.pi * t_ratio),
            min(self.current_time / 7200.0, 1.0),
        ]

        # C. 현재 신호 (5d)
        sc    = self._get_signal(cur, abs_t)
        s_sig = [sc["exists"], sc["sin"], sc["cos"], sc["remain"], sc["has_lt"]]

        # D. 1-hop (K×8 = 32d)
        neighbors = self.get_valid_actions()[:K_HOP1]
        hop1 = []
        for nb_id in neighbors:
            # 링크 ID 찾기
            link_id = next((lid for nb, lid in self.adj[cur] if nb == nb_id), None)
            lk      = self.links[link_id]
            v_ms    = self._get_link_speed_ms(link_id, abs_t)
            eta     = lk["len"] / max(v_ms, 0.1)
            nbx, nby = self.nodes[nb_id]["pos"]
            sn      = self._get_signal(nb_id, abs_t + eta)
            hop1.append([
                (nbx - self.map_x_min) / self.map_w,
                (nby - self.map_y_min) / self.map_h,
                (v_ms * 3.6) / SPEED_MAX,
                lk["len"] / self.max_link_len,
                sn["sin"],
                sn["cos"],
                sn["remain"],
                sn["has_lt"],
            ])
        while len(hop1) < K_HOP1:
            hop1.append([0.0] * 8)
        s_hop1 = [v for row in hop1 for v in row]

        # E. 2-hop (M×3 = 12d)
        seen = {cur, prev}
        hop2_list: list[tuple[str, str, float]] = []
        for nb_id in neighbors:
            link_id1 = next((lid for nb, lid in self.adj[cur] if nb == nb_id), None)
            v1  = self._get_link_speed_ms(link_id1, abs_t)
            eta1 = self.links[link_id1]["len"] / max(v1, 0.1)
            for nb2_id, lid2 in sorted(self.adj.get(nb_id, [])):
                if nb2_id in seen:
                    continue
                seen.add(nb2_id)
                v2   = self._get_link_speed_ms(lid2, abs_t + eta1)
                eta2 = eta1 + self.links[lid2]["len"] / max(v2, 0.1)
                hop2_list.append((nb2_id, lid2, eta2))
                if len(hop2_list) == M_HOP2:
                    break
            if len(hop2_list) == M_HOP2:
                break

        hop2 = []
        for nb2_id, lid2, eta in hop2_list:
            v2  = self._get_link_speed_ms(lid2, abs_t + eta)
            sn2 = self._get_signal(nb2_id, abs_t + eta)
            hop2.append([(v2 * 3.6) / SPEED_MAX, sn2["sin"], sn2["cos"]])
        while len(hop2) < M_HOP2:
            hop2.append([0.0, 0.0, 0.0])
        s_hop2 = [v for row in hop2 for v in row]

        state = s_pos + s_time + s_sig + s_hop1 + s_hop2
        assert len(state) == STATE_SIZE, f"State dim error: {len(state)}"
        return np.array(state, dtype=np.float32)
