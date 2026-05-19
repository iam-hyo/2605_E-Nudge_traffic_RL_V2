"""
environment.py
--------------
RoadNetworkEnv — OpenAI Gym 유사 인터페이스.

State 벡터 (229차원):
  A. 위치              s[0–4]      (5d)
  B. 시간              s[5–7]      (3d)
  C. 현재 노드 신호    s[8–16]     (9d)
  D. 1-hop 노드(K=4)   s[17–60]    (44d, 노드당 11d = pos 2 + sig 9)
  E. 1-hop 링크(K=4)   s[61–68]    (8d,  링크당 2d = len + speed)
  F. 2-hop 노드(N=8)   s[69–156]   (88d, 노드당 11d = pos 2 + sig 9)
  G. 2-hop 링크(L=12)  s[157–228]  (72d, 링크당 6d = len + speed + parent_onehot[4])

Action: 인접 노드 ID (문자열) → node_to_idx 로 정수 변환

신호 9d 인코딩:
  [cycle/180, green_ratio, left_ratio,
   phase_onehot(green/left/red) 3d,
   remain_sec/cycle,
   sin(2π·local_t/cycle), cos(2π·local_t/cycle)]
  - phase type 통합: green→green, {left, left_turn}→left, {red, yellow}→red
  - 비신호 노드: 9d 전체 0
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
K_HOP1      = 4         # 1-hop 노드 수 (최대)
N_HOP2      = 8         # 2-hop 고유 노드 수 (최대)
L_HOP2      = 12        # 2-hop 경로(링크) 수 (최대): K_HOP1 × 3

NODE_FEAT   = 11        # 노드당 피처: pos(2) + sig(9)
LINK1_FEAT  = 2         # 1-hop 링크: len + speed
LINK2_FEAT  = 6         # 2-hop 링크: len + speed + parent_onehot[K_HOP1]
SIG_FEAT    = 9         # 신호 피처 차원

STATE_SIZE  = 5 + 3 + SIG_FEAT + K_HOP1*NODE_FEAT + K_HOP1*LINK1_FEAT \
              + N_HOP2*NODE_FEAT + L_HOP2*LINK2_FEAT   # = 229

SPEED_MAX   = 80.0      # km/h — 정규화 기준
CYCLE_MAX   = 180.0     # s   — 사이클 정규화 기준
NOISE_SIGMA = 0.20      # 속도 노이즈 비율 (±20%)
SPEED_MIN   = 5.0       # km/h 하한
V_TURN_RIGHT = 20 / 3.6 # m/s
V_TURN_LEFT  = 30 / 3.6 # m/s
ACCEL_MS2    = 2.5       # m/s²

# road_type별 노이즈 σ (km/h)
ROAD_SIGMA = {"arterial": 2.0, "local": 3.5}


def _phase_category(ph_type: str) -> str:
    """phase type을 3개 카테고리로 통합.
    green → 'green', {left, left_turn} → 'left', 그 외(red, yellow 등) → 'red'.
    """
    if ph_type == "green":
        return "green"
    if ph_type in ("left", "left_turn"):
        return "left"
    return "red"


def _movement_type(prev_pos, cur_pos, to_pos) -> str:
    """
    회전 종류 판정 — 'straight' / 'left' / 'right' / 'uturn'.

    좌표계 가정: y 상방 (수학적 좌표) — 격자 데이터(row*spacing) 및 강남구 GIS
                (위도) 모두 y가 위로 증가.
    외적 부호:
      cross = dx1·dy2 - dy1·dx2
      cross > 0 → CCW = 운전자 관점 좌회전
      cross < 0 → CW  = 운전자 관점 우회전
      cross ≈ 0, 같은 방향 → 직진 / 반대 방향 → uturn
    """
    if prev_pos is None or cur_pos is None or to_pos is None:
        return "straight"

    dx1 = cur_pos[0] - prev_pos[0]
    dy1 = cur_pos[1] - prev_pos[1]
    dx2 = to_pos[0]  - cur_pos[0]
    dy2 = to_pos[1]  - cur_pos[1]

    # prev == cur (에피소드 시작 직후): 좌/우 판단 불가 → 직진 처리
    if abs(dx1) + abs(dy1) < 1e-6:
        return "straight"
    if abs(dx2) + abs(dy2) < 1e-6:
        return "straight"

    cross = dx1 * dy2 - dy1 * dx2
    dot   = dx1 * dx2 + dy1 * dy2

    # U턴 (반대 방향)
    norm1 = math.hypot(dx1, dy1)
    norm2 = math.hypot(dx2, dy2)
    if dot < 0 and abs(cross) < 0.1 * norm1 * norm2:
        return "uturn"

    # 직진 임계: 외적이 매우 작으면 직진
    if abs(cross) < 0.1 * norm1 * norm2:   # 약 6° 이내
        return "straight"

    return "left" if cross > 0 else "right"


def _phase_allows(phase_type: str, movement: str) -> bool:
    """
    phase 종류와 movement 조합에 대한 통행 가능 여부.

      green        → 직진·우회전
      left/left_turn → 좌회전
      red/yellow   → 전체 정지
      no_signal    → 전체 통행
    """
    cat = _phase_category(phase_type)
    if cat == "red":
        return False
    if cat == "green":
        return movement in ("straight", "right")
    if cat == "left":
        return movement == "left"
    return True


def _node_allows_left(node: dict) -> bool:
    """
    좌회전 허용 여부. 다음 순서로 판정:
      1. node['left_turn_allowed']  (signal_topology.py가 명시한 경우)
      2. signal.phases에 left/left_turn type 존재 여부
      3. 신호 없음 → 좌회전 허용 (이면도로 가정)
    """
    if "left_turn_allowed" in node:
        return bool(node["left_turn_allowed"])
    sig = node.get("signal")
    if sig is None:
        return True
    return any(_phase_category(p["type"]) == "left" for p in sig["phases"])


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

    def _signal_features(self, node_id: str, at_sec: float) -> list[float]:
        """
        신호 9d 인코딩.
        [cycle/180, green_ratio, left_ratio,
         phase_onehot[3], remain_sec/cycle,
         sin(2π·local_t/cycle), cos(2π·local_t/cycle)]
        비신호 노드 또는 use_signal=False → 모두 0.
        """
        zeros = [0.0] * SIG_FEAT
        if not self.use_signal:
            return zeros

        sig = self.nodes[node_id].get("signal")
        if sig is None:
            return zeros

        cycle   = sig["cycle_length"]
        offset  = sig.get("offset", 0)
        local_t = (at_sec + offset) % cycle

        # phase 카테고리별 총 시간
        green_total = 0.0
        left_total  = 0.0
        for ph in sig["phases"]:
            cat = _phase_category(ph["type"])
            if cat == "green":
                green_total += ph["duration"]
            elif cat == "left":
                left_total  += ph["duration"]
            # red 계열은 cycle - green - left 로 유도 가능 → 제외

        # 현재 phase 카테고리 + 잔여 시간
        elapsed   = 0.0
        cur_cat   = "red"
        remain_s  = 0.0
        for ph in sig["phases"]:
            if elapsed <= local_t < elapsed + ph["duration"]:
                cur_cat  = _phase_category(ph["type"])
                remain_s = elapsed + ph["duration"] - local_t
                break
            elapsed += ph["duration"]

        phase_oh = [0.0, 0.0, 0.0]   # [green, left, red]
        if cur_cat == "green":
            phase_oh[0] = 1.0
        elif cur_cat == "left":
            phase_oh[1] = 1.0
        else:
            phase_oh[2] = 1.0

        return [
            min(cycle / CYCLE_MAX, 1.0),
            green_total / cycle,
            left_total  / cycle,
            phase_oh[0], phase_oh[1], phase_oh[2],
            remain_s / cycle,
            math.sin(2 * math.pi * local_t / cycle),
            math.cos(2 * math.pi * local_t / cycle),
        ]

    def _calc_wait(self, node_id: str, arrive_sec: float,
                   movement: str = "straight") -> float:
        """
        movement-aware 신호 대기 (초).

        주의: dynamics는 `use_signal` 설정과 무관하게 항상 신호 준수.
              `use_signal=False`는 State에서 신호 9d를 가릴 뿐, 실제 운전 규칙은
              모든 모델이 동일하게 따라야 학습-시뮬 일관성·모델 간 공정 비교가 성립.

        매개변수
        --------
        node_id    : 통과/회전할 노드
        arrive_sec : 노드 도달 절대 시각
        movement   : 'straight' / 'left' / 'right' / 'uturn'

        반환: 통과 허용 phase까지 대기 시간(초).
        """
        sig = self.nodes[node_id].get("signal")
        if sig is None:
            return 0.0

        cycle   = sig["cycle_length"]
        offset  = sig.get("offset", 0)
        local_t = (arrive_sec + offset) % cycle

        # 현재 phase 인덱스 찾기
        elapsed = 0.0
        cur_idx = 0
        cur_remain = 0.0
        for i, ph in enumerate(sig["phases"]):
            if elapsed <= local_t < elapsed + ph["duration"]:
                cur_idx = i
                cur_remain = elapsed + ph["duration"] - local_t
                # 현재 phase에서 movement 허용?
                if _phase_allows(ph["type"], movement):
                    return 0.0
                break
            elapsed += ph["duration"]

        # 다음 phase부터 순회하여 movement 허용 첫 phase까지 누적
        wait = cur_remain
        n    = len(sig["phases"])
        for j in range(1, n + 1):
            nxt = sig["phases"][(cur_idx + j) % n]
            if _phase_allows(nxt["type"], movement):
                return wait
            wait += nxt["duration"]
        return 0.0  # 모든 phase 차단 시 (이상 케이스)

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
        선택 가능한 다음 노드 목록.

        제외 규칙:
          1. U턴 (previous_node)
          2. 좌회전 불가 노드에서의 좌회전 이동
             (cur 노드의 좌회전 phase 부재 또는 left_turn_allowed=False)

        정렬로 순서 고정 → State 패딩 일관성 유지.
        """
        cur_node = self.nodes[self.current_node]
        cur_pos  = cur_node["pos"]

        # 시작 직후(prev == cur)에는 좌/우 판단 불가 → 좌회전 필터 미적용
        prev_known = (self.previous_node != self.current_node
                      and self.previous_node in self.nodes)
        prev_pos   = self.nodes[self.previous_node]["pos"] if prev_known else None

        allow_left = _node_allows_left(cur_node)

        valid = []
        for nb, _ in self.adj.get(self.current_node, []):
            if nb == self.previous_node:
                continue
            if prev_known and not allow_left:
                to_pos = self.nodes[nb]["pos"]
                if _movement_type(prev_pos, cur_pos, to_pos) == "left":
                    continue
            valid.append(nb)
        return sorted(valid)

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

        # ── 1. cur 노드 출발 신호 대기 (movement-aware) ──────────────────────
        cur_pos  = self.nodes[self.current_node]["pos"]
        to_pos   = self.nodes[action]["pos"]
        prev_pos = (self.nodes[self.previous_node]["pos"]
                    if self.previous_node != self.current_node
                       and self.previous_node in self.nodes else None)
        movement = _movement_type(prev_pos, cur_pos, to_pos)

        abs_now    = self.start_time_sec + self.current_time
        t_wait     = self._calc_wait(self.current_node, abs_now, movement)
        abs_depart = abs_now + t_wait

        # ── 2. 링크 통과 ─────────────────────────────────────────────────────
        lk    = self.links[link_id]
        v_ms  = self._get_link_speed_ms(link_id, abs_depart)

        # 진출 목표 속도: action 노드 신호 종류에 따라 회전 감속
        next_sig = self.nodes[action].get("signal")
        next_has_lt = next_sig and any(_phase_category(p["type"]) == "left"
                                       for p in next_sig["phases"])
        v_exit = (V_TURN_LEFT if next_has_lt
                  else V_TURN_RIGHT if next_sig is not None
                  else v_ms)

        profile  = SpeedProfile(
            v_cruise = v_ms,
            v_entry  = v_ms * 0.7,
            v_exit   = min(v_exit, v_ms),
            link_len = lk["len"],
        )
        t_travel = profile.total_time()

        # 연료 — VT-Micro 출력 L/s → mL 환산 (보상 스케일 정합)
        fuel_drive = profile.total_fuel() * 1000.0
        fuel_wait  = fuel_idle(t_wait)    * 1000.0
        fuel_total = fuel_drive + fuel_wait

        # ── 3. 상태 전이 ─────────────────────────────────────────────────────
        self.current_time  += t_wait + t_travel
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
            "wait_time":    t_wait,           # cur 노드 출발 대기 (movement-aware)
            "fuel_drive":   fuel_drive,
            "fuel_idle":    fuel_wait,
            "fuel_total":   fuel_total,
            "distance":     lk["len"],
            "speed_kmh":    v_ms * 3.6,
            "movement":     movement,         # 'straight'/'left'/'right'/'uturn'
            "abs_depart":   abs_depart,       # cur 출발 절대 시각 (대기 종료 시점)
            "abs_arrive":   abs_depart + t_travel,  # action 도착 절대 시각
            "reached_goal": reached,
            "is_timeout":   timeout,
        }
        return self._get_state(), reward, done, info

    # ── State 벡터 ────────────────────────────────────────────────────────────
    def _get_state(self) -> np.ndarray:
        cur   = self.current_node
        prev  = self.previous_node
        abs_t = self.start_time_sec + self.current_time

        # 같은 State 생성 안에서는 동일 (link_id, slot) 조합당 단일 속도 샘플 사용
        # → 1-hop 속도 / 2-hop ETA 계산이 일관성 있게 유지
        speed_cache: dict[tuple, float] = {}

        def cached_speed(lid: str, at_sec: float) -> float:
            slot = self._time_slot(at_sec)
            key  = (lid, slot)
            if key not in speed_cache:
                speed_cache[key] = self._get_link_speed_ms(lid, at_sec)
            return speed_cache[key]

        cx, cy = self.nodes[cur]["pos"]
        gx, gy = self._goal_center

        # ── A. 위치 (5d) ─────────────────────────────────────────────────────
        s_pos = [
            (cx - self.map_x_min) / self.map_w,
            (cy - self.map_y_min) / self.map_h,
            (gx - cx) / self.map_w,
            (gy - cy) / self.map_h,
            math.hypot(gx - cx, gy - cy) / self.map_diag,
        ]

        # ── B. 시간 (3d) ─────────────────────────────────────────────────────
        t_ratio = abs_t / 86400.0
        s_time = [
            math.sin(2 * math.pi * t_ratio),
            math.cos(2 * math.pi * t_ratio),
            min(self.current_time / 7200.0, 1.0),
        ]

        # ── C. 현재 신호 (9d) ────────────────────────────────────────────────
        s_sig_cur = self._signal_features(cur, abs_t)

        # ── D, E. 1-hop 노드 + 진입 링크 ─────────────────────────────────────
        neighbors = self.get_valid_actions()[:K_HOP1]
        hop1_nodes_block: list[list[float]] = []
        hop1_links_block: list[list[float]] = []

        # 부모 1-hop k → (1-hop_id, eta1)  : 2-hop에서 ETA 추정에 사용
        parent_info: dict[int, tuple[str, float]] = {}

        for k, nb_id in enumerate(neighbors):
            link_id = next((lid for nb, lid in self.adj[cur] if nb == nb_id), None)
            lk      = self.links[link_id]
            v_ms    = cached_speed(link_id, abs_t)
            eta     = lk["len"] / max(v_ms, 0.1)

            nbx, nby = self.nodes[nb_id]["pos"]
            sn       = self._signal_features(nb_id, abs_t + eta)

            hop1_nodes_block.append([
                (nbx - self.map_x_min) / self.map_w,
                (nby - self.map_y_min) / self.map_h,
                *sn,
            ])
            hop1_links_block.append([
                lk["len"] / self.max_link_len,
                (v_ms * 3.6) / SPEED_MAX,
            ])
            parent_info[k] = (nb_id, eta)

        # 패딩 (pos = -1 sentinel: 실제 노드는 pos ∈ [0,1])
        while len(hop1_nodes_block) < K_HOP1:
            hop1_nodes_block.append([-1.0, -1.0] + [0.0] * SIG_FEAT)
            hop1_links_block.append([0.0] * LINK1_FEAT)

        s_hop1_nodes = [v for row in hop1_nodes_block for v in row]
        s_hop1_links = [v for row in hop1_links_block for v in row]

        # ── F, G. 2-hop 노드 + 링크 (BFS round-robin) ────────────────────────
        # 부모별 후보를 모은 후 round-robin 으로 추출 → DFS 편향 제거
        candidates_by_parent: list[list[tuple[str, str]]] = [[] for _ in range(K_HOP1)]
        for k, (nb_id, _eta) in parent_info.items():
            for nb2_id, lid_out in sorted(self.adj.get(nb_id, [])):
                if nb2_id == cur or nb2_id == prev or nb2_id == nb_id:
                    continue
                candidates_by_parent[k].append((nb2_id, lid_out))

        hop2_nodes_list: list[tuple[str, float]] = []      # (node_id, eta_at_node)
        hop2_links_list: list[tuple[int, str, str]] = []   # (parent_k, lid_out, nb2_id)
        seen_nodes: set[str] = set()

        max_rounds = max((len(c) for c in candidates_by_parent), default=0)
        for round_idx in range(max_rounds):
            for k in range(K_HOP1):
                if round_idx >= len(candidates_by_parent[k]):
                    continue
                nb2_id, lid_out = candidates_by_parent[k][round_idx]
                _, eta1 = parent_info[k]

                # 모든 경로(링크)는 별개 — 중복 허용
                if len(hop2_links_list) < L_HOP2:
                    hop2_links_list.append((k, lid_out, nb2_id))

                # 노드는 dedup
                if nb2_id not in seen_nodes and len(hop2_nodes_list) < N_HOP2:
                    seen_nodes.add(nb2_id)
                    v_out = cached_speed(lid_out, abs_t + eta1)
                    eta2  = eta1 + self.links[lid_out]["len"] / max(v_out, 0.1)
                    hop2_nodes_list.append((nb2_id, eta2))

                if (len(hop2_links_list) >= L_HOP2
                        and len(hop2_nodes_list) >= N_HOP2):
                    break
            if (len(hop2_links_list) >= L_HOP2
                    and len(hop2_nodes_list) >= N_HOP2):
                break

        # 2-hop 노드 블록
        hop2_nodes_block: list[list[float]] = []
        for nb2_id, eta2 in hop2_nodes_list:
            nbx, nby = self.nodes[nb2_id]["pos"]
            sn2      = self._signal_features(nb2_id, abs_t + eta2)
            hop2_nodes_block.append([
                (nbx - self.map_x_min) / self.map_w,
                (nby - self.map_y_min) / self.map_h,
                *sn2,
            ])
        while len(hop2_nodes_block) < N_HOP2:
            hop2_nodes_block.append([-1.0, -1.0] + [0.0] * SIG_FEAT)
        s_hop2_nodes = [v for row in hop2_nodes_block for v in row]

        # 2-hop 링크 블록
        hop2_links_block: list[list[float]] = []
        for parent_k, lid_out, _nb2_id in hop2_links_list:
            _, eta1 = parent_info[parent_k]
            v_out   = cached_speed(lid_out, abs_t + eta1)
            parent_oh = [0.0] * K_HOP1
            parent_oh[parent_k] = 1.0
            hop2_links_block.append([
                self.links[lid_out]["len"] / self.max_link_len,
                (v_out * 3.6) / SPEED_MAX,
                *parent_oh,
            ])
        while len(hop2_links_block) < L_HOP2:
            hop2_links_block.append([0.0] * LINK2_FEAT)
        s_hop2_links = [v for row in hop2_links_block for v in row]

        state = (s_pos + s_time + s_sig_cur
                 + s_hop1_nodes + s_hop1_links
                 + s_hop2_nodes + s_hop2_links)
        assert len(state) == STATE_SIZE, f"State dim error: {len(state)} != {STATE_SIZE}"
        return np.array(state, dtype=np.float32)
