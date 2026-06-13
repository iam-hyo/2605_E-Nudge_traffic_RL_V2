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

from util.fuel_calculate import SpeedProfile, fuel_idle, fc_rate
from util.reward import RewardCalculator

# ── 전역 상수 ─────────────────────────────────────────────────────────────────
K_HOP1      = 4         # 1-hop 노드 수 (최대)
N_HOP2      = 8         # 2-hop 고유 노드 수 (최대)
L_HOP2      = 12        # 2-hop 경로(링크) 수 (최대): K_HOP1 × 3

NODE_FEAT   = 11        # 노드당 피처: pos(2) + sig(9)
LINK1_FEAT  = 2         # 1-hop 링크: len + speed
LINK2_FEAT  = 6         # 2-hop 링크: len + speed + parent_onehot[K_HOP1]
SIG_FEAT    = 9         # 신호 피처 차원

# ── v3 경로트리 확률 State (254d) ─────────────────────────────────────────────
B2_HOP    = 2                       # 2-hop 분기 cap
B3_HOP    = 2                       # 3-hop 분기 cap
N2_EDGES  = K_HOP1 * B2_HOP         # 8
N3_EDGES  = N2_EDGES * B3_HOP       # 16
GLOB_FEAT = 6                       # 시간3 + 목적지3
EDGE_FEAT = 8                       # valid,mv_l,mv_r,pass,speed,len,exp_fuel,goal_prog
EXP_FUEL_NORM = 30.0
STATE_SIZE = (GLOB_FEAT + K_HOP1*EDGE_FEAT
              + N2_EDGES*(EDGE_FEAT+1) + N3_EDGES*(EDGE_FEAT+1)
              + K_HOP1)             # + to_visited(1-hop, 배회 억제) = 258

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


def _movement_type(prev_pos, cur_pos, to_pos, xs: float = 1.0) -> str:
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

    # xs: 경도(x) 축 스케일 보정. 위경도 좌표는 1°경도 < 1°위도 (위도37.5°서 0.8배)라
    # degree 공간 그대로 외적/내적하면 각도가 ~20% 왜곡 → 회전 오분류. xs=cos(lat) 곱해
    # 미터 비율로 정렬. 격자(미터 좌표)는 xs=1.0 이라 영향 없음.
    dx1 = (cur_pos[0] - prev_pos[0]) * xs
    dy1 =  cur_pos[1] - prev_pos[1]
    dx2 = (to_pos[0]  - cur_pos[0])  * xs
    dy2 =  to_pos[1]  - cur_pos[1]

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

    # 직진 임계 (2026-05-22 개정: ±6° → ±30°).
    #   기존 0.1(sin≈±5.7°)은 격자(0°·±90°)에는 충분하나, 강남구 실도로의
    #   완만한 곡률(도로가 휘어도 같은 길)을 좌/우회전으로 오분류했다.
    #   0.5(sin=±30°)로 확대해 실도로 곡률을 직진으로 올바르게 인정한다.
    #   격자는 0°·±90° 뿐이라 영향 없음. dot>0 조건은 ±30° 확대가 180°
    #   부근(near-uturn)을 직진으로 삼키지 않도록 하는 안전장치.
    if dot > 0 and abs(cross) < 0.5 * norm1 * norm2:   # 약 ±30° 이내
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
    if movement == "right":          # 비보호 우회전 — 적신호에도 통과 허용
        return True
    if cat == "red":
        return False
    if cat == "green":
        return movement in ("straight", "right")
    if cat == "left":
        return movement == "left"
    return True


def _node_allows_left(node: dict) -> bool:
    """
    좌회전 허용 여부. 다음 순서로 판정 (2026-05-22 개정 — 신호 유무를 최우선):
      1. 신호 없음 → 좌·직·우 모두 허용 (무신호 노드는 만능: 이면도로/비보호좌회전).
         강남구 토폴로지는 무신호 노드에도 left_turn_allowed=false 가 일괄
         명시돼 있어, 이 우선순위가 없으면 전 노드의 93%에서 좌회전이 막혀
         Dijkstra 최단경로조차 실행 불가가 된다.
      2. node['left_turn_allowed'] 명시값 (신호 노드 한정).
      3. signal.phases 에 left/left_turn type 존재 여부.
    """
    sig = node.get("signal")
    if sig is None:
        return True
    if "left_turn_allowed" in node:
        return bool(node["left_turn_allowed"])
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
        deterministic_speed: bool = False,
    ):
        """
        deterministic_speed:
          True  → (link_id, slot) 해시 기반 결정론적 노이즈.
                  같은 링크·같은 시간대는 모든 모델·모든 호출에서 동일 속도.
                  시뮬레이션 시각화에서 모델 간 wall-clock 동기화에 필수.
          False → random.gauss (학습/평가용 기본)
        """
        self.use_signal = use_signal
        self.deterministic_speed = deterministic_speed
        _RC_KEYS = {"alpha", "arrival_bonus", "penalty_timeout", "penalty_dead"}
        rc_kw = {k: v for k, v in (reward_cfg or {}).items() if k in _RC_KEYS}
        self.reward_calc = RewardCalculator(**rc_kw)

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

        # 좌표계 판별 + 경도축 보정 스케일.
        #   geographic(위경도) 이면 1°경도 < 1°위도 → x 축에 cos(lat) 곱해 각도/방향 정렬.
        #   격자(미터 좌표)면 xs=1.0. 회전판정(_movement_type)·shaping에 사용.
        mx = (self.map_x_min + self.map_x_max) / 2
        my = (self.map_y_min + self.map_y_max) / 2
        self.is_geo  = (124 < mx < 132) and (33 < my < 39)   # 한국 위경도 범위
        self.lon_scale = math.cos(math.radians(my)) if self.is_geo else 1.0

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
        링크 속도 샘플링 (m/s). 양방향 동일 속도, ±20% 가우시안 노이즈.

        deterministic_speed=True 일 때 (link_id, slot) 해시 RNG 사용 →
        모든 모델이 같은 링크·같은 시간대에서 동일 속도. 시뮬레이션 시각화의
        Driver 일관성 확보용.
        """
        slot    = self._time_slot(abs_sec)
        base_kh = self.speed_db.get(link_id, [35.0] * 24)[slot]
        sigma   = base_kh * NOISE_SIGMA
        if self.deterministic_speed:
            rng = random.Random(hash((link_id, slot)) & 0xFFFFFFFF)
            v_kh = rng.gauss(base_kh, sigma)
        else:
            v_kh = random.gauss(base_kh, sigma)
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
        self._last_link_speed_ms = None   # 직진 시 이어받을 직전 링크 순항속도
        self._last_link_id = None         # 직전 링크 id (도착시각 분산 추정용)
        self._visited = {self.current_node}   # 방문 노드 (State to_visited, 배회 억제)
        self._goal_center  = self._calc_goal_center()
        return self._get_state()

    def get_valid_actions(self) -> list[str]:
        """
        선택 가능한 다음 노드 목록 (엣지-상대적 행동 공간).

        제외 규칙:
          1. U턴 (previous_node)
          2. 좌회전 불가 노드에서의 좌회전 이동
             (cur 노드의 좌회전 phase 부재 또는 left_turn_allowed=False)

        정렬·절단 규칙:
          나가는 엣지의 방위각(atan2)으로 오름차순 정렬 후 최대 K_HOP1 개로 절단.
          → 슬롯 인덱스 k 가 토폴로지·노드 ID와 무관하게 "방위" 라는 물리적
            의미를 갖는다 (엣지-상대적 행동 공간의 핵심). 모델 출력 슬롯 k 가
            정확히 이 리스트의 k 번째 엣지에 대응 — State 1-hop 블록 순서와도 일치.
        """
        cur_node = self.nodes[self.current_node]
        cur_pos  = cur_node["pos"]

        # 시작 직후(prev == cur)에는 좌/우 판단 불가 → 좌회전 필터 미적용
        prev_known = (self.previous_node != self.current_node
                      and self.previous_node in self.nodes)
        prev_pos   = self.nodes[self.previous_node]["pos"] if prev_known else None

        allow_left = _node_allows_left(cur_node)

        # 막다른 노드(degree==1) 에서는 U턴 허용 — 실제 도로의 dead-end 행동 반영.
        # 강남구 토폴로지의 degree-1 stub 397개(20%) 가 학습을 막던 함정 노드 문제 해소.
        nbs_all = self.adj.get(self.current_node, [])
        is_deadend = (len(nbs_all) <= 1)

        valid = []
        for nb, _ in nbs_all:
            if nb == self.previous_node and not is_deadend:
                continue
            if prev_known and not allow_left and nb != self.previous_node:
                to_pos = self.nodes[nb]["pos"]
                if _movement_type(prev_pos, cur_pos, to_pos, self.lon_scale) == "left":
                    continue
            valid.append(nb)

        def _bearing_key(nb: str) -> tuple[float, str]:
            tx, ty = self.nodes[nb]["pos"]
            return (math.atan2(ty - cur_pos[1], tx - cur_pos[0]), nb)

        valid.sort(key=_bearing_key)
        return valid[:K_HOP1]

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
        movement = _movement_type(prev_pos, cur_pos, to_pos, self.lon_scale)

        abs_now    = self.start_time_sec + self.current_time
        t_wait     = self._calc_wait(self.current_node, abs_now, movement)
        abs_depart = abs_now + t_wait

        # ── 2. 링크 통과 ─────────────────────────────────────────────────────
        lk    = self.links[link_id]
        v_ms  = self._get_link_speed_ms(link_id, abs_depart)

        # 진입 속도 — 드라이버 운동 모델 (2026-05-21 개정):
        #   · 좌/우회전 → 회전 감속 유지 (회전 속도로 진입 후 가속)
        #   · 직진      → 노드에서의 명시적 감속 없음. 직전 링크 순항속도를
        #                 이어받아 링크 간 속도 차이만큼만 가·감속.
        # 신호 정지 모델 (2026-06-11): 대기(t_wait>0)가 발생하면 실제 정지 →
        #   감속(직전 순항→0) + 공회전 + 재가속(0→순항). 정지(stop-and-go)가
        #   연료의 지배 요인이라는 분석을 dynamics에 직접 반영.
        v_prev  = (self._last_link_speed_ms
                   if self._last_link_speed_ms is not None else v_ms)
        stopped = (t_wait > 0.0)
        fuel_stop = 0.0
        if stopped:
            d_dec = max(1.0, v_prev * v_prev / (2 * ACCEL_MS2))
            fuel_stop = (SpeedProfile(v_prev, v_prev, 0.001, d_dec)
                         .total_fuel() * 1000.0)   # 감속(→0) 연료
            v_entry = 0.001                          # 정지 후 재가속 (링크 내 포함)
        elif movement == "left":
            v_entry = min(V_TURN_LEFT, v_ms)
        elif movement == "right":
            v_entry = min(V_TURN_RIGHT, v_ms)
        else:                                   # straight / uturn / 에피소드 시작
            v_entry = v_prev

        profile  = SpeedProfile(
            v_cruise = v_ms,
            v_entry  = v_entry,
            v_exit   = v_ms,
            link_len = lk["len"],
        )
        self._last_link_speed_ms = v_ms
        self._last_link_id       = link_id
        t_travel = profile.total_time()

        # 연료 — VT-Micro 출력 L/s → mL (보상 스케일 정합)
        fuel_drive = profile.total_fuel() * 1000.0
        fuel_wait  = fuel_idle(t_wait)    * 1000.0
        fuel_total = fuel_drive + fuel_wait + fuel_stop

        # ── 3. 상태 전이 ─────────────────────────────────────────────────────
        self.current_time  += t_wait + t_travel
        self.previous_node  = self.current_node
        self.current_node   = action
        self.steps         += 1
        self._visited.add(action)

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

    # ── v3 확률 State 헬퍼 ────────────────────────────────────────────────────
    def _mean_speed_ms(self, lid: str, abs_sec: float) -> float:
        """노이즈 없는 평균 속도(m/s). State는 평균만 관측(POMDP)."""
        slot = self._time_slot(abs_sec)
        base = self.speed_db.get(lid, [35.0] * 24)[slot]
        return max(SPEED_MIN, base) / 3.6

    def _goal_d(self, pos) -> float:
        gx, gy = self._goal_center
        return math.hypot((gx - pos[0]) * self.lon_scale, gy - pos[1])

    # 3-point 가우시안 구적 (속도 최적화: state 생성당 _calc_wait 호출 최소화)
    _GQ3 = ((-1.0, 0.242), (0.0, 0.399), (1.0, 0.242))

    def _signal_stats(self, node: str, mu: float, sigma: float,
                      movement: str) -> tuple:
        """도착시각 ~ N(mu, sigma) 일 때 (통과확률, 기대대기초).
        use_signal=False(base) 또는 비보호 우회전 → (1.0, 0.0)."""
        if not self.use_signal or movement == "right":
            return 1.0, 0.0
        if self.nodes[node].get("signal") is None:
            return 1.0, 0.0
        sigma = max(sigma, 0.5)
        wsum = 0.0; p = 0.0; ew = 0.0
        for z, w in self._GQ3:
            wt = self._calc_wait(node, mu + z * sigma, movement)
            if wt == 0.0:
                p += w
            ew += w * wt; wsum += w
        return p / wsum, ew / wsum

    def _edge_exp_fuel(self, lid: str, v_ms: float, pass_prob: float) -> float:
        """이 링크 1개의 예상 연료(mL): 주행 + (1-pass)×정지비용."""
        lk = self.links[lid]
        drive = fc_rate(v_ms, 0.0) * 1000.0 * (lk["len"] / max(v_ms, 0.5))
        if not self.use_signal:
            return drive
        stop = 0.143 * v_ms * v_ms               # 정지(감속+재가속) 근사 mL
        return drive + (1.0 - pass_prob) * stop

    def _forward_edges(self, node: str, came_from):
        """node 에서 came_from 으로 되돌아가지 않는 전방 엣지(좌회전금지 반영).
        목표 접근도 내림차순 정렬 후 반환 [(nb, lid)]."""
        nbs = self.adj.get(node, [])
        is_dead = (len(nbs) <= 1)
        npos = self.nodes[node]["pos"]
        cpos = (self.nodes[came_from]["pos"]
                if came_from and came_from in self.nodes and came_from != node
                else None)
        left_ok = _node_allows_left(self.nodes[node])
        out = []
        for nb, lid in nbs:
            if nb == came_from and not is_dead:
                continue
            if (not left_ok) and cpos is not None and nb != came_from:
                if _movement_type(cpos, npos, self.nodes[nb]["pos"],
                                  self.lon_scale) == "left":
                    continue
            out.append((nb, lid))
        gd_node = self._goal_d(npos)
        out.sort(key=lambda e: -(gd_node - self._goal_d(self.nodes[e[0]]["pos"])))
        return out

    def _edge_feat(self, src, dst, lid, prev, mu, var):
        """엣지(src→dst) 의 8-피처 + 도착분포 전이. 반환 (feat8, mu_dst, var_dst)."""
        spos = self.nodes[src]["pos"]; dpos = self.nodes[dst]["pos"]
        ppos = (self.nodes[prev]["pos"]
                if prev and prev != src and prev in self.nodes else None)
        mv   = _movement_type(ppos, spos, dpos, self.lon_scale)
        v_ms = self._mean_speed_ms(lid, mu)
        pp, ewait = self._signal_stats(src, mu, math.sqrt(max(var, 0.0)), mv)
        ln   = self.links[lid]["len"]
        ef   = self._edge_exp_fuel(lid, v_ms, pp)
        gp   = max(-1.0, min(1.0, (self._goal_d(spos) - self._goal_d(dpos))
                             / max(ln, 1.0)))
        feat = [
            1.0,
            1.0 if mv == "left"  else 0.0,
            1.0 if mv == "right" else 0.0,
            pp,
            (v_ms * 3.6) / SPEED_MAX,
            ln / self.max_link_len,
            min(ef / EXP_FUEL_NORM, 2.0),
            gp,
        ]
        tt    = ln / max(v_ms, 0.5)
        return feat, mu + ewait + tt, var + (0.20 * tt) ** 2

    # ── State 벡터 (254d 경로트리 확률) ──────────────────────────────────────
    def _get_state(self) -> np.ndarray:
        cur  = self.current_node
        prev = self.previous_node
        abs_t = self.start_time_sec + self.current_time
        cx, cy = self.nodes[cur]["pos"]
        gx, gy = self._goal_center

        # 전역 (6)
        t_ratio = abs_t / 86400.0
        ddx = (gx - cx) * self.lon_scale; ddy = gy - cy
        dd  = math.hypot(ddx, ddy) or 1.0
        glob = [
            math.sin(2 * math.pi * t_ratio),
            math.cos(2 * math.pi * t_ratio),
            min(self.current_time / 7200.0, 1.0),
            ddx / dd, ddy / dd,
            min(dd * 111.0 / 10.0, 1.0),
        ]

        # 현재 노드 도착분포 분산 — 직전 링크에서 누적
        var0 = 0.0
        if self._last_link_id is not None and self._last_link_id in self.links:
            ln0 = self.links[self._last_link_id]["len"]
            v0  = self._mean_speed_ms(self._last_link_id, abs_t)
            var0 = (0.20 * ln0 / max(v0, 0.5)) ** 2

        # 1-hop
        slots1 = self.get_valid_actions()[:K_HOP1]
        e1 = [None] * K_HOP1
        for k, M in enumerate(slots1):
            lid = next((l for nb, l in self.adj[cur] if nb == M), None)
            if lid is None:
                continue
            feat, mu_M, var_M = self._edge_feat(cur, M, lid, prev, abs_t, var0)
            e1[k] = (feat, mu_M, var_M, M)

        # 2-hop (parent k = idx // B2)
        e2 = [None] * N2_EDGES
        for k in range(K_HOP1):
            if e1[k] is None:
                continue
            _, mu_M, var_M, M = e1[k]
            for j, (P, lidP) in enumerate(self._forward_edges(M, cur)[:B2_HOP]):
                feat, mu_P, var_P = self._edge_feat(M, P, lidP, cur, mu_M, var_M)
                e2[k * B2_HOP + j] = (feat, mu_P, var_P, P, M)

        # 3-hop (parent 2-slot = idx // B3)
        e3 = [None] * N3_EDGES
        for s in range(N2_EDGES):
            if e2[s] is None:
                continue
            _, mu_P, var_P, P, M = e2[s]
            for i, (Q, lidQ) in enumerate(self._forward_edges(P, M)[:B3_HOP]):
                feat, _, _ = self._edge_feat(P, Q, lidQ, M, mu_P, var_P)
                e3[s * B3_HOP + i] = (feat,)

        ZERO = [0.0] * EDGE_FEAT
        f1 = [e[0] if e else ZERO for e in e1]
        f2 = [e[0] if e else ZERO for e in e2]
        f3 = [e[0] if e else ZERO for e in e3]
        par2 = [((idx // B2_HOP) / (K_HOP1 - 1)) if e2[idx] else 0.0
                for idx in range(N2_EDGES)]
        par3 = [((idx // B3_HOP) / (N2_EDGES - 1)) if e3[idx] else 0.0
                for idx in range(N3_EDGES)]

        state = list(glob)
        for fi in range(EDGE_FEAT):
            state += [f1[k][fi] for k in range(K_HOP1)]
        for fi in range(EDGE_FEAT):
            state += [f2[s][fi] for s in range(N2_EDGES)]
        state += par2
        for fi in range(EDGE_FEAT):
            state += [f3[t][fi] for t in range(N3_EDGES)]
        state += par3

        # to_visited (1-hop): 이 엣지의 도착 노드를 이미 방문했는가 (배회 억제)
        vis = getattr(self, "_visited", set())
        state += [1.0 if (e1[k] and e1[k][3] in vis) else 0.0
                  for k in range(K_HOP1)]

        assert len(state) == STATE_SIZE, f"State dim {len(state)} != {STATE_SIZE}"
        return np.array(state, dtype=np.float32)
