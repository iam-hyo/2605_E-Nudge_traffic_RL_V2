"""
dijkstra_models.py
------------------
두 가지 Dijkstra 기반 모델.

  ShortestDijkstra   — 링크 길이(m) 최소화
  StaticFuelDijkstra — Time-Dependent Dijkstra, 예상 연료 최소화
                       (속도 기댓값 사용, 신호 시간의존성 반영)

두 클래스 모두 DQNAgent와 동일한 act() 인터페이스를 구현해
run_experiment.py에서 동일하게 호출 가능.
"""

from __future__ import annotations

import heapq
from typing import Optional

import numpy as np

from util.environment import (_movement_type, _node_allows_left,
                               V_TURN_LEFT, V_TURN_RIGHT)
from util.fuel_calculate import SpeedProfile, fuel_idle


class _DijkstraBase:
    """공통 인터페이스."""
    epsilon = 0.0   # 실험 코드에서 epsilon 참조 시 에러 방지

    def act(self, state: np.ndarray, valid_actions: list[str]) -> str:
        raise NotImplementedError

    def remember(self, *a, **kw): pass
    def replay(self): return None
    def end_episode(self): pass
    def save(self, path): pass
    def load(self, path): pass


class ShortestDijkstra(_DijkstraBase):
    """
    회전제한 인지 최단거리 Dijkstra — "신호 준수 최단거리".

    2026-05-23 개정: 기존 무방향-그래프 Dijkstra 는 신호 노드의 회전제한
    (좌회전 금지)을 모르고 최단경로를 계산해, 강남구 OD-2(양재→영동대교)
    같은 사례에서 통행불가 좌회전이 포함된 경로를 반환 → 에이전트가
    `get_valid_actions` 에서 그 경로를 따라갈 수 없어 탈선·미도달했다.

    본 버전은 `StaticFuelDijkstra` 와 동일하게:
      · state = (현재 노드, 진입 노드) — 진입 방향에 따라 다음 허용 회전이
        달라지므로 prev 를 함께 추적.
      · 노드 u 에서 좌회전이 금지(`_node_allows_left(u)==False`)이고
        진입 prev 가 알려져 있을 때, _movement_type(prev,u,next)=='left'
        간선을 확장에서 제외.
      · U턴(다음 노드 == prev) 차단.
    결과: 모든 모델이 따르는 `env.get_valid_actions` 와 정합한, 실제로
    주행 가능한 **최단거리** 경로를 계산한다 (회전제한 인지 후에도
    여전히 거리 최소화가 목적).
    """

    def __init__(self, env):
        self.env = env

    def _run(self, src: str, src_prev: Optional[str] = None
             ) -> tuple[dict, dict]:
        """src(진입 prev=src_prev) 에서 도달 가능한 (node, prev) 상태 최단 거리.

        반환:
          dist: {(node, prev): cum_distance}
          prev_state: {(node, prev): predecessor_state or None}
        """
        init = (src, src_prev)
        dist: dict[tuple, float] = {init: 0.0}
        prev_state: dict[tuple, Optional[tuple]] = {init: None}
        pq = [(0.0, src, src_prev)]

        while pq:
            d, u, u_prev = heapq.heappop(pq)
            st = (u, u_prev)
            if d > dist.get(st, float("inf")):
                continue
            u_node = self.env.nodes[u]
            u_left_ok = _node_allows_left(u_node)
            u_pos = u_node["pos"]
            prev_pos = (self.env.nodes[u_prev]["pos"]
                        if u_prev and u_prev != u and u_prev in self.env.nodes
                        else None)
            nbs_all = self.env.adj.get(u, [])
            is_deadend = (len(nbs_all) <= 1)            # degree-1 stub
            for nb, lid in nbs_all:
                # U턴 — degree-1 dead-end 에서만 허용 (env.get_valid_actions 와 정합).
                if nb == u_prev and not is_deadend:
                    continue
                # 좌회전 금지 노드의 좌회전 간선 제외 (U턴은 좌회전 분류 대상 아님)
                if (not u_left_ok and prev_pos is not None
                        and nb != u_prev):
                    if _movement_type(prev_pos, u_pos,
                                      self.env.nodes[nb]["pos"]) == "left":
                        continue
                nd = d + self.env.links[lid]["len"]
                nb_state = (nb, u)
                if nd < dist.get(nb_state, float("inf")):
                    dist[nb_state] = nd
                    prev_state[nb_state] = st
                    heapq.heappush(pq, (nd, nb, u))
        return dist, prev_state

    def act(self, state: np.ndarray, valid_actions: list[str]) -> str:
        src      = self.env.current_node
        src_prev = (self.env.previous_node
                    if self.env.previous_node != src else None)
        goals    = set(self.env.goal_nodes)

        dist, prev_state = self._run(src, src_prev)

        # 목표 노드 — 어떤 진입 방향(state) 으로든 도달 가능하면 최소 거리 선택
        reachable = [(d, st) for st, d in dist.items() if st[0] in goals]
        if not reachable:
            return valid_actions[0] if valid_actions else src

        _, goal_state = min(reachable, key=lambda x: x[0])

        # 경로 복원 — state chain → 노드 리스트
        chain: list[tuple] = []
        cur_st: Optional[tuple] = goal_state
        while cur_st is not None:
            chain.append(cur_st)
            cur_st = prev_state.get(cur_st)
        chain.reverse()                     # init … goal
        path = [st[0] for st in chain]

        if len(path) < 2:
            return valid_actions[0] if valid_actions else src
        next_node = path[1]
        return next_node if next_node in valid_actions else (
            valid_actions[0] if valid_actions else src)


class StaticFuelDijkstra(_DijkstraBase):
    """
    Time-Dependent Dijkstra — 예상 연료 최소 경로.
    속도: CSV 기댓값 사용 (노이즈 없음)
    신호: 도착 시각 기준 대기 시간 반영
    """

    def __init__(self, env):
        self.env = env

    def _expected_speed_ms(self, link_id: str, abs_sec: float) -> float:
        slot   = max(0, min(23, int((abs_sec - 7 * 3600) // 300)))
        v_kh   = self.env.speed_db.get(link_id, [35.0] * 24)[slot]
        return max(5.0, v_kh) / 3.6

    def _link_fuel(self, link_id: str, abs_sec: float, v_entry: float,
                   src: str, prev: Optional[str] = None) -> tuple[float, float]:
        """
        (연료 mL, 소요 시간 초) 반환.

        새 환경 규약(env.step과 동일): 출발 신호 대기 + 링크 통과.
          - cur 노드(src) 에서 dst 방향 movement 판정
          - cur 노드 신호의 movement-허용 phase 까지 대기
          - 대기 후 출발, 링크 통과
        """
        lk  = self.env.links[link_id]
        dst = lk["end2"] if str(lk["end1"]) == str(src) else lk["end1"]

        cur_pos  = self.env.nodes[src]["pos"]
        to_pos   = self.env.nodes[dst]["pos"]
        prev_pos = self.env.nodes[prev]["pos"] if prev and prev != src else None
        movement = _movement_type(prev_pos, cur_pos, to_pos)

        t_w = self.env._calc_wait(src, abs_sec, movement)
        abs_depart = abs_sec + t_w

        v_ms = self._expected_speed_ms(link_id, abs_depart)
        # 드라이버 운동 모델(2026-05-21 개정)과 정합 — 회전 시 회전속도 진입,
        # 직진 시 직전 링크 순항속도 이어받음, 진출은 순항속도(노드 감속 없음).
        if movement == "left":
            v_in = min(V_TURN_LEFT, v_ms)
        elif movement == "right":
            v_in = min(V_TURN_RIGHT, v_ms)
        else:
            v_in = v_entry
        prof = SpeedProfile(v_ms, v_in, v_ms, lk["len"])
        t_tr = prof.total_time()

        # VT-Micro 출력 L/s → mL 환산 (env.step과 단위 정합)
        fuel = prof.total_fuel() * 1000.0 + fuel_idle(t_w) * 1000.0
        return fuel, t_w + t_tr

    def _run(self, src: str, abs_start: float,
             src_prev: Optional[str] = None):
        """
        Time-Dependent Dijkstra.

        state per node: (fuel, abs_t, v_exit, prev_node)
        prev_node 추적 → 좌/우 movement 판정에 사용.
        """
        # (fuel, abs_t, v_exit, prev_node)
        dist: dict[str, tuple[float, float, float, Optional[str]]] = {
            src: (0.0, abs_start, 5.0, src_prev)
        }
        prev: dict[str, Optional[str]] = {src: None}
        pq   = [(0.0, abs_start, 5.0, src, src_prev)]

        while pq:
            f, t, v, u, u_prev = heapq.heappop(pq)
            if f > dist.get(u, (float("inf"),))[0]:
                continue
            u_node = self.env.nodes[u]
            u_left_ok = _node_allows_left(u_node)
            nbs_all = self.env.adj.get(u, [])
            is_deadend = (len(nbs_all) <= 1)
            for nb, lid in nbs_all:
                # U턴 — degree-1 dead-end 에서만 허용 (env.get_valid_actions 와 정합).
                if nb == u_prev and not is_deadend:
                    continue
                # 좌회전 금지 노드의 좌회전 간선 제외 — env.get_valid_actions 와
                # 정합. 미반영 시 계산 경로가 실제 통행 불가 간선을 포함해 탈선.
                # U턴(nb==u_prev) 은 좌회전 분류 대상 아니므로 제외.
                if (not u_left_ok and u_prev is not None and u_prev != u
                        and nb != u_prev):
                    if _movement_type(self.env.nodes[u_prev]["pos"],
                                      u_node["pos"],
                                      self.env.nodes[nb]["pos"]) == "left":
                        continue
                nf, dt  = self._link_fuel(lid, t, v, src=u, prev=u_prev)
                total_f = f + nf
                if total_f < dist.get(nb, (float("inf"),))[0]:
                    v_out = self._expected_speed_ms(lid, t + dt)
                    dist[nb] = (total_f, t + dt, v_out, u)
                    prev[nb] = u
                    heapq.heappush(pq, (total_f, t + dt, v_out, nb, u))
        return prev

    def act(self, state: np.ndarray, valid_actions: list[str]) -> str:
        src      = self.env.current_node
        src_prev = self.env.previous_node if self.env.previous_node != src else None
        abs_now  = self.env.start_time_sec + self.env.current_time
        goals    = set(self.env.goal_nodes)
        prev     = self._run(src, abs_now, src_prev)

        reachable = [g for g in goals if g in prev]
        if not reachable:
            return valid_actions[0] if valid_actions else src

        # 연료 기준 최적 목표
        goal = reachable[0]
        path = []
        cur  = goal
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path = list(reversed(path))

        if len(path) < 2:
            return valid_actions[0] if valid_actions else src
        next_node = path[1]
        return next_node if next_node in valid_actions else (
            valid_actions[0] if valid_actions else src)
