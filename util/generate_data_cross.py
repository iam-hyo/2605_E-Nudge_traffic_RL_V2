"""
generate_data_cross.py
----------------------
6x6 = 36 노드 "사선 도로" 테스트베드 — 토폴로지 자기 모순 해소판.

설계 의도 (0520_2 실험 실패 교훈):
  이전 6x6 은 외곽 도로 link 길이를 짧게 만들어 Shortest Dijkstra 가
  link 길이만으로 외곽 우회를 자동 선택 → 신호 모델의 학습 여지 소실.

  본 cross 토폴로지는 link 길이(거리)와 신호 유불리를 **명시적으로 분리**:
    · 명시적 최단 경로 = 사선 B (col 1) — 거리는 짧지만 신호가 최악
    · 명시적 비최단 경로 = 사선 C (col 2) — 거리는 길지만 신호가 우호적
  → Shortest 는 B 를, Signal-aware 는 C 를 선택하게 유도.

구조:
  6개 사선 도로(col) × 6개 가로선(row) = 36 노드.  ID = row*6 + col + 1
  · 가로선: x축에 평행 (같은 row 노드 연결)
  · 세로선: 사선 (같은 col 의 row→row+1 연결, col 별 기울기 다름)
  · 노드 = 사선과 가로선의 교차점

사선별 (bottom y=0 → top y=5) x 좌표:
  col 0 (A): (0.0 → 0.0)   좌변, 수직
  col 1 (B): (1.0 → 2.0)   목적지(우상) 방향 → 명시적 최단, 신호 불리
  col 2 (C): (4.0 → 3.0)   반목적지(좌상) 방향 → 명시적 비최단, 신호 우호
  col 3 (D): (5.0 → 5.5)   목적지 방향 (short),  신호 불리
  col 4 (E): (6.5 → 6.0)   반목적지 방향 (non-short), 신호 우호
  col 5 (F): (8.0 → 7.0)   우변

출발 1 (0,0) → 목적 36 (7,5).

신호 설계:
  · col 1,3 (불리): cycle 120s, green 36s(30%), red 84s, offset = anti-sync
                    (사선 진행 시 매 노드 red 중간 도착 → 잦은 정지)
                    진입 노드(row 0) 좌회전 phase 8s (진입 자체 대기)
  · col 2,4 (우호): cycle 80s,  green 36s(45%), red 44s, offset = green-wave
                    (사선 진행 시 연속 green 통과 → 무정차)
                    진입 노드(row 0) 좌회전 phase 20s (진입 용이)
  · col 0,5: 무신호

속도 파일: 노드 인접 구조가 기존 6x6 과 동일(가로+col세로)하므로
           data/6x6_speed_data.csv 를 그대로 재사용 (재생성 불필요).
"""

from __future__ import annotations
import json
import math
import os
from pathlib import Path

GRID  = 6
SCALE = 350.0   # 좌표 단위 → m

# 사선별 (bottom_x, top_x) — y 는 row (0~5)
# 주: 사용자 명세 좌표(C bottom x=4.0)로는 peak 저속 시 C 경로의 거리 손해
# (~700m)가 신호 이득을 압도. C/E 사선을 수직에 가깝게 당겨 거리 격차를
# 신호로 만회 가능한 수준(약 300m)으로 조정. 목적지(x=7)에서 여전히 멀어
# "명시적 비최단" 성격은 유지.
SLANT_X = {
    0: (0.0, 0.0),
    1: (1.0, 2.0),   # B — 명시적 최단 (목적지 방향)
    2: (3.0, 3.0),   # C — 명시적 비최단 (수직, 목적지서 먼 위치)
    3: (5.0, 5.5),   # D — 목적지 방향 (short)
    4: (6.0, 6.0),   # E — 비최단 (수직)
    5: (8.0, 7.0),   # F — 우변
}

AVG_SPEED_MS = 20.0 / 3.6   # green-wave offset 추정용 평균 속도
                            # peak(병목) 시간대 속도 기준으로 맞춤 — peak 에서
                            # C 사선이 무정차가 되도록 (off_peak 는 더 빠르므로
                            # 약간 일찍 도착, green 50% 라 곧 통과)

FAVORABLE = {2, 4}   # 사선 C, E — 신호 우호 (비최단 경로)
UNFAVOR   = {1, 3}   # 사선 B, D — 신호 불리 (최단 경로)

START_NODE = "1"
GOAL_NODE  = "36"


def node_id(r: int, c: int) -> str:
    return str(r * GRID + c + 1)


def node_xy(r: int, c: int) -> list[float]:
    bx, tx = SLANT_X[c]
    x = (bx + (tx - bx) * r / 5.0) * SCALE
    y = r * SCALE
    return [round(x, 1), round(y, 1)]


def _slant_len(c: int, r: int) -> float:
    """col c 사선의 row r→r+1 링크 길이 (m)."""
    p1, p2 = node_xy(r, c), node_xy(r + 1, c)
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _cumulative_eta(c: int) -> list[float]:
    """col c 사선을 row 0 부터 진입할 때 각 row 노드 도착 누적 시각(s)."""
    eta = [0.0]
    for r in range(GRID - 1):
        eta.append(eta[-1] + _slant_len(c, r) / AVG_SPEED_MS)
    return eta


def make_signal(r: int, c: int, eta: list[float]):
    """반환: (signal_dict|None, left_turn_allowed)."""
    if c in (0, 5):
        return None, True

    arrival = eta[r]

    if c in FAVORABLE:
        # 우호: 70s 사이클 중 green 35s(50%) — green-wave 무정차.
        # green 50%는 직진신호 이론 상한 → offset 빗나가도 절반은 통과.
        cycle, green = 70, 35
        left = 24 if r == 0 else 0
        red  = cycle - green - left
        offset = int(round(-arrival)) % cycle
    else:
        # 불리: 200s 사이클 중 green 15s(7.5%) — anti-sync 잦은 장시간 정지.
        # C 경로의 거리 손해(약 700m, peak 저속 시 ~126mL)를 신호 대기로
        # 압도하도록 격차 극대화. green 15s 는 최소 duration 충족.
        cycle, green = 200, 15
        left = 8 if r == 0 else 0
        red  = cycle - green - left
        offset = int(round(-arrival + green + red / 2)) % cycle

    phases = [{"type": "green", "duration": green}]
    if left > 0:
        phases.append({"type": "left_turn", "duration": left})
    phases.append({"type": "red", "duration": red})
    return {"cycle_length": cycle, "offset": offset, "phases": phases}, (left > 0)


def _road_type(c1: int, c2: int) -> str:
    """링크 도로 타입 (json 메타용 — 속도는 csv 재사용이라 영향 없음)."""
    if c1 in FAVORABLE or c2 in FAVORABLE:
        return "arterial"
    return "local"


def generate_topology(out_path: str):
    # 사선별 누적 eta 미리 계산
    eta_by_col = {c: _cumulative_eta(c) for c in range(GRID)}

    nodes = []
    stats = {"none": 0, "favorable": 0, "unfavor": 0, "lt": 0}
    for r in range(GRID):
        for c in range(GRID):
            sig, lt = make_signal(r, c, eta_by_col[c])
            nodes.append({
                "id":   node_id(r, c),
                "pos":  node_xy(r, c),
                "signal": sig,
                "left_turn_allowed": lt,
            })
            if sig is None:
                stats["none"] += 1
            elif c in FAVORABLE:
                stats["favorable"] += 1
            else:
                stats["unfavor"] += 1
            if lt and sig is not None:
                stats["lt"] += 1

    links = []
    seen = set()

    def add_link(r1, c1, r2, c2):
        a, b = sorted([node_id(r1, c1), node_id(r2, c2)])
        lid = f"{a}_{b}"
        if lid in seen:
            return
        seen.add(lid)
        p1, p2 = node_xy(r1, c1), node_xy(r2, c2)
        length = round(math.hypot(p2[0] - p1[0], p2[1] - p1[1]), 1)
        links.append({
            "id": lid, "end1": node_id(r1, c1), "end2": node_id(r2, c2),
            "len": length, "road_type": _road_type(c1, c2),
            "LANES": "2",
        })

    for r in range(GRID):
        for c in range(GRID):
            if c + 1 < GRID:           # 가로선
                add_link(r, c, r, c + 1)
            if r + 1 < GRID:           # 사선 (같은 col)
                add_link(r, c, r + 1, c)

    topology = {
        "metadata": {
            "start_node": START_NODE,
            "goal_nodes": [GOAL_NODE],
            "max_steps": 80,
            "start_hour": 7.0,
            "grid_size": GRID,
            "spacing_m": int(SCALE),
            "description": (
                "6x6 cross-slant testbed (1..36). 6 slant roads × 6 horizontal lines. "
                "col1(B)=explicit-shortest/signal-hostile, col2(C)=explicit-detour/signal-friendly, "
                "col3(D)=hostile, col4(E)=friendly. Goal: Shortest picks B, Signal-aware picks C."
            ),
        },
        "nodes": nodes,
        "links": links,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(topology, f, ensure_ascii=False, indent=2)

    print(f"[cross] topology → {out_path}")
    print(f"  nodes: {len(nodes)} {stats}")
    print(f"  links: {len(links)}")
    return topology


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    generate_topology(str(base / "data" / "6x6_cross_topology.json"))
