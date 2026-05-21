"""
generate_data_cross.py
----------------------
6x6 = 36 노드 "사선 도로" 테스트베드 (2026-05-21 재설계판).

설계 의도
=========
신호를 학습한 모델이 "거리는 가장 길지만 신호가 가장 우호적인" 세로 도로
col 2 (n3→n33) 를 채택하도록 유도한다. 단순 최단거리 모델은 짧은 도로(col 1)를,
무신호 도로에 의존하던 기존 행동은 더 이상 통하지 않게(거의 모든 노드에 신호)
만든다.

좌표 (사용자 명세, n1..n6 = row0 / n31..n36 = row5)
  col 0 : n1(0,0)   → n31(0,5)     수직 좌변
  col 1 : n2(1,0)   → n32(2.9,5)   명시적 최단 경로, 신호 불리, n2 좌회전 대기 길게
  col 2 : n3(4,0)   → n33(3.1,5)   명시적 최장 경로, 신호 우호(green-wave), 우회도로
  col 3 : n4(5,0)   → n34(5.5,5)   신호 약간 불친절
  col 4 : n5(6.5,0) → n35(6,5)     신호 꽤 친절(불완전)
  col 5 : n6(7,0)   → n36(7,5)     수직 우변
  · col 1 top / col 2 top 은 명세상 (3,5) 로 동일 → 길이 0 링크를 피하려고
    2.9 / 3.1 로 0.2 만큼만 분리 (사선이 정상부에서 거의 만나는 cross 형태 유지).

출발 n1(0,0) → 목적 n36(7,5).  ID = row*6 + col + 1.

신호 설계 (거의 모든 노드 = 36/36 에 신호, 모두 [green, left, red] 3-phase)
  · col 2 friendly  : green-wave (offset 위상 고정) → 무정차 통과, red 12s 로 최악도 짧음
  · col 1 hostile   : 사이클 220s·green 16s, 매 노드 red 중심 도착 → 반복 장시간 정지
  · col 0,5 moderate: 매 노드 red 후반 도착 → 중간 정지
  · col 3 slight    : 매 노드 red 중간 도착 → 약한 정지
  · col 4 quite     : 매 노드 red 후반 도착 → 짧은 정지 (col1/3 보다 확연히 우호적)
  좌회전 phase 는 36/36 모두 보유 (>50% 요건 충족). col 1 은 left 8s 로 짧아
  진입(좌회전) 대기가 길다.

green-wave / anti-wave 오프셋
  한 노드를 green 시작에 출발한 차량이 다음 노드에 도착할 때의 위상을
  rf(red landing fraction) 로 직접 지정한다.
    offset_r = base - eta[r] + r*SHIFT      (mod cycle)
    friendly : SHIFT=0,                     base=green/2          → 매 노드 green
    그 외    : SHIFT = base = green+left+red*rf                   → 매 노드 red 의 rf 지점
  rf 이 작을수록(=red 앞쪽 도착) 잔여 red 가 길어 대기가 커진다.

속도 파일: 인접 구조가 기존 6x6 격자와 동일 → data/6x6_speed_data.csv 재사용.
"""

from __future__ import annotations
import json
import math
import os
from pathlib import Path

GRID  = 6
SCALE = 350.0   # 좌표 단위 → m

# 사선별 (bottom_x, top_x) — y 는 row (0~5)
SLANT_X = {
    0: (0.0, 0.0),
    1: (1.0, 2.9),   # B — 명시적 최단 (목적지 방향), 신호 불리
    2: (4.0, 3.1),   # C — 명시적 최장 (우회도로), 신호 우호  ← 학습 모델이 채택해야
    3: (5.0, 5.5),   # D — 신호 약간 불친절
    4: (6.5, 6.0),   # E — 신호 꽤 친절(불완전)
    5: (7.0, 7.0),   # F — 우변
}

# green-wave 오프셋 추정용 평균 속도 (peak/병목 시간대 기준 ≈ 13 km/h)
AVG_SPEED_MS = 13.0 / 3.6

# 사선별 신호 성격.  cycle = green + left + red.
#   rf=None → friendly green-wave.  rf∈(0,1) → red 의 rf 지점 도착(작을수록 불리).
PERSONA = {
    0: dict(cycle=180, green=24, left=12, red=144, rf=0.22, tag="hostile"),
    1: dict(cycle=220, green=16, left=8,  red=196, rf=0.22, tag="hostile"),
    2: dict(cycle=70,  green=44, left=14, red=12,  rf=None, tag="friendly"),
    3: dict(cycle=150, green=34, left=12, red=104, rf=0.34, tag="slight"),
    4: dict(cycle=110, green=42, left=14, red=54,  rf=0.48, tag="quite"),
    5: dict(cycle=170, green=28, left=12, red=130, rf=0.46, tag="moderate"),
}

FRIENDLY = {2, 4}    # 신호 우호 사선 (메타/도로타입용)

# 사선별 도로 속도 등급 — col 2 만 간선(arterial, 빠름).
#   현실 서사: col 2 = 신호 연동(green-wave) 된 빠른 간선도로 / col 0,1 = 정체된
#   느린 이면도로. → "가장 길지만 가장 빠르고 연료 효율적인 길" 을 학습하게 됨.
COL_TIER = {0: "local", 1: "local", 2: "arterial",
            3: "medium", 4: "medium", 5: "medium"}
# 등급별 (off_peak km/h, peak km/h) 기준 속도
TIER_SPEED = {
    "arterial": (38.0, 26.0),
    "medium":   (27.0, 14.0),
    "local":    (20.0,  8.5),
}

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


def make_signal(r: int, c: int, eta: list[float]) -> dict:
    """col c, row r 노드의 신호 dict 반환 (모든 노드 신호 보유)."""
    p     = PERSONA[c]
    cycle = p["cycle"]
    green = p["green"]
    left  = p["left"]
    red   = p["red"]

    if p["rf"] is None:                       # friendly green-wave
        base, shift = green / 2.0, 0.0
    else:                                     # anti-wave
        base = shift = green + left + red * p["rf"]

    offset = int(round(base - eta[r] + r * shift)) % cycle

    phases = [
        {"type": "green",     "duration": green},
        {"type": "left_turn", "duration": left},
        {"type": "red",       "duration": red},
    ]
    return {"cycle_length": cycle, "offset": offset, "phases": phases}


def _road_type(c1: int, c2: int) -> str:
    return "arterial" if (c1 in FRIENDLY or c2 in FRIENDLY) else "local"


def generate_topology(out_path: str):
    eta_by_col = {c: _cumulative_eta(c) for c in range(GRID)}

    nodes = []
    stats = {p["tag"]: 0 for p in PERSONA.values()}
    for r in range(GRID):
        for c in range(GRID):
            sig = make_signal(r, c, eta_by_col[c])
            nodes.append({
                "id":   node_id(r, c),
                "pos":  node_xy(r, c),
                "signal": sig,
                "left_turn_allowed": True,   # 36/36 좌회전 phase 보유
            })
            stats[PERSONA[c]["tag"]] += 1

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
            "len": length, "road_type": _road_type(c1, c2), "LANES": "2",
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
                "6x6 cross-slant testbed v3 (1..36, 36/36 signalized). "
                "col1(B)=explicit-shortest/signal-hostile, "
                "col2(C)=explicit-longest/signal-friendly green-wave (detour), "
                "col3(D)=slight-hostile, col4(E)=quite-friendly. "
                "Goal: shortest model picks B, signal-aware model picks C."
            ),
        },
        "nodes": nodes,
        "links": links,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(topology, f, ensure_ascii=False, indent=2)

    print(f"[cross v3] topology → {out_path}")
    print(f"  nodes: {len(nodes)}  signal personas: {stats}")
    print(f"  links: {len(links)}  (모든 노드 신호 + 좌회전 phase 보유)")
    return topology


def _link_tier(e1: str, e2: str) -> str:
    """링크 도로 등급 — 세로(사선) 링크는 col 등급, 가로 링크는 medium."""
    c1 = (int(e1) - 1) % GRID
    c2 = (int(e2) - 1) % GRID
    return COL_TIER[c1] if c1 == c2 else "medium"


def generate_speed_csv(topo: dict, out_path: str):
    """
    링크별 24개(5분 슬롯, 07:00~08:55) 속도 CSV 생성.
    등급별 기준 속도 + slot 12(08:00) 중심 peak 정체 + 링크별 결정론적 변동.
    """
    import csv
    import random as _rnd

    rows = []
    for lk in topo["links"]:
        tier = _link_tier(str(lk["end1"]), str(lk["end2"]))
        v_op, v_pk = TIER_SPEED[tier]
        rng = _rnd.Random(hash(("speed", lk["id"])) & 0xFFFFFFFF)
        jitter = 1.0 + rng.uniform(-0.12, 0.12)      # 링크별 ±12% 고정 변동
        speeds = []
        for s in range(24):
            dip = math.exp(-((s - 12) / 6.0) ** 2)   # slot 12 에서 1.0
            v   = (v_op - (v_op - v_pk) * dip) * jitter
            # 슬롯별 미세 노이즈 (결정론적)
            v  *= 1.0 + rng.uniform(-0.05, 0.05)
            speeds.append(round(max(5.0, v), 2))
        rows.append([lk["id"], *speeds])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["link_id"] + [f"t_{i}" for i in range(24)])
        w.writerows(rows)
    print(f"[cross v3] speed   → {out_path}  ({len(rows)} links)")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    topo = generate_topology(str(base / "data" / "6x6_cross_topology.json"))
    generate_speed_csv(topo, str(base / "data" / "6x6_cross_speed_data.csv"))
