"""
generate_data_6x6.py
--------------------
6x6 = 36 노드 테스트베드 토폴로지 + 속도 데이터 생성기.

설계 목표:
  알고리즘 검증을 위해 "코어=병목 신호 / 외곽=유한 신호" 구조로 설계.
  좌하단(1) → 우중단(18) 경로 기준:
    - 직선 (코어 통과 6~7 step): 노드 ID 2~17 — 빡빡한 신호로 대기 ↑↑↑
    - 우회 (외곽 통과 9~10 step): 노드 ID 19~36 — 유한 신호로 거의 무대기
  → 신호 학습 모델이 우회로를 선택해야 연료 최적.

노드 ID 매핑 (1-based, ID = row*6 + col + 1):
  row=5: 31 32 33 34 35 36
  row=4: 25 26 27 28 29 30
  row=3: 19 20 21 22 23 24
  row=2: 13 14 15 16 17 18  ← 18=우중단 (goal)
  row=1:  7  8  9 10 11 12
  row=0:  1  2  3  4  5  6  ← 1=좌하단 (start)

신호 분류 (사용자 명시 "ID 2~17 빡빡" 정확 반영):
  - 무신호: {1, 6, 18, 31, 36}                       - 출발/도착/모서리
  - core_strong (강 병목): {8, 9, 10, 11, 14, 15, 16, 17}   - 코어 내부 8개
        cycle 100s, green  8s (8%), yellow 5s, red 87s, offset 랜덤(비동기)
  - core_weak   (약 병목): {2, 3, 4, 5, 7, 12, 13}          - 코어 외곽 7개
        cycle  80s, green 15s (19%), yellow 5s, red 60s, offset 랜덤
  - outer       (유한):    {19..30, 32, 33, 34, 35}        - 외곽 16개
        cycle  60s, green 45s (75%), left 5s, yellow 5s, red 5s, offset green-wave
        일부 (50%) 에 좌회전 phase

링크 속도 (외곽 1.0 기준):
  - core_strong 통과: 0.45  (매우 느림)
  - core_weak 통과:   0.65
  - outer 도로:       1.00

링크 길이:
  - 외곽 도로: 350~400m (짧음)
  - 코어 도로: 400~500m
"""

from __future__ import annotations
import json
import csv
import math
import os
import random
from pathlib import Path

# ── 상수 ──────────────────────────────────────────────────────────────────────
GRID_N        = 6
SPACING_M     = 400
NODE_ID_BASE  = 1
N_SLOTS       = 24
START_NODE_ID = "1"
GOAL_NODES    = ["18", "36"]

# 시간대별 기준 속도 (외곽 도로 기준, km/h)
BASE_SPEED_PROFILE = [
    48, 46, 44, 41, 37,          # 07:00~07:20
    33, 28, 24, 21, 19,          # 07:25~07:45
    18, 18, 19, 21, 24,          # 07:50~08:10  (피크)
    28, 32, 36, 40, 43,          # 08:15~08:35
    45, 47, 48, 49,              # 08:40~08:55
]

# 노드 ID 집합 (1-based)
NO_SIGNAL_NODES = {1, 6, 18, 31, 36}
CORE_STRONG_IDS = {8, 9, 10, 11, 14, 15, 16, 17}
CORE_WEAK_IDS   = {2, 3, 4, 5, 7, 12, 13}


def node_id(row: int, col: int) -> str:
    return str(NODE_ID_BASE + row * GRID_N + col)


def node_category(nid: int) -> str:
    if nid in NO_SIGNAL_NODES:
        return "none"
    if nid in CORE_STRONG_IDS:
        return "core_strong"
    if nid in CORE_WEAK_IDS:
        return "core_weak"
    return "outer"


# ── 신호 생성 ─────────────────────────────────────────────────────────────────
def make_signal(row: int, col: int, rng: random.Random):
    """반환: (signal_dict_or_None, left_turn_allowed: bool)."""
    nid = int(node_id(row, col))
    cat = node_category(nid)

    if cat == "none":
        # 무신호 — 모든 방향 통행 허용
        return None, True

    if cat == "core_strong":
        # 매우 빡빡: 110s 사이클 중 녹색 5s (4.5%), 적색 100s
        cycle = 110
        green, yellow, red = 5, 5, 100
        phases = [
            {"type": "green",  "duration": green},
            {"type": "yellow", "duration": yellow},
            {"type": "red",    "duration": red},
        ]
        return {"cycle_length": cycle, "offset": rng.randint(0, cycle - 1),
                "phases": phases}, False

    if cat == "core_weak":
        # 빡빡: 90s 사이클 중 녹색 10s (11%), 적색 75s
        cycle = 90
        green, yellow, red = 10, 5, 75
        phases = [
            {"type": "green",  "duration": green},
            {"type": "yellow", "duration": yellow},
            {"type": "red",    "duration": red},
        ]
        return {"cycle_length": cycle, "offset": rng.randint(0, cycle - 1),
                "phases": phases}, False

    # outer — 50% 확률로 좌회전 phase 포함
    has_left = (nid % 2 == 0)   # 결정론적 패턴 (재현성)
    if has_left:
        cycle = 65
        green, left, yellow, red = 40, 10, 5, 10
        phases = [
            {"type": "green",     "duration": green},
            {"type": "left_turn", "duration": left},
            {"type": "yellow",    "duration": yellow},
            {"type": "red",       "duration": red},
        ]
    else:
        cycle = 60
        green, yellow, red = 50, 5, 5
        phases = [
            {"type": "green",  "duration": green},
            {"type": "yellow", "duration": yellow},
            {"type": "red",    "duration": red},
        ]
    # green wave: col 진행 방향으로 offset 동기 (코어와 별개로 외곽 자체 동조)
    offset = (col * 4) % cycle
    return {"cycle_length": cycle, "offset": offset, "phases": phases}, has_left


# ── 도로 타입 / 속도 scale ────────────────────────────────────────────────────
def link_road_type(r1, c1, r2, c2) -> str:
    """링크의 도로 분류 — 두 끝점 중 더 빡빡한 쪽 기준."""
    nid1 = int(node_id(r1, c1))
    nid2 = int(node_id(r2, c2))
    cats = {node_category(nid1), node_category(nid2)}
    if "core_strong" in cats:
        return "core_strong"
    if "core_weak" in cats:
        return "core_weak"
    return "outer"


ROAD_TYPE_SCALE = {
    "core_strong": 0.35,    # 매우 느림 (코어 도심 정체)
    "core_weak":   0.55,
    "outer":       1.00,
}
ROAD_TYPE_NOISE = {
    "core_strong": 3.0,
    "core_weak":   2.5,
    "outer":       1.5,
}


# ── 토폴로지 생성 ────────────────────────────────────────────────────────────
def generate_topology(out_path: str):
    rng = random.Random(42)
    nodes = []
    stats = {"none": 0, "core_strong": 0, "core_weak": 0, "outer": 0, "lt_allowed": 0}

    for r in range(GRID_N):
        for c in range(GRID_N):
            sig, lt_ok = make_signal(r, c, rng)
            node = {
                "id":   node_id(r, c),
                "pos":  [c * SPACING_M, r * SPACING_M],
                "signal": sig,
                "left_turn_allowed": lt_ok,
            }
            nodes.append(node)
            stats[node_category(int(node["id"]))] += 1
            if lt_ok:
                stats["lt_allowed"] += 1

    links = []
    seen = set()
    rng2 = random.Random(99)

    def add_link(r1, c1, r2, c2):
        a, b = sorted([node_id(r1, c1), node_id(r2, c2)])
        lid = f"{a}_{b}"
        if lid in seen:
            return
        seen.add(lid)
        rtype = link_road_type(r1, c1, r2, c2)
        # 외곽 도로는 약간 짧게 (우회 매력 ↑)
        if rtype == "outer":
            ln = rng2.randint(350, 400)
        elif rtype == "core_weak":
            ln = rng2.randint(400, 460)
        else:  # core_strong
            ln = rng2.randint(420, 500)
        links.append({
            "id":        lid,
            "end1":      node_id(r1, c1),
            "end2":      node_id(r2, c2),
            "len":       ln,
            "road_type": rtype,
            "LANES":     "2" if rtype == "outer" else "1",
        })

    for r in range(GRID_N):
        for c in range(GRID_N):
            if c + 1 < GRID_N:
                add_link(r, c, r, c + 1)
            if r + 1 < GRID_N:
                add_link(r, c, r + 1, c)

    topology = {
        "metadata": {
            "start_node":  START_NODE_ID,
            "goal_nodes":  GOAL_NODES,
            "max_steps":   80,
            "start_hour":  7.0,
            "grid_size":   GRID_N,
            "spacing_m":   SPACING_M,
            "description": (
                "6x6 testbed (1..36). "
                "core_strong {8,9,10,11,14,15,16,17}: 100s cycle, 8% green. "
                "core_weak {2-5,7,12,13}: 80s cycle, 19% green. "
                "outer {19-30,32-35}: 60s cycle, ≥75% green w/ green-wave offset. "
                "no-signal {1,6,18,31,36}."
            ),
        },
        "nodes": nodes,
        "links": links,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(topology, f, ensure_ascii=False, indent=2)

    print(f"[6x6] topology → {out_path}")
    print(f"  nodes: {len(nodes)} {stats}")
    print(f"  links: {len(links)}  "
          f"(core_strong={sum(1 for l in links if l['road_type']=='core_strong')}, "
          f"core_weak={sum(1 for l in links if l['road_type']=='core_weak')}, "
          f"outer={sum(1 for l in links if l['road_type']=='outer')})")
    return topology


# ── 속도 CSV ──────────────────────────────────────────────────────────────────
def generate_speed_csv(topology: dict, out_path: str):
    rng = random.Random(123)
    rows = []
    for link in topology["links"]:
        lid   = link["id"]
        rtype = link.get("road_type", "outer")
        scale = ROAD_TYPE_SCALE.get(rtype, 1.0)
        sigma = ROAD_TYPE_NOISE.get(rtype, 1.5)
        offset_kh = rng.gauss(0, 1.0)

        row = {"link_id": lid}
        for t in range(N_SLOTS):
            base = BASE_SPEED_PROFILE[t] * scale + offset_kh
            v_kh = base + rng.gauss(0, sigma)
            v_kh = round(max(8.0, min(70.0, v_kh)), 2)
            row[f"t_{t}"] = v_kh
        rows.append(row)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fields = ["link_id"] + [f"t_{i}" for i in range(N_SLOTS)]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[6x6] speed → {out_path}  ({len(rows)} links, {N_SLOTS} slots)")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    topo = generate_topology(str(base / "data" / "6x6_topology.json"))
    generate_speed_csv(topo, str(base / "data" / "6x6_speed_data.csv"))
