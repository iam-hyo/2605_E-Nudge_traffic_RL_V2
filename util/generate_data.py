"""
generate_data.py
----------------
10x10 그리드 topology.json + speed_data.csv 자동 생성 스크립트.
- 노드: 100개, 6자리 ID (100101 ~ 101010)
- 링크: 양방향 (direction 컬럼 없음, 동일 평균 속도 사용)
- 신호: 약 70% 노드에 랜덤 신호 배정
- 속도: 출퇴근 병목(07:30~08:30)이 뚜렷하게 드러나는 프로파일

단독 실행: python util/generate_data.py
"""

import json
import csv
import random
import math
import os
from pathlib import Path

random.seed(42)

# ── 상수 ──────────────────────────────────────────────────────────────────────
GRID_N       = 10          # 10x10
SPACING_M    = 400         # 노드 간 간격 (m)
NODE_ID_BASE = 100101      # 시작 ID
SIGNAL_RATIO = 0.70        # 신호 교차로 비율
LINK_LEN_MIN = 350         # 링크 최소 길이 (m)
LINK_LEN_MAX = 500         # 링크 최대 길이 (m)
N_SLOTS      = 24          # 5분 × 24 = 07:00~08:55

# 출퇴근 속도 프로파일 기준값 (km/h) — 07:00부터 5분 단위
# 07:00~07:20 원활 → 07:25~08:20 병목 → 08:25~ 회복
BASE_SPEED_PROFILE = [
    48, 46, 44, 41, 37,          # 07:00~07:20  (원활→진입)
    33, 28, 24, 21, 19,          # 07:25~07:45  (병목 심화)
    18, 18, 19, 21, 24,          # 07:50~08:10  (첨두)
    28, 32, 36, 40, 43,          # 08:15~08:35  (회복)
    45, 47, 48, 49,              # 08:40~08:55  (안정)
]

# 도로 타입별 속도 스케일 (arterial이 더 빠름)
ROAD_TYPE_SCALE = {"arterial": 1.0, "local": 0.75}
ROAD_TYPE_NOISE = {"arterial": 2.0, "local": 3.5}   # σ (km/h)


def make_node_id(row: int, col: int) -> str:
    """(row, col) → 6자리 문자열 ID, 예: (0,0)→'100101', (9,9)→'101010'"""
    return str(NODE_ID_BASE + row * GRID_N + col)


def make_signal(rng: random.Random) -> dict | None:
    """랜덤 신호 생성. SIGNAL_RATIO 확률로 신호 있음."""
    if rng.random() > SIGNAL_RATIO:
        return None

    cycle = rng.choice([60, 80, 90, 100])
    offset = rng.randint(0, cycle - 1)
    has_left = rng.random() < 0.4   # 40% 확률로 좌회전 페이즈

    if has_left:
        green  = rng.randint(20, 35)
        left   = rng.randint(10, 20)
        yellow = 5
        red    = cycle - green - left - yellow
        if red < 10:
            red = 10
            green = cycle - left - yellow - red
        phases = [
            {"type": "green",     "duration": green},
            {"type": "left_turn", "duration": left},
            {"type": "yellow",    "duration": yellow},
            {"type": "red",       "duration": red},
        ]
    else:
        green  = rng.randint(30, 50)
        yellow = 5
        red    = cycle - green - yellow
        if red < 10:
            red = 10
            green = cycle - yellow - red
        phases = [
            {"type": "green",  "duration": green},
            {"type": "yellow", "duration": yellow},
            {"type": "red",    "duration": red},
        ]

    return {"cycle_length": cycle, "offset": offset, "phases": phases}


def make_link_id(id1: str, id2: str) -> str:
    a, b = sorted([id1, id2])
    return f"{a}_{b}"


def assign_road_type(row1, col1, row2, col2) -> str:
    """주요 도로(짝수 행/열)는 arterial, 나머지는 local"""
    if (row1 % 3 == 0 and row2 % 3 == 0) or (col1 % 3 == 0 and col2 % 3 == 0):
        return "arterial"
    return "local"


def generate_topology(out_path: str):
    rng = random.Random(42)
    nodes = []
    node_grid = {}   # (row, col) → node_id

    for r in range(GRID_N):
        for c in range(GRID_N):
            nid = make_node_id(r, c)
            node_grid[(r, c)] = nid
            nodes.append({
                "id":     nid,
                "pos":    [c * SPACING_M, r * SPACING_M],
                "signal": make_signal(rng),
            })

    links = []
    seen_links = set()

    for r in range(GRID_N):
        for c in range(GRID_N):
            nid = node_grid[(r, c)]
            # 오른쪽 이웃
            if c + 1 < GRID_N:
                nb = node_grid[(r, c + 1)]
                lid = make_link_id(nid, nb)
                if lid not in seen_links:
                    seen_links.add(lid)
                    rtype = assign_road_type(r, c, r, c + 1)
                    links.append({
                        "id":        lid,
                        "end1":      nid,
                        "end2":      nb,
                        "len":       rng.randint(LINK_LEN_MIN, LINK_LEN_MAX),
                        "road_type": rtype,
                        "LANES":     "2" if rtype == "arterial" else "1",
                    })
            # 아래쪽 이웃
            if r + 1 < GRID_N:
                nb = node_grid[(r + 1, c)]
                lid = make_link_id(nid, nb)
                if lid not in seen_links:
                    seen_links.add(lid)
                    rtype = assign_road_type(r, c, r + 1, c)
                    links.append({
                        "id":        lid,
                        "end1":      nid,
                        "end2":      nb,
                        "len":       rng.randint(LINK_LEN_MIN, LINK_LEN_MAX),
                        "road_type": rtype,
                        "LANES":     "2" if rtype == "arterial" else "1",
                    })

    # start: 좌상단, goal: 우하단
    start_node = node_grid[(0, 0)]
    goal_node  = node_grid[(GRID_N - 1, GRID_N - 1)]

    topology = {
        "metadata": {
            "start_node":  start_node,
            "goal_nodes":  [goal_node],
            "max_steps":   300,
            "start_hour":  7.0,
            "grid_size":   GRID_N,
            "spacing_m":   SPACING_M,
            "description": f"{GRID_N}x{GRID_N} grid network, {len(nodes)} intersections, {len(links)} links",
        },
        "nodes": nodes,
        "links": links,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(topology, f, ensure_ascii=False, indent=2)

    print(f"[generate_data] topology saved → {out_path}")
    print(f"  nodes: {len(nodes)}, links: {len(links)}")
    return topology


def generate_speed_csv(topology: dict, out_path: str):
    """
    링크별 평균 속도를 출퇴근 프로파일 기반으로 생성.
    - 양방향 동일 속도 (direction 컬럼 없음)
    - road_type에 따라 스케일 조정
    - 링크마다 독립적인 미세 노이즈 추가 (프로파일 형태 유지)
    """
    rng = random.Random(123)
    rows = []

    for link in topology["links"]:
        lid   = link["id"]
        rtype = link.get("road_type", "local")
        scale = ROAD_TYPE_SCALE[rtype]
        noise_sigma = ROAD_TYPE_NOISE[rtype]

        # 링크별 고유 오프셋 (±3 km/h) — 링크마다 다른 기저 속도
        link_offset = rng.gauss(0, 3.0)

        row = {"link_id": lid}
        for t_idx in range(N_SLOTS):
            base   = BASE_SPEED_PROFILE[t_idx] * scale + link_offset
            # 슬롯마다 미세 노이즈 (프로파일 형태는 유지)
            speed  = base + rng.gauss(0, noise_sigma)
            speed  = round(max(10.0, min(70.0, speed)), 2)
            row[f"t_{t_idx}"] = speed

        rows.append(row)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["link_id"] + [f"t_{i}" for i in range(N_SLOTS)]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[generate_data] speed_data saved → {out_path}")
    print(f"  links: {len(rows)}, slots: {N_SLOTS} (07:00~08:55, 5min)")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    topo = generate_topology(str(base / "data" / "10x10_topology.json"))
    generate_speed_csv(topo, str(base / "data" / "speed_data.csv"))
