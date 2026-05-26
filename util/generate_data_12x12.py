"""
generate_data_12x12.py
----------------------
12x12 = 144 노드 "중간 규모" 테스트베드 — 6x6_cross(36노드) → 강남구(1995노드)
전이를 위한 중간 단계 환경 (중간 발표용).

설계 의도 — 명령문 평가항목 5종 대응
====================================
1. 신호 회피 혜택이 뚜렷이 보이는 규모
   · 주 대각선 사선 코리도(n1→n144, 사선링크 11개)가 기하학적 최단(~4.7km)이지만
     모든 교차로가 4현시 대형신호 + local 저속도로 → "최단=신호지옥" 함정.
   · 외곽 간선(arterial) 우회로는 ~6.6km로 더 길지만 고속 + 신호 희소(무신호 다수)
     → 대기·연료가 적다. shortest 는 함정, 신호인식 모델은 우회 간선을 택해야 우위.
2. 현실 복잡도 — 144 노드, 격자 + 사선 코리도 혼합, 노드 위치 지터, 일부 열 사선화,
   도로 3등급(arterial/medium/local), 신호 3종(무신호/2현시/3현시) 혼재.
3. 강남구 신호·속도 반영
   · 신호: T-DATA SPAT 실측 기반 TRA_LIGHT 0/3/4 샘플링 로직(명령문 환경1).
       TRA_LIGHT=3 → 2현시 N(μ=131,σ=30) clip[60,180], green=cycle×0.321
       TRA_LIGHT=4 → 3현시 N(μ=180,σ=30) clip[80,180], green×0.256 left×0.144
       offset = randint(0, cycle//2)
   · 속도: gangnam_speed_data.csv 경향(평균 ~28km/h, 아침 첨두 정체) 반영 —
       등급별 off-peak/peak 기준속도 + slot 13(08:05) 첨두 비대칭 dip.
4. 링크 길이 ~260~470m, 속도 10~52km/h — 한국 도심 간선/이면도로 현실값.
5. 사선도로 — 주 대각 코리도(코너-코너 사선) + 5개 열 사선화로 "적당히 혼합".

좌표/ID  : ID = row*12 + col + 1.  출발 n1(0,0) 좌하단 → 목적 n144(11,11) 우상단.
차수 보장: 모든 노드 degree ≤ 4 (코리도 노드는 grid S·W 링크를 버리고 사선 2개 추가)
           → env.get_valid_actions 의 K_HOP1=4 슬롯이 잘리지 않음.
"""
from __future__ import annotations
import json
import math
import os
import random
from pathlib import Path

GRID  = 12
S     = 300.0          # 기본 노드 간격 (m)
START = "1"
GOAL  = "144"

# 간선(arterial) 행·열 — 빠르고 신호가 희소한 우회 통로
ART_ROWS = {0, 6, 11}
ART_COLS = {0, 6, 11}

# 일부 열을 사선화 (x 드리프트 = SLANT × (row-5.5) × S). 코리도 노드(r==c)는 제외.
SLANT_COLS = {1: 0.13, 3: -0.11, 5: 0.14, 8: -0.12, 10: 0.10}

# 도로 등급별 (off_peak, peak) 기준 속도 km/h — 강남 실측 수준으로 보정.
# 간선은 첨두에 2배 가까이 감속(피크/비피크 유의미한 차이), 이면도로(코리도)는
# 비첨두에도 느려 거리 함정이 양 시간대 모두 성립하도록 한다.
TIER_SPEED = {
    "arterial": (46.0, 25.0),   # 간선 — 빠름
    "medium":   (35.0, 16.0),   # 보조간선
    "local":    (19.0,  9.0),   # 이면도로 — 느림 (코리도)
}
TIER_LANES = {"arterial": "3", "medium": "2", "local": "1"}

# 신호 현시 비율 (명령문 환경1, T-DATA SPAT 실측 default 기반)
RATIO_T3 = {"green": 0.321}
RATIO_T4 = {"green": 0.256, "left": 0.144}

START_HOUR = 7.0
MAX_STEPS  = 150


# ── 좌표 / ID ────────────────────────────────────────────────────────────────
def nid(r: int, c: int) -> str:
    return str(r * GRID + c + 1)


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def node_xy(r: int, c: int, rng: random.Random) -> list[float]:
    """노드 좌표. 코리도 노드(r==c)는 깔끔한 대각선 유지(사선화 제외)."""
    if r == c:
        x = c * S
    else:
        x = c * S + SLANT_COLS.get(c, 0.0) * (r - 5.5) * S
    y = r * S
    return [round(x + rng.uniform(-20, 20), 1),
            round(y + rng.uniform(-20, 20), 1)]


# ── 신호 ─────────────────────────────────────────────────────────────────────
def assign_tra_light(r: int, c: int, rng: random.Random) -> int:
    """노드별 TRA_LIGHT 유형(0/3/4) 결정 — 강남 신호 분포 + 실험 구조 반영."""
    if (r, c) in ((0, 0), (GRID - 1, GRID - 1)):
        return 0                                  # 출발/도착 무신호
    if r == c:
        return 4                                  # 대각 코리도 = 신호 지옥
    on_ar, on_ac = r in ART_ROWS, c in ART_COLS
    if on_ar and on_ac:
        return 4                                  # 간선×간선 = 대형 교차로
    if on_ar or on_ac:
        # 간선 통로 (대형 교차로 사이) — 무신호 다수 + 일부 2현시
        return rng.choices([0, 3], weights=[0.62, 0.38])[0]
    # 내부 이면도로 격자 — 중심부일수록 3현시 신호 밀집
    d  = max(abs(r - 5.5), abs(c - 5.5)) / 5.5     # 0=중심 .. 1=가장자리
    p4 = 0.20 + 0.45 * (1.0 - d)
    p0 = 0.18
    p3 = max(0.0, 1.0 - p0 - p4)
    return rng.choices([0, 3, 4], weights=[p0, p3, p4])[0]


def make_signal(tra_light: int, rng: random.Random):
    """명령문 환경1 — TRA_LIGHT 유형별 신호 dict 생성. 반환 (signal|None, left_ok)."""
    if tra_light == 0:
        return None, True                         # 무신호 — 전방향 통행

    if tra_light == 3:                            # 2현시: 직진 + 적색
        cycle = int(round(_clip(rng.gauss(131, 30), 60, 180)))
        green = int(round(cycle * RATIO_T3["green"]))
        red   = cycle - green
        phases = [{"type": "green", "duration": green},
                  {"type": "red",   "duration": red}]
        left_ok = False                           # 2현시 — 좌회전 금지(현실 반영)
    else:                                         # 3현시: 직진 + 좌회전 + 적색
        cycle = int(round(_clip(rng.gauss(180, 30), 80, 180)))
        green = int(round(cycle * RATIO_T4["green"]))
        left  = int(round(cycle * RATIO_T4["left"]))
        red   = cycle - green - left
        phases = [{"type": "green", "duration": green},
                  {"type": "left",  "duration": left},
                  {"type": "red",   "duration": red}]
        left_ok = True

    offset = rng.randint(0, cycle // 2)            # 연동계획 부재 → 사이클 절반 균등
    return {"cycle_length": cycle, "offset": offset, "phases": phases}, left_ok


# ── 도로 등급 ────────────────────────────────────────────────────────────────
def link_tier(r1: int, c1: int, r2: int, c2: int) -> str:
    if r1 != r2 and c1 != c2:
        return "local"                            # 대각 코리도 = 저속
    if r1 == r2 and r1 in ART_ROWS:
        return "arterial"                         # 간선 가로
    if c1 == c2 and c1 in ART_COLS:
        return "arterial"                         # 간선 세로
    return "medium"


# ── 토폴로지 생성 ────────────────────────────────────────────────────────────
def generate_topology(out_path: str):
    pos_rng = random.Random(20260521)
    sig_rng = random.Random(770521)

    pos = {(r, c): node_xy(r, c, pos_rng)
           for r in range(GRID) for c in range(GRID)}

    nodes, tl_stat = [], {0: 0, 3: 0, 4: 0}
    for r in range(GRID):
        for c in range(GRID):
            tl = assign_tra_light(r, c, sig_rng)
            sig, left_ok = make_signal(tl, sig_rng)
            tl_stat[tl] += 1
            nodes.append({
                "id": nid(r, c),
                "pos": pos[(r, c)],
                "signal": sig,
                "left_turn_allowed": left_ok,
            })

    # ── 링크 ──────────────────────────────────────────────────────────────────
    # 코리도 노드 (k,k), k=1..10 은 grid S·W 링크를 버리고 사선 2개를 받는다
    # → degree = N + E grid + NE + SW 사선 = 4.
    corridor = {(k, k) for k in range(GRID)}
    drop = set()
    for k in range(1, GRID - 1):
        drop.add(tuple(sorted([nid(k - 1, k), nid(k, k)], key=int)))  # S of (k,k)
        drop.add(tuple(sorted([nid(k, k - 1), nid(k, k)], key=int)))  # W of (k,k)

    links, seen = [], set()

    def add_link(r1, c1, r2, c2):
        a, b = sorted([nid(r1, c1), nid(r2, c2)], key=int)
        lid = f"{a}_{b}"
        if lid in seen or (a, b) in drop:
            return
        seen.add(lid)
        p1, p2 = pos[(r1, c1)], pos[(r2, c2)]
        length = round(math.hypot(p2[0] - p1[0], p2[1] - p1[1]), 1)
        tier = link_tier(r1, c1, r2, c2)
        links.append({
            "id": lid, "end1": nid(r1, c1), "end2": nid(r2, c2),
            "len": length, "road_type": tier, "LANES": TIER_LANES[tier],
        })

    for r in range(GRID):
        for c in range(GRID):
            if c + 1 < GRID:
                add_link(r, c, r, c + 1)           # 가로 grid
            if r + 1 < GRID:
                add_link(r, c, r + 1, c)           # 세로 grid
    for k in range(GRID - 1):                      # 주 대각 코리도 사선
        add_link(k, k, k + 1, k + 1)

    # degree 검증
    deg = {n["id"]: 0 for n in nodes}
    for lk in links:
        deg[lk["end1"]] += 1
        deg[lk["end2"]] += 1
    maxdeg = max(deg.values())

    topology = {
        "metadata": {
            "start_node": START,
            "goal_nodes": [GOAL],
            "max_steps":  MAX_STEPS,
            "start_hour": START_HOUR,
            "grid_size":  GRID,
            "spacing_m":  int(S),
            "description": (
                "12x12 mid-scale testbed (144 nodes, 1..144). "
                "Main diagonal corridor n1->n144 = geometric shortest but "
                "all-4-phase signals + local-slow roads (signal trap). "
                "Arterial rows/cols {0,6,11} = longer detour but fast + "
                "sparse signals. Signals sampled per T-DATA SPAT logic "
                "(TRA_LIGHT 0/3/4). Goal: shortest picks the diagonal trap, "
                "signal-aware models pick the arterial detour."
            ),
        },
        "nodes": nodes,
        "links": links,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(topology, f, ensure_ascii=False, indent=2)

    tcnt = {}
    for lk in links:
        tcnt[lk["road_type"]] = tcnt.get(lk["road_type"], 0) + 1
    print(f"[12x12] topology → {out_path}")
    print(f"  nodes: {len(nodes)}  signal(TRA_LIGHT): "
          f"무신호={tl_stat[0]} 2현시={tl_stat[3]} 3현시={tl_stat[4]}")
    print(f"  links: {len(links)}  tiers={tcnt}  max_degree={maxdeg}")
    return topology


# ── 속도 CSV ─────────────────────────────────────────────────────────────────
def _dip(s: int) -> float:
    """slot s(0~23, 07:00~08:55)의 정체 강도 0~1. 첨두 ~slot 13(08:05),
    완만한 상승 + 더 느린 회복 — gangnam_speed_data.csv 경향(아침 첨두) 반영."""
    c = 13.0
    w = 6.0 if s <= c else 11.0
    return math.exp(-((s - c) / w) ** 2)


def generate_speed_csv(topo: dict, out_path: str):
    """링크별 24슬롯(5분, 07:00~08:55) 속도 CSV. 등급별 기준속도 + 첨두 dip +
    링크별 결정론적 변동(±10%) + 슬롯별 미세 노이즈(±4%)."""
    import csv
    rows = []
    for lk in topo["links"]:
        v_op, v_pk = TIER_SPEED[lk["road_type"]]
        rng = random.Random(hash(("v12", lk["id"])) & 0xFFFFFFFF)
        jitter = 1.0 + rng.uniform(-0.10, 0.10)
        speeds = []
        for s in range(24):
            v = v_op - (v_op - v_pk) * _dip(s)
            v *= jitter * (1.0 + rng.uniform(-0.04, 0.04))
            speeds.append(round(max(5.0, v), 2))
        rows.append([lk["id"], *speeds])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["link_id"] + [f"t_{i}" for i in range(24)])
        w.writerows(rows)
    print(f"[12x12] speed    → {out_path}  ({len(rows)} links)")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    topo = generate_topology(str(base / "data" / "12x12_topology.json"))
    generate_speed_csv(topo, str(base / "data" / "12x12_speed_data.csv"))
