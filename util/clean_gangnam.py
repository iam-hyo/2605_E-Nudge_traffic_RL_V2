"""
clean_gangnam.py
----------------
강남구 실데이터 토폴로지 정제 — 다중 토폴로지 학습에 사용 가능하도록 보정.

원본 data/gangnam_topology.json 문제점:
  1. 링크 양 끝점 일부가 nodes 목록에 없음 (dangling) → RoadNetworkEnv 로드 시 KeyError
  2. link["len"] 값이 비현실적 (0.01~2.39) — 단위 불명, 다른 토폴로지(m)와 불일치
정제:
  - 양 끝점이 모두 nodes 에 존재하는 링크만 유지
  - 길이를 노드 좌표(위경도)로부터 haversine 으로 재계산 (m 단위, 다른 토폴로지와 정합)
  - 신호/메타데이터는 그대로 유지
출력: data/gangnam_clean_topology.json

사용: python util/clean_gangnam.py
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
SRC       = ROOT / "data" / "gangnam_topology.json"
DST       = ROOT / "data" / "gangnam_clean_topology.json"
SRC_SPEED = ROOT / "data" / "gangnam_speed_data.csv"
DST_SPEED = ROOT / "data" / "gangnam_clean_speed_data.csv"

EARTH_R = 6_371_000.0   # m


def haversine(p1: list[float], p2: list[float]) -> float:
    """p = [lat, lon] (deg) → 거리 (m)."""
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def main():
    topo = json.load(open(SRC, encoding="utf-8"))
    nodes = {str(n["id"]): n for n in topo["nodes"]}

    kept, dropped = [], 0
    for lk in topo["links"]:
        e1, e2 = str(lk["end1"]), str(lk["end2"])
        if e1 not in nodes or e2 not in nodes:
            dropped += 1
            continue
        length = max(haversine(nodes[e1]["pos"], nodes[e2]["pos"]), 1.0)
        kept.append({**lk, "end1": e1, "end2": e2, "len": round(length, 1)})

    clean = {
        "metadata": {**topo["metadata"],
                     "description": "gangnam (cleaned: dangling links removed, "
                                    "lengths recomputed by haversine)"},
        "nodes": topo["nodes"],
        "links": kept,
    }
    json.dump(clean, open(DST, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    lens = [l["len"] for l in kept]
    print(f"[clean_gangnam] → {DST}")
    print(f"  nodes {len(nodes)} | links {len(kept)} (dangling {dropped} 제거)")
    print(f"  len(m): min {min(lens):.1f}  max {max(lens):.1f}  "
          f"mean {sum(lens)/len(lens):.1f}")
    sig = sum(1 for n in topo['nodes'] if n.get('signal'))
    print(f"  signal nodes: {sig}")

    # ── 속도 CSV 변환: 원본 GIS 포맷 → env 포맷(link_id, t_0..t_23) ──────────
    # 원본 컬럼: LINK_ID + '07:00','07:05',...,'08:55' (5분 24슬롯)
    time_cols = [f"{h:02d}:{m:02d}" for h in (7, 8) for m in range(0, 60, 5)]
    n_spd = 0
    with open(SRC_SPEED, encoding="utf-8-sig") as fin, \
         open(DST_SPEED, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["link_id"] + [f"t_{i}" for i in range(24)])
        for row in reader:
            lid = (row.get("LINK_ID") or "").strip()
            if not lid:
                continue
            vals = []
            for tc in time_cols:
                try:
                    vals.append(round(max(5.0, float(row.get(tc, "") or 0)), 2))
                except ValueError:
                    vals.append(35.0)
            writer.writerow([lid] + vals)
            n_spd += 1
    print(f"[clean_gangnam] speed → {DST_SPEED}  ({n_spd} links)")


if __name__ == "__main__":
    main()
