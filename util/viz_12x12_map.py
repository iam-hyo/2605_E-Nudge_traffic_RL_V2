"""viz_12x12_map.py — 12x12 토폴로지 정적 지도 (신호 유형 · 도로 등급 · 코리도)."""
from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as _fm


def _pick_korean_font() -> str | None:
    """플랫폼별 한글(CJK) 폰트 선택. Noto Sans CJK 는 pan-CJK 라 JP face 도
    한글 글리프를 포함한다. (simulation.py 와 동일 패턴 — 한글 깨짐 방지)"""
    avail = {f.name for f in _fm.fontManager.ttflist}
    for cand in ("Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo",
                 "NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR",
                 "Noto Sans KR", "Noto Sans CJK JP"):
        if cand in avail:
            return cand
    return None


_KF = _pick_korean_font()
matplotlib.rcParams["font.family"] = [_KF, "DejaVu Sans"] if _KF else ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parent.parent
topo = json.load(open(ROOT / "data" / "12x12_topology.json", encoding="utf-8"))
pos  = {n["id"]: n["pos"] for n in topo["nodes"]}

TIER_C = {"arterial": "#2563c9", "medium": "#b9bdcc", "local": "#e0392b"}
SIG_C  = {0: "#9aa0b4", 3: "#1f9d55", 4: "#e8821f"}


def sig_type(n):
    s = n.get("signal")
    if s is None:
        return 0
    return 4 if any(p["type"] == "left" for p in s["phases"]) else 3


fig, ax = plt.subplots(figsize=(11, 11))
fig.patch.set_facecolor("white")

for lk in topo["links"]:
    p1, p2 = pos[lk["end1"]], pos[lk["end2"]]
    t = lk["road_type"]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=TIER_C[t],
            lw=3.4 if t == "local" else (2.2 if t == "arterial" else 1.0),
            zorder=2 if t == "local" else 1, solid_capstyle="round")

for n in topo["nodes"]:
    st = sig_type(n)
    x, y = n["pos"]
    ax.scatter(x, y, c=SIG_C[st], s=130 if st == 4 else (90 if st == 3 else 45),
               edgecolors="white", linewidths=1.0, zorder=4)

for nid, mk, col, lab in [("1", "*", "#16a858", "출발 n1"),
                          ("144", "*", "#d4a017", "목적 n144")]:
    ax.scatter(*pos[nid], marker=mk, s=900, c=col, edgecolors="black",
               linewidths=1.2, zorder=6)

ax.set_title("12x12 테스트베드 — 대각 코리도(적색·신호지옥·최단) vs 간선 우회로(청색)\n"
             "노드: 무신호(회) · 2현시(녹) · 3현시(주)", fontsize=12, pad=14)
ax.set_aspect("equal"); ax.axis("off")
leg = [mlines.Line2D([], [], color=TIER_C["local"], lw=3.4, label="local 코리도"),
       mlines.Line2D([], [], color=TIER_C["arterial"], lw=2.2, label="arterial 간선"),
       mlines.Line2D([], [], color=TIER_C["medium"], lw=1.0, label="medium"),
       mlines.Line2D([], [], color="none", marker="o", markerfacecolor=SIG_C[4],
                     ms=11, label="3현시(좌회전)"),
       mlines.Line2D([], [], color="none", marker="o", markerfacecolor=SIG_C[3],
                     ms=10, label="2현시"),
       mlines.Line2D([], [], color="none", marker="o", markerfacecolor=SIG_C[0],
                     ms=8, label="무신호")]
ax.legend(handles=leg, loc="upper left", fontsize=9, framealpha=0.95)

out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "output" / "12x12_topology_map.png")
Path(out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved → {out}")
