"""OD별 전체 모델 주행 경로를 한 PNG에 표시 (results.csv path 사용)."""
import sys, csv, json
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, rcParams
_fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(_fp)
rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

import math
OUT = ROOT / "output" / "11_2225_specialized"
topo = json.load(open(ROOT / "data/gangnam_clean_topology.json", encoding="utf-8"))
# cos(위도) 경도 보정 — 위경도 왜곡 제거 (등거리 근사)
_lat0 = sum(n["pos"][1] for n in topo["nodes"]) / len(topo["nodes"])
LON_SCALE = math.cos(math.radians(_lat0))
POS = {str(n["id"]): (n["pos"][0] * LON_SCALE, n["pos"][1]) for n in topo["nodes"]}
LINKS = topo["links"]

STYLE = [  # (model, label, color, lw, ls, z)
    ("shortest_dijkstra",     "① Shortest",          "#9aa0aa", 2.2, "-",  3),
    ("static_fuel_dijkstra",  "② 연료 최소 Dijkstra", "#16a34a", 2.6, "-",  4),
    ("rl_base",               "③ RL Base",           "#ea7317", 2.2, "--", 5),
    ("rl_signal",             "④ RL Signal",         "#2176e8", 2.8, "-",  6),
    ("rl_signal_attention",   "⑤ RL Sig+Attn",       "#d6336c", 2.2, ":",  5),
]

def rep_path(rows):
    """도달한 run 중 연료 중앙값에 가장 가까운 경로."""
    rr = [r for r in rows if r["reached"] == "True"]
    if not rr: rr = rows
    fs = sorted(float(r["fuel_total"]) for r in rr)
    med = fs[len(fs)//2]
    best = min(rr, key=lambda r: abs(float(r["fuel_total"]) - med))
    return best["path"].split("->"), float(best["fuel_total"])

def draw(od, title, csv_path, pad=1.15, suffix=""):
    rows = list(csv.DictReader(open(csv_path)))
    by = defaultdict(list)
    for r in rows: by[r["model"]].append(r)

    fig = plt.figure(figsize=(10, 10), dpi=200)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.94]); ax.axis("off")
    # 배경 도로망 (옅게)
    for lk in LINKS:
        a, b = str(lk["end1"]), str(lk["end2"])
        if a in POS and b in POS:
            ax.plot([POS[a][0], POS[b][0]], [POS[a][1], POS[b][1]],
                    color="#e3e6ea", lw=0.5, zorder=1)
    # 모델 경로
    handles = []
    start = goal = None
    for m, lab, col, lw, ls, z in STYLE:
        if m not in by: continue
        path, fuel = rep_path(by[m])
        xs = [POS[n][0] for n in path if n in POS]
        ys = [POS[n][1] for n in path if n in POS]
        ax.plot(xs, ys, color=col, lw=lw, ls=ls, zorder=z, alpha=0.9,
                solid_capstyle="round")
        handles.append(plt.Line2D([], [], color=col, lw=lw, ls=ls,
                                  label=f"{lab}  ({fuel:.0f}mL)"))
        start, goal = path[0], path[-1]
    if start in POS:
        ax.scatter(*POS[start], s=260, marker="*", color="#16a858", zorder=10, ec="white")
    if goal in POS:
        ax.scatter(*POS[goal], s=260, marker="*", color="#e4a017", zorder=10, ec="white")
    handles += [plt.Line2D([], [], marker="*", color="#16a858", ms=14, ls="none", label="출발"),
                plt.Line2D([], [], marker="*", color="#e4a017", ms=14, ls="none", label="도착")]
    ax.legend(handles=handles, loc="upper right", fontsize=11, framealpha=0.95)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_aspect("equal")
    # 경로 영역 → 정사각 플롯 (좌우 짜부 방지)
    allx = [POS[n][0] for m,_,_,_,_,_ in STYLE if m in by for n in rep_path(by[m])[0] if n in POS]
    ally = [POS[n][1] for m,_,_,_,_,_ in STYLE if m in by for n in rep_path(by[m])[0] if n in POS]
    xc = (min(allx)+max(allx))/2; yc = (min(ally)+max(ally))/2
    half = max(max(allx)-min(allx), max(ally)-min(ally))/2 * pad
    ax.set_xlim(xc-half, xc+half); ax.set_ylim(yc-half, yc+half)
    out = OUT / f"allmodels_{od}_routes{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved", out)

draw("od1", "OD1 신논현→수서 — 전체 모델 주행 경로", OUT / "results_od1_full.csv")
draw("od2", "OD2 양재→영동 — 전체 모델 주행 경로", OUT / "results_od2_full.csv")
draw("od2", "OD2 양재→영동 — 전체 모델 주행 경로", OUT / "results_od2_full.csv", pad=1.7, suffix="_wide")
