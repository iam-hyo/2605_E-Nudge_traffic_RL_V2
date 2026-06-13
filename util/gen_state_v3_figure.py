"""State v3 (3-hop 구조형 확률) — 논문용 figure 도식화."""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, rcParams

# ── 한글 폰트 ──
_fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(_fp)
_name = font_manager.FontProperties(fname=_fp).get_name()
rcParams["font.family"] = _name
rcParams["axes.unicode_minus"] = False
print("font:", _name)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

def _gauss(x, mu, sd):
    return np.exp(-0.5*((x-mu)/sd)**2) / (sd*np.sqrt(2*np.pi))

# 팔레트
C_BG = "#ffffff"
C_GLOB = "#6b7280"
C_1HOP = "#2D6CDF"
C_2HOP = "#16a34a"
C_3HOP = "#ea7317"
C_INK = "#1f2430"
C_STAR = "#d6336c"

fig = plt.figure(figsize=(17, 10.5), dpi=200, facecolor=C_BG)
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0],
                      width_ratios=[1.45, 1.0, 1.0],
                      hspace=0.28, wspace=0.22,
                      left=0.035, right=0.975, top=0.9, bottom=0.06)

fig.suptitle("State v3 — 3-hop 구조형 확률 표현  (338차원)",
             fontsize=22, fontweight="bold", color=C_INK, y=0.965)
fig.text(0.5, 0.925,
         "실제 주행속도 미관측(평균만) → 도착시각 불확실 → 신호 통과가 확률 (POMDP).  "
         "각 경로 링크마다 통과확률·예상대기를 개별 표기",
         ha="center", fontsize=12.5, color="#555")

# ════════════════════════════════════════════════════════════
# Panel A — 결정 트리 (1→4→12→36 경로)
# ════════════════════════════════════════════════════════════
axA = fig.add_subplot(gs[0, :2]); axA.axis("off")
axA.set_xlim(0, 10); axA.set_ylim(0, 6)
axA.set_title("A.  경로 트리 — 위치가 경로(회전 시퀀스)를 인코딩",
              fontsize=14, fontweight="bold", loc="left", color=C_INK, pad=8)

# 현재 노드 (차량 진입)
x0 = 0.5
axA.add_patch(plt.Circle((x0, 3.0), 0.18, color=C_INK, zorder=5))
axA.annotate("", xy=(x0-0.05, 3.0), xytext=(x0-1.0, 3.0),
             arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=2.2))
axA.text(x0-1.05, 3.32, "링크 진입 시\n결정", fontsize=9.5, color=C_INK, ha="left")
axA.text(x0, 2.55, "현재 노드 N", fontsize=10, ha="center", color=C_INK, fontweight="bold")

x1, x2, x3 = 2.6, 5.4, 8.6
ys1 = np.linspace(5.2, 0.8, 4)            # 4 (1-hop)
np.random.seed(3)
# 1-hop 노드
pos2 = {}
for a, y1 in enumerate(ys1):
    axA.plot([x0+0.18, x1], [3.0, y1], color=C_1HOP, lw=1.8, alpha=0.85, zorder=2)
    axA.add_patch(plt.Circle((x1, y1), 0.13, color=C_1HOP, zorder=5))
    # 2-hop (각 1-hop당 3)
    ys2 = np.linspace(y1+0.55, y1-0.55, 3)
    pos2[a] = []
    for b, y2 in enumerate(ys2):
        axA.plot([x1, x2], [y1, y2], color=C_2HOP, lw=1.1, alpha=0.7, zorder=1)
        axA.add_patch(plt.Circle((x2, y2), 0.085, color=C_2HOP, zorder=4))
        pos2[a].append(y2)
        # 3-hop (각 2-hop당 3)
        ys3 = np.linspace(y2+0.16, y2-0.16, 3)
        for c, y3 in enumerate(ys3):
            axA.plot([x2, x3], [y2, y3], color=C_3HOP, lw=0.55, alpha=0.6, zorder=0)
            axA.add_patch(plt.Circle((x3, y3), 0.045, color=C_3HOP, zorder=3))

# 헤더 라벨
for xx, lab, n, col in [(x1, "1-hop", 4, C_1HOP), (x2, "2-hop", 12, C_2HOP),
                        (x3, "3-hop", 36, C_3HOP)]:
    axA.text(xx, 5.75, f"{lab}\n{n} 경로", ha="center", fontsize=11,
             fontweight="bold", color=col)
axA.text((x0+x1)/2, 5.78, "분기 ×4", ha="center", fontsize=9, color="#888")
axA.text((x1+x2)/2, 5.78, "×3", ha="center", fontsize=9, color="#888")
axA.text((x2+x3)/2, 5.78, "×3", ha="center", fontsize=9, color="#888")

# ── inset: 같은 노드, 다른 경로 (격자) ──
axI = axA.inset_axes([0.0, -0.04, 0.30, 0.40])
axI.set_xlim(-0.5, 2.5); axI.set_ylim(-0.5, 2.5); axI.axis("off")
axI.set_title("같은 노드 ≠ 같은 슬롯", fontsize=8.5, color=C_INK, pad=2)
for gx in range(3):
    for gy in range(3):
        axI.add_patch(plt.Circle((gx, gy), 0.06, color="#ccc", zorder=1))
# start at (0,1)
axI.add_patch(plt.Circle((0,1), 0.1, color=C_INK, zorder=3))
# path1: 직(→(1,1)) 좌(→(1,2))  파랑
axI.annotate("", xy=(1,1), xytext=(0,1), arrowprops=dict(arrowstyle="-|>", color=C_1HOP, lw=1.6))
axI.annotate("", xy=(1,2), xytext=(1,1), arrowprops=dict(arrowstyle="-|>", color=C_1HOP, lw=1.6))
# path2: 좌(→(0,2)) 우(→(1,2))  주황
axI.annotate("", xy=(0,2), xytext=(0,1), arrowprops=dict(arrowstyle="-|>", color=C_3HOP, lw=1.6))
axI.annotate("", xy=(1,2), xytext=(0,2), arrowprops=dict(arrowstyle="-|>", color=C_3HOP, lw=1.6))
axI.add_patch(plt.Circle((1,2), 0.12, fill=False, ec=C_STAR, lw=2, zorder=4))
axI.text(1.15, 2.25, "동일 노드", fontsize=7.5, color=C_STAR)
axI.text(2.5, 0.6, "직→좌", color=C_1HOP, fontsize=7.5, ha="right")
axI.text(2.5, 0.2, "좌→우", color=C_3HOP, fontsize=7.5, ha="right")
axI.text(-0.5, -0.5, "→ 경로(위치)로 표기하면 별도 슬롯, 합쳐지지 않음",
         fontsize=7.8, color=C_INK)

# ════════════════════════════════════════════════════════════
# Panel B — 엣지당 피처 스키마
# ════════════════════════════════════════════════════════════
axB = fig.add_subplot(gs[0, 2]); axB.axis("off")
axB.set_xlim(0, 10); axB.set_ylim(0, 10)
axB.set_title("B.  엣지당 피처 (떠먹임)", fontsize=14, fontweight="bold",
              loc="left", color=C_INK, pad=8)

def feat_card(ax, x, y, w, h, color, title, feats):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                 fc=color+"18", ec=color, lw=1.8))
    ax.text(x+0.2, y+h-0.32, title, fontsize=10.5, fontweight="bold", color=color)
    for i, (f, star) in enumerate(feats):
        ax.text(x+0.35, y+h-0.72-i*0.42, ("★ " if star else "•  ")+f,
                fontsize=8.6, color=(C_STAR if star else C_INK),
                fontweight=("bold" if star else "normal"))

feat_card(axB, 0.2, 6.7, 9.4, 3.1, C_1HOP, "1-hop  (4경로 × 8 = 32d)",
          [("valid · mv_left · mv_right", False), ("pass_prob  통과확률", True),
           ("exp_wait  예상대기", True), ("next_speed · next_len · goal_prog", False)])
feat_card(axB, 0.2, 3.5, 9.4, 3.0, C_2HOP, "2-hop  (12경로 × 7 = 84d)",
          [("valid · mv_left · mv_right", False), ("pass_prob  통과확률", True),
           ("exp_wait  예상대기", True), ("next_speed · goal_prog", False)])
feat_card(axB, 0.2, 0.5, 9.4, 2.8, C_3HOP, "3-hop  (36경로 × 6 = 216d)",
          [("valid · mv_left · mv_right", False), ("pass_prob  통과확률", True),
           ("exp_wait  예상대기", True), ("goal_prog", False)])

# ════════════════════════════════════════════════════════════
# Panel C — 왜 확률인가 (도착시각 분포 × 신호 사이클)
# ════════════════════════════════════════════════════════════
axC = fig.add_subplot(gs[1, 0])
axC.set_title("C.  pass_prob = 도착시각 분포 ∩ 녹색구간",
              fontsize=13, fontweight="bold", loc="left", color=C_INK, pad=8)
t = np.linspace(0, 60, 600)
# 신호 사이클: 0-25 녹색, 25-30 황, 30-55 적, 55-60 녹
for (a, b, c) in [(0, 25, "#2bbf5e"), (25, 30, "#f6c344"),
                  (30, 55, "#e5484d"), (55, 60, "#2bbf5e")]:
    axC.axvspan(a, b, ymin=0, ymax=0.18, color=c, alpha=0.85)
axC.text(12.5, -0.012, "녹색", ha="center", fontsize=8.5, color="#178a43")
axC.text(42.5, -0.012, "적색", ha="center", fontsize=8.5, color="#b3262b")
# 도착시각 분포 (평균 18s, σ=6)
mu, sd = 18, 7
y = _gauss(t, mu, sd); y = y/ y.max()*0.7 + 0.2
axC.plot(t, y, color=C_1HOP, lw=2.2)
axC.fill_between(t, 0.2, y, where=(t <= 25), color=C_1HOP, alpha=0.35)
axC.fill_between(t, 0.2, y, where=(t > 25), color="#e5484d", alpha=0.18)
axC.annotate("통과(녹색)\n≈ 0.83", xy=(15, 0.55), fontsize=9.5, color=C_1HOP,
             ha="center", fontweight="bold")
axC.annotate("정지", xy=(33, 0.30), fontsize=9, color="#b3262b", ha="center")
axC.text(30, 0.97, "속도 ±20% 노이즈 → 도착시각 퍼짐 → 통과가 확률",
         ha="center", fontsize=9.5, color="#555")
axC.set_xlim(0, 60); axC.set_ylim(0.0, 1.05)
axC.set_xlabel("노드 도착 시각 (s, 사이클 내)", fontsize=10)
axC.set_yticks([]); axC.spines[["top", "right", "left"]].set_visible(False)

# ════════════════════════════════════════════════════════════
# Panel D — 차원 분해 + 축소 옵션
# ════════════════════════════════════════════════════════════
axD = fig.add_subplot(gs[1, 1])
axD.set_title("D.  차원 분해", fontsize=13, fontweight="bold", loc="left",
              color=C_INK, pad=8)
segs = [("전역", 6, C_GLOB), ("1-hop", 32, C_1HOP),
        ("2-hop", 84, C_2HOP), ("3-hop", 216, C_3HOP)]
segs2 = [("전역", 6, C_GLOB), ("1-hop", 32, C_1HOP),
         ("2-hop", 84, C_2HOP), ("3-hop(cap)", 96, C_3HOP)]
for row, (label, data, total) in enumerate([("기본 (4·3·3)", segs, 338),
                                            ("축소 (4·2·2)", segs2, 218)]):
    yb = 1 - row
    left = 0
    for name, val, col in data:
        axD.barh(yb, val, left=left, color=col, edgecolor="white", height=0.55)
        if val > 20:
            axD.text(left+val/2, yb, f"{name}\n{val}", ha="center", va="center",
                     fontsize=8.3, color="white", fontweight="bold")
        left += val
    axD.text(left+6, yb, f"= {total}d", va="center", fontsize=10.5,
             fontweight="bold", color=C_INK)
    axD.text(-8, yb, label, va="center", ha="right", fontsize=9.5, color=C_INK)
axD.set_xlim(0, 360); axD.set_ylim(-0.6, 1.6); axD.axis("off")
axD.text(180, -0.45, "3-hop이 64% → 분기 cap(3→2)으로 216→96d 축소",
         ha="center", fontsize=9.2, color="#555")

# ════════════════════════════════════════════════════════════
# Panel E — 연료의 핵심: 정지 vs 대기
# ════════════════════════════════════════════════════════════
axE = fig.add_subplot(gs[1, 2])
axE.set_title("E.  왜 '통과확률'인가 — 정지 ≫ 대기",
              fontsize=13, fontweight="bold", loc="left", color=C_INK, pad=8)
bars = [("정지 1회\n(감속+재가속)", 17.6, C_STAR),
        ("대기 5초\n(공회전)", 1.9, "#9aa0aa"),
        ("등속 5초", 4.3, "#c7ccd4")]
xs = np.arange(len(bars))
axE.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.6)
for i, (lab, v, _) in enumerate(bars):
    axE.text(i, v+0.5, f"{v} mL", ha="center", fontsize=10, fontweight="bold",
             color=C_INK)
axE.set_xticks(xs); axE.set_xticklabels([b[0] for b in bars], fontsize=9)
axE.set_ylabel("연료 (mL)", fontsize=10); axE.set_ylim(0, 21)
axE.spines[["top", "right"]].set_visible(False)
axE.text(1.0, 19.2, "정지 1회 ≈ 공회전 47초.\n→ 대기시간 아닌 '통과확률'이 연료 직결",
         ha="center", fontsize=9, color=C_STAR, fontweight="bold")

fig.savefig("/tmp/state_v3_figure.png", dpi=200, bbox_inches="tight",
            facecolor=C_BG)
print("saved /tmp/state_v3_figure.png")
