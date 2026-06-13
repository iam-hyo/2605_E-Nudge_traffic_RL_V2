"""프로젝트 프레임워크 개요도 (논문 Figure 1 스타일)."""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, rcParams
_fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(_fp)
rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
rcParams["axes.unicode_minus"] = False

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# ── 팔레트 (절제된 학술 톤) ──
INK   = "#1f2733"
SUB   = "#5b6472"
BORD  = "#aab4c2"
ENVH  = "#2f5d8a"   # 환경 헤더 (네이비)
AGTH  = "#3a6b5c"   # 에이전트 헤더 (딥그린)
STH   = "#4a5a78"   # 상태
NETH  = "#6a4e8c"   # 신경망
ACTH  = "#b8762a"   # 행동
DEPH  = "#496074"   # 배포
DATA  = "#7a8694"   # 데이터
BODY  = "#fbfcfe"
ARR   = "#3a4250"

fig = plt.figure(figsize=(17, 9.2), dpi=200, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 17); ax.set_ylim(0, 9.2); ax.axis("off")

def box(x, y, w, h, header, lines=None, hc=ENVH, body_fs=10.5, head_fs=11.5,
        lh=0.34, align="left"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.10", fc=BODY, ec=BORD, lw=1.3, zorder=3))
    hh = 0.52
    ax.add_patch(FancyBboxPatch((x, y+h-hh), w, hh,
        boxstyle="round,pad=0,rounding_size=0.10", fc=hc, ec=hc, lw=0, zorder=4))
    # 헤더 하단 직각 보정
    ax.add_patch(plt.Rectangle((x, y+h-hh), w, hh*0.5, fc=hc, ec="none", zorder=4))
    ax.text(x+w/2, y+h-hh/2, header, color="white", fontsize=head_fs,
            fontweight="bold", ha="center", va="center", zorder=5)
    if lines:
        ty = y + h - hh - 0.42
        for ln in lines:
            ax.text(x+0.28 if align=="left" else x+w/2,
                    ty, ln, color=INK, fontsize=body_fs,
                    ha=align, va="top", zorder=5)
            ty -= lh
    return (x, y, w, h)

def arrow(p0, p1, label=None, rad=0.0, lw=2.4, color=ARR, fs=10, ls="-",
          loff=(0, 0.18)):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18,
        lw=lw, color=color, connectionstyle=f"arc3,rad={rad}", zorder=2,
        linestyle=ls)
    ax.add_patch(a)
    if label:
        mx, my = (p0[0]+p1[0])/2 + loff[0], (p0[1]+p1[1])/2 + loff[1]
        ax.text(mx, my, label, fontsize=fs, color=INK, ha="center", va="center",
                zorder=6, bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                    ec="none", alpha=0.9))

# ── 제목 ──
ax.text(0.4, 8.92, "신호·속도 인지 강화학습 기반 최소연료 경로 탐색 프레임워크",
        fontsize=17, fontweight="bold", color=INK, ha="left", va="top")

# ════ 입력 데이터 (상단) ════
dy, dh = 7.55, 0.86
dboxes = [("표준 노드·링크", "도로망 위상"),
          ("신호 현시·주기", "SPAT"),
          ("시간대 평균 속도", "speed.csv")]
dx = 0.55; dw = 1.55; gap = 0.18
for i, (t1, t2) in enumerate(dboxes):
    x = dx + i*(dw+gap)
    ax.add_patch(FancyBboxPatch((x, dy), dw, dh,
        boxstyle="round,pad=0,rounding_size=0.08", fc="#eef1f6", ec=BORD, lw=1.1, zorder=3))
    ax.text(x+dw/2, dy+dh*0.62, t1, fontsize=9.6, color=INK, ha="center", fontweight="bold")
    ax.text(x+dw/2, dy+dh*0.26, t2, fontsize=8.4, color=SUB, ha="center")
ax.text(0.55, dy+dh+0.18, "입력 데이터", fontsize=10.5, color=DATA, fontweight="bold")

# ════ 환경 ════
ex, ey, ew, eh = 0.55, 2.15, 5.1, 4.6
box(ex, ey, ew, eh, "환경 (Environment) · 부분관측 시뮬레이터",
    ["• 차량 동역학 + VT-Micro 연료 모델",
     "• 신호 대기 (movement별) · 비보호 우회전",
     "• 주행 속도 ±20% 확률 변동",
     "• 관측은 평균 속도만 — 실주행속도 은닉",
     "• 정지(감속·재가속)가 연료의 지배 요인"],
    hc=ENVH, body_fs=10.2, lh=0.62)

# 데이터 → 환경
for i in range(3):
    x = dx + i*(dw+gap) + dw/2
    arrow((x, dy), (x, ey+eh+0.02), lw=1.6)

# ════ 에이전트 ════
ax_, ay_, aw_, ah_ = 6.45, 1.45, 7.3, 6.05
ax.add_patch(FancyBboxPatch((ax_, ay_), aw_, ah_,
    boxstyle="round,pad=0,rounding_size=0.12", fc="#f4f7f6", ec=AGTH, lw=2.0, zorder=2))
ax.text(ax_+aw_/2, ay_+ah_-0.32, "강화학습 에이전트 (DQN)",
        fontsize=13, fontweight="bold", color=AGTH, ha="center", va="center", zorder=6)

# State
sx, sy, sw, sh = 6.85, 4.45, 6.5, 2.05
box(sx, sy, sw, sh, "상태 표현  (190차원)",
    ["전역(6): 시각 · 목적지 방위/거리",
     "엣지별: 통과확률 · 예상대기 · 속도 · 목표접근"],
    hc=STH, body_fs=9.6, lh=0.40)
# 미니 트리 (상태 박스 우측)
tx = sx + 4.55
for j, yy in enumerate(np.linspace(sy+0.45, sy+1.05, 1)):
    pass
root = (tx, sy+0.78)
ax.add_patch(Circle(root, 0.05, color=STH, zorder=6))
ys1 = np.linspace(sy+1.15, sy+0.35, 4)
for y1 in ys1:
    ax.plot([root[0], tx+0.5], [root[1], y1], color=STH, lw=0.8, alpha=0.7, zorder=5)
    ax.add_patch(Circle((tx+0.5, y1), 0.035, color="#16a34a", zorder=6))
    for y2 in np.linspace(y1+0.12, y1-0.12, 2):
        ax.plot([tx+0.5, tx+1.0], [y1, y2], color="#16a34a", lw=0.5, alpha=0.55, zorder=5)
        ax.add_patch(Circle((tx+1.0, y2), 0.022, color="#ea7317", zorder=6))
ax.text(tx+0.5, sy+1.32, "1·2·3-hop  (4·8·16 경로)", fontsize=7.8, color=SUB, ha="center")

# Network
nx, ny, nw, nh = 6.85, 2.95, 6.5, 1.2
box(nx, ny, nw, nh, "심층 Q-신경망",
    ["Double · Dueling DQN  (+ Attention)"],
    hc=NETH, body_fs=9.8, lh=0.36, align="center")

# Action
acx, acy, acw, ach = 6.85, 1.75, 6.5, 0.95
box(acx, acy, acw, ach, "행동  a",
    ["엣지-상대 슬롯 선택  argmax Q,  K=4"],
    hc=ACTH, body_fs=9.8, lh=0.32, align="center")

# 내부 흐름 화살표
arrow((sx+sw/2-1.8, sy), (nx+nw/2-1.8, ny+nh), lw=1.8)
arrow((nx+nw/2-1.8, ny), (acx+acw/2-1.8, acy+ach), lw=1.8)

# ════ 환경 ⇄ 에이전트 루프 ════
arrow((ex+ew, ey+eh-1.2), (ax_, sy+sh/2), "상태 s_t · 보상 r_t", lw=2.6, loff=(0, 0.22))
arrow((acx, acy+ach/2), (ex+ew-0.1, ey+0.9), "행동 a_t : 다음 진입 링크",
      rad=0.18, lw=2.6, loff=(0, -0.35))

# ════ 보상 (하단 띠) ════
rby = 0.45
ax.add_patch(FancyBboxPatch((ex, rby), aw_+ (ax_-ex) +0.0, 0.78,
    boxstyle="round,pad=0,rounding_size=0.08", fc="#fdf6ec", ec="#d9b873", lw=1.3, zorder=3))
ax.text(ex+0.25, rby+0.39,
        "보상  r = − 연료 (정지 회피 중심)  +  도착 보너스  +  경로 shaping        →  경험 재생 · 목표망 갱신",
        fontsize=10, color="#8a5a14", va="center", ha="left", fontweight="bold", zorder=5)
arrow((acx+acw/2, acy), (acx+acw/2, rby+0.78), lw=1.6)

# ════ 배포 ════
px, py, pw, ph = 14.15, 1.75, 2.55, 4.75
box(px, py, pw, ph, "배포",
    ["• 정책 전방 전개(unroll)", "  → 전체 경로 생성",
     "• 실시간 재탐색", "• 1-forward 추론 (ms)",
     "• 전역 최적화 불요"],
    hc=DEPH, body_fs=9.4, lh=0.55)
arrow((ax_+aw_, ny+nh/2), (px, py+ph-1.3), "학습 후", lw=2.0, loff=(0,0.22))

# 캡션
ax.text(0.55, 0.18, "그림 1.  전체 프레임워크 — 환경(POMDP)에서 평균 속도만 관측하는 에이전트가 "
        "구조형 확률 상태(190d)로 정지 회피 경로를 학습한다.",
        fontsize=9.5, color=SUB, ha="left", va="bottom")

fig.savefig("/tmp/framework_figure.png", dpi=200, bbox_inches="tight", facecolor="white")
print("saved")
