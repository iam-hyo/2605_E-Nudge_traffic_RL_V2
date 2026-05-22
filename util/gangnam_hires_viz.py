"""
gangnam_hires_viz.py
--------------------
강남구(1995 노드) 고해상도 시각화 방법론 MVP.

기존 simulation.py 의 한계
  · 5km 도로망 전체를 1296px·72dpi 한 프레임에 렌더 → 노드가 sub-pixel,
    구조 파악 불가. 확대하면 래스터라 픽셀 열화로 경로 식별 불가.

본 모듈의 방법론
  1. 미터 투영 — 위경도를 등거리 평면 좌표(m)로 투영해 왜곡 없는 비율.
  2. 자유 확대 — render_window(center, half_width_m) 가 임의 영역을
     "프레임 픽셀 전부"로 그려, 어느 배율에서도 선명(벡터 품질).
  3. 카메라 자동 이동 — 경로를 따라 카메라 중심이 이동/줌하는 GIF 생성.
  4. 고DPI 정적 산출물 — 전체도 + 줌 인셋.

사용: python util/gangnam_hires_viz.py <out_dir>
"""
from __future__ import annotations
import io
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as _fm


def _pick_korean_font() -> str | None:
    avail = {f.name for f in _fm.fontManager.ttflist}
    for c in ("Malgun Gothic", "AppleGothic", "NanumGothic",
              "Noto Sans CJK KR", "Noto Sans KR", "Noto Sans CJK JP"):
        if c in avail:
            return c
    return None


_KF = _pick_korean_font()
matplotlib.rcParams["font.family"] = [_KF, "DejaVu Sans"] if _KF else ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt   # noqa: E402
import matplotlib.lines as mlines  # noqa: E402

try:
    from PIL import Image
    _PIL = True
except ImportError:
    _PIL = False

ROOT = Path(__file__).resolve().parent.parent


# ── 데이터 로드 + 미터 투영 ───────────────────────────────────────────────────
class GangnamMap:
    """강남구 토폴로지를 미터 평면으로 투영해 보관."""

    def __init__(self, topo_path: str):
        topo = json.load(open(topo_path, encoding="utf-8"))
        self.meta  = topo["metadata"]
        self.nodes = {str(n["id"]): n for n in topo["nodes"]}
        self.links = [(str(l["end1"]), str(l["end2"])) for l in topo["links"]]

        lats = [n["pos"][0] for n in self.nodes.values()]
        lons = [n["pos"][1] for n in self.nodes.values()]
        self.lat0, self.lon0 = sum(lats) / len(lats), sum(lons) / len(lons)
        self._mx = math.cos(math.radians(self.lat0)) * 111_320.0
        self._my = 110_540.0

        self.xy = {nid: self._proj(n["pos"]) for nid, n in self.nodes.items()}

        # 차수 / 좌회전 허용
        self.deg: dict[str, int] = {nid: 0 for nid in self.nodes}
        for e1, e2 in self.links:
            self.deg[e1] += 1
            self.deg[e2] += 1

    def _proj(self, pos):
        lat, lon = pos
        return ((lon - self.lon0) * self._mx, (lat - self.lat0) * self._my)

    def left_ok(self, nid: str) -> bool:
        return bool(self.nodes[nid].get("left_turn_allowed"))

    def has_signal(self, nid: str) -> bool:
        return self.nodes[nid].get("signal") is not None


# ── 임의 영역 렌더 (자유 확대의 핵심) ─────────────────────────────────────────
def render_window(ax, gm: GangnamMap, cx: float, cy: float, half_w: float,
                  route=None, agent_path=None, infeasible=None,
                  start=None, goal=None, show_deadend=False, title=""):
    """
    중심 (cx,cy), 반경 half_w(m) 영역을 ax 에 렌더.
    어느 배율이든 프레임 픽셀 전체를 사용 → 항상 선명.
    """
    ax.clear()
    ax.set_facecolor("#eef0f5")
    ax.set_xlim(cx - half_w, cx + half_w)
    # 화면 비율(가로:세로)에 맞춰 세로 반경 조정
    bb = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    aspect = (fh * bb.height) / (fw * bb.width)
    half_h = half_w * aspect
    ax.set_ylim(cy - half_h, cy + half_h)
    # set_aspect 미사용 — half_h 가 이미 축 박스 비율과 일치(등축척) +
    # 박스를 꽉 채움. set_aspect("equal")은 박스를 수축시켜 여백을 만든다.
    ax.axis("off")

    # 배율에 따라 선·마커 굵기 자동 (확대할수록 굵게)
    z = max(0.3, min(3.5, 700.0 / half_w))
    margin = half_w * 1.4

    # 링크 — 뷰 안의 것만
    for e1, e2 in gm.links:
        x1, y1 = gm.xy[e1]
        x2, y2 = gm.xy[e2]
        if (max(x1, x2) < cx - margin or min(x1, x2) > cx + margin or
                max(y1, y2) < cy - margin or min(y1, y2) > cy + margin):
            continue
        ax.plot([x1, x2], [y1, y2], color="#c2c6d4",
                lw=0.9 * z, zorder=1, solid_capstyle="round")

    # 막다른(degree-1) 노드 강조
    if show_deadend:
        dx = [gm.xy[n][0] for n, d in gm.deg.items() if d == 1]
        dy = [gm.xy[n][1] for n, d in gm.deg.items() if d == 1]
        ax.scatter(dx, dy, s=10 * z, c="#e8521f", marker="x",
                   linewidths=1.2 * z, zorder=3, label="막다른 노드(degree 1)")

    # Dijkstra 경로
    if route:
        rx = [gm.xy[n][0] for n in route]
        ry = [gm.xy[n][1] for n in route]
        ax.plot(rx, ry, "-", color="#ffffff", lw=6.5 * z, zorder=4,
                solid_capstyle="round", solid_joinstyle="round")
        ax.plot(rx, ry, "-", color="#2176e8", lw=3.6 * z, zorder=5,
                solid_capstyle="round", solid_joinstyle="round",
                label="Dijkstra 최단경로")

    # 에이전트 실제 주행(이탈) 경로
    if agent_path:
        ax_ = [gm.xy[n][0] for n in agent_path if n in gm.xy]
        ay_ = [gm.xy[n][1] for n in agent_path if n in gm.xy]
        ax.plot(ax_, ay_, "--", color="#e8821f", lw=3.0 * z, zorder=6,
                dashes=(3, 2), label="에이전트 실제 주행(이탈)")

    # infeasible 좌회전 노드
    if infeasible:
        ix = [gm.xy[n][0] for n in infeasible if n in gm.xy]
        iy = [gm.xy[n][1] for n in infeasible if n in gm.xy]
        ax.scatter(ix, iy, s=240 * z, facecolors="none", edgecolors="#e0202a",
                   linewidths=2.6 * z, zorder=8, label="좌회전 금지 위반 노드")
        ax.scatter(ix, iy, s=40 * z, c="#e0202a", marker="x",
                   linewidths=2.2 * z, zorder=9)

    if start and start in gm.xy:
        ax.scatter(*gm.xy[start], s=420 * z, c="#16a858", marker="*",
                   edgecolors="white", linewidths=1.3 * z, zorder=10, label="출발")
    if goal and goal in gm.xy:
        ax.scatter(*gm.xy[goal], s=420 * z, c="#e4a017", marker="*",
                   edgecolors="white", linewidths=1.3 * z, zorder=10, label="목적지")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color="#22242a", pad=8)


# ── 카메라 자동 이동 GIF (방법론 MVP 핵심 데모) ───────────────────────────────
def camera_gif(gm: GangnamMap, route, out_path: str, agent_path=None,
               infeasible=None, start=None, goal=None,
               px=900, dpi=110, fps=20):
    """경로를 따라 카메라가 줌인 → 자동 이동하는 GIF.
    각 프레임이 좁은 영역을 픽셀 전부로 렌더 → 어디서나 선명."""
    if not _PIL:
        print("  [경고] Pillow 미설치 — 카메라 GIF 생략")
        return
    pts = [gm.xy[n] for n in route]
    # 경로를 따라 일정 간격으로 카메라 중심 샘플
    seg = []
    for i in range(len(pts) - 1):
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        d = math.hypot(x2 - x1, y2 - y1)
        steps = max(2, int(d / 18))
        for s in range(steps):
            t = s / steps
            seg.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
    seg.append(pts[-1])

    infeas_xy = [gm.xy[n] for n in (infeasible or []) if n in gm.xy]
    fig = plt.figure(figsize=(px / dpi, px / dpi), dpi=dpi)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.92])
    frames = []

    full_w = max(max(x for x, _ in pts) - min(x for x, _ in pts),
                 max(y for _, y in pts) - min(y for _, y in pts)) / 2 + 300
    n_zoom = 26                              # 줌인 프레임 수
    cx0, cy0 = pts[0]

    total = n_zoom + len(seg)
    for fi in range(total):
        if fi < n_zoom:                      # ── 1단계: 전체→출발점 줌인
            t = fi / n_zoom
            half_w = full_w * (1 - t) + 240 * t
            ccx = (cx0 + cx0) / 2 * t + (min(x for x, _ in pts) +
                   max(x for x, _ in pts)) / 2 * (1 - t)
            ccy = (cy0) * t + (min(y for _, y in pts) +
                   max(y for _, y in pts)) / 2 * (1 - t)
            cap = "1단계 — 자유 확대: 전체 도로망에서 출발점으로 줌인"
        else:                                # ── 2단계: 경로 따라 카메라 이동
            ccx, ccy = seg[fi - n_zoom]
            half_w = 240
            cap = "2단계 — 카메라 자동 이동: 경로를 따라가며 교차로 확인"
        render_window(ax, gm, ccx, ccy, half_w, route=route,
                      agent_path=agent_path, infeasible=infeasible,
                      start=start, goal=goal)
        # 근처 infeasible 노드 경고 라벨
        for ix, iy in infeas_xy:
            if abs(ix - ccx) < half_w and abs(iy - ccy) < half_w * 1.2:
                ax.annotate("좌회전 금지 — 통행 불가", (ix, iy),
                            textcoords="offset points", xytext=(0, 16),
                            ha="center", fontsize=9, fontweight="bold",
                            color="#e0202a")
        ax.set_title(cap, fontsize=11, fontweight="bold", color="#22242a", pad=8)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        frames.append(buf.getvalue())
        buf.close()
        if fi % 40 == 0:
            print(f"  [camera] frame {fi}/{total}", flush=True)

    plt.close(fig)
    imgs = (Image.open(io.BytesIO(b)) for b in frames)
    first = next(imgs)
    first.save(out_path, save_all=True, append_images=imgs,
               duration=int(1000 / fps), loop=0, optimize=False)
    print(f"  [camera] saved → {out_path}  ({total} frames)")


# ── 정적 산출물 ───────────────────────────────────────────────────────────────
def figure_fullmap(gm: GangnamMap, route, agent_path, infeasible,
                   start, goal, out_path: str):
    fig = plt.figure(figsize=(13, 13), dpi=200)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.92])
    xs = [p[0] for p in gm.xy.values()]
    ys = [p[1] for p in gm.xy.values()]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 200
    render_window(ax, gm, cx, cy, half, route=route, agent_path=agent_path,
                  infeasible=infeasible, start=start, goal=goal,
                  show_deadend=True,
                  title="강남구 도로망 1995노드 — Dijkstra 최단경로와 통행 불가 지점")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def figure_zoom_insets(gm: GangnamMap, route, agent_path, infeasible,
                       start, goal, out_path: str):
    """전체도 + infeasible 지점 줌 인셋 — 픽셀 열화 없는 확대 비교."""
    n = min(3, len(infeasible))
    fig = plt.figure(figsize=(15, 5.4), dpi=200)
    for k in range(n):
        ax = fig.add_subplot(1, n, k + 1)
        cx, cy = gm.xy[infeasible[k]]
        render_window(ax, gm, cx, cy, 130, route=route, agent_path=agent_path,
                      infeasible=infeasible, start=start, goal=goal,
                      title=f"통행 불가 지점 #{k+1}  (노드 {infeasible[k]})")
    fig.suptitle("좌회전 금지 노드 줌인 — 고해상도라 어느 배율에서도 선명",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def figure_pixel_compare(gm: GangnamMap, route, start, goal, out_path: str):
    """기존 저해상도(72dpi 전체 렌더 후 확대) vs 본 방법론(영역별 고해상도)."""
    fig = plt.figure(figsize=(14, 7), dpi=150)
    cx, cy = gm.xy[route[len(route) // 2]]

    # (좌) 기존 방식 — 전체를 저해상도로 렌더한 뒤 일부만 잘라 확대
    ax1 = fig.add_subplot(1, 2, 1)
    tmp = plt.figure(figsize=(4, 4), dpi=72)
    tax = tmp.add_axes([0, 0, 1, 1])
    xs = [p[0] for p in gm.xy.values()]; ys = [p[1] for p in gm.xy.values()]
    fcx, fcy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
    fhw = max(max(xs)-min(xs), max(ys)-min(ys))/2 + 200
    render_window(tax, gm, fcx, fcy, fhw, route=route, start=start, goal=goal)
    buf = io.BytesIO(); tmp.savefig(buf, format="png", dpi=72); plt.close(tmp)
    buf.seek(0)
    if _PIL:
        im = Image.open(buf)
        W, H = im.size
        # 중앙부 확대 crop
        fx = (cx - fcx) / (2 * fhw) + 0.5
        fy = 0.5 - (cy - fcy) / (2 * fhw)
        cw = W // 9
        box = (int(fx * W - cw), int(fy * H - cw), int(fx * W + cw), int(fy * H + cw))
        crop = im.crop(box).resize((460, 460), Image.NEAREST)
        ax1.imshow(crop)
    ax1.axis("off")
    ax1.set_title("기존: 전체 72dpi 렌더 후 확대 → 픽셀 열화",
                  fontsize=11, fontweight="bold", color="#c0392b")

    # (우) 본 방법론 — 같은 영역을 직접 고해상도 렌더
    ax2 = fig.add_subplot(1, 2, 2)
    render_window(ax2, gm, cx, cy, 220, route=route, start=start, goal=goal)
    ax2.set_title("본 방법론: 해당 영역을 직접 고해상도 렌더 → 선명",
                  fontsize=11, fontweight="bold", color="#0f7d3f")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def main(out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gm = GangnamMap(str(ROOT / "data" / "gangnam_clean_topology.json"))
    A = json.load(open(out / "_analysis.json", encoding="utf-8"))
    route      = A["dijkstra_path"]
    agent_path = A.get("agent_derail_path")
    infeasible = [n for n, _ in A.get("fail_nodes", [])]
    start, goal = gm.meta["start_node"], gm.meta["goal_nodes"][0]

    print("[1/4] 전체도")
    figure_fullmap(gm, route, agent_path, infeasible, start, goal,
                   str(out / "gn_fullmap.png"))
    print("[2/4] 줌 인셋")
    figure_zoom_insets(gm, route, agent_path, infeasible, start, goal,
                       str(out / "gn_zoom_infeasible.png"))
    print("[3/4] 픽셀 열화 비교")
    figure_pixel_compare(gm, route, start, goal,
                         str(out / "gn_pixel_compare.png"))
    print("[4/4] 카메라 자동이동 GIF")
    camera_gif(gm, route, str(out / "gn_camera_pan.gif"),
               agent_path=agent_path, infeasible=infeasible,
               start=start, goal=goal)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         str(ROOT / "output" / "22_1356_GN_RLMetod"))
