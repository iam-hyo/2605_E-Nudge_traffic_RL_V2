"""
simulation.py – 교통 RL 고주사율 시뮬레이션 시각화

사용법:
  python simulation.py --models shortest_dijkstra rl_base --route long_01 --time_slot peak
  python simulation.py --models all --route short_01
  python simulation.py --models rl_signal --route long_01 --speed 2   # 2배속
  python simulation.py --models rl_signal rl_signal_attention --interval 25   # 40fps
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path

import matplotlib
from matplotlib import font_manager as _fm


def _pick_korean_font() -> str | None:
    """플랫폼별로 사용 가능한 한글 폰트를 골라 반환 (없으면 None).
    Noto Sans CJK 는 pan-CJK 폰트라 JP face 도 한글 글리프를 동일하게 포함한다."""
    available = {f.name for f in _fm.fontManager.ttflist}
    for cand in ("Malgun Gothic",                       # Windows
                 "AppleGothic", "Apple SD Gothic Neo",  # macOS
                 "NanumGothic", "NanumBarunGothic",      # Linux (nanum)
                 "Noto Sans CJK KR", "Noto Sans KR",     # Linux (noto, KR face)
                 "Noto Sans CJK JP"):                    # Linux (noto, JP face — 한글 포함)
        if cand in available:
            return cand
    return None


_KOREAN_FONT = _pick_korean_font()
matplotlib.rcParams["font.family"] = (
    [_KOREAN_FONT, "DejaVu Sans"] if _KOREAN_FONT else ["DejaVu Sans"]
)
matplotlib.rcParams["axes.unicode_minus"] = False
if _KOREAN_FONT is None:
    print("  [경고] 한글 폰트를 찾지 못했습니다 — 텍스트의 한글이 깨질 수 있습니다.",
          file=sys.stderr)
# --gif_only 플래그 조기 감지: pyplot import 전에 백엔드 결정
_GIF_ONLY_MODE = "--gif_only" in sys.argv
matplotlib.use("Agg" if _GIF_ONLY_MODE else "TkAgg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import io
from datetime import datetime
try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from util.environment import RoadNetworkEnv
from util.agent import DQNAgent
from util.dijkstra_models import ShortestDijkstra, StaticFuelDijkstra
from util.viz_scale import viz_params

# ── 애니메이션 파라미터 ────────────────────────────────────────────────────────
# --speed N 으로 FRAMES_PER_LINK 를 나누어 N배속 달성
BASE_FRAMES_PER_LINK = 25   # 1배속 기준 링크 당 프레임 수 (≈30fps × 0.83s/link)
MAX_WAIT_FRAMES      = 60   # 신호 대기 최대 프레임 수 (1배속 기준)

# ── 테마 ──────────────────────────────────────────────────────────────────────
BG_FIGURE = "#f4f6fb"
BG_CARD   = "#ffffff"
BG_HDR    = "#f0f2f8"
BG_MAP    = "#eef0f5"
TEXT_DARK = "#22242a"
TEXT_MID  = "#555770"
TEXT_LITE = "#8890aa"
GRID_CLR  = "#dde0ea"

MODEL_META: dict[str, dict] = {
    "shortest_dijkstra":    {"no": "①", "color": "#2176e8"},
    "static_fuel_dijkstra": {"no": "②", "color": "#e87a21"},
    "rl_base":              {"no": "③", "color": "#27a85e"},
    "rl_signal":            {"no": "④", "color": "#d43535"},
    "rl_signal_attention":  {"no": "⑤", "color": "#8e4fcf"},
}
ALL_MODELS = list(MODEL_META.keys())

TIMESLOT_LABEL   = {"off_peak": "07:00 한산", "peak": "08:00 병목"}
ROUTE_TYPE_LABEL = {
    "cross_main": "메인 경로", "cross_aux1": "보조 경로 ①",
    "cross_aux2": "보조 경로 ②", "cross_aux3": "보조 경로 ③",
}

# 신호 페이즈 색 (yellow → red 처리)
SIGNAL_COLORS = {
    "green":     "#16c45e",
    "red":       "#e03535",
    "yellow":    "#e03535",
    "left_turn": "#1e90ff",
}
NO_SIG_COLOR = "#c0c4d6"


# ── 유틸 ─────────────────────────────────────────────────────────────────────
def _signal_phase_at(node: dict, abs_sec: float) -> str:
    """해당 시각의 신호 페이즈 타입 반환 (no_signal / green / left_turn / red / yellow)."""
    sig = node.get("signal")
    if sig is None:
        return "no_signal"
    local_t = (abs_sec + sig["offset"]) % sig["cycle_length"]
    elapsed = 0.0
    for ph in sig["phases"]:
        if elapsed <= local_t < elapsed + ph["duration"]:
            return ph["type"]
        elapsed += ph["duration"]
    return "no_signal"


def _signal_color(node: dict, abs_sec: float) -> str:
    ph = _signal_phase_at(node, abs_sec)
    return SIGNAL_COLORS.get(ph, NO_SIG_COLOR)


def _node_phase_state(node: dict, abs_sec: float):
    """
    노드의 현재 phase 상태.
    반환: (phase_type, remain_ratio, color)  / 무신호면 None
      remain_ratio: 현재 phase 중 남은 비율 (0~1)
    """
    sig = node.get("signal")
    if sig is None:
        return None
    cycle  = sig["cycle_length"]
    offset = sig.get("offset", 0)
    local_t = (abs_sec + offset) % cycle
    elapsed = 0.0
    for ph in sig["phases"]:
        if elapsed <= local_t < elapsed + ph["duration"]:
            remain = (elapsed + ph["duration"] - local_t) / ph["duration"]
            ph_type = ph["type"]
            color   = SIGNAL_COLORS.get(ph_type, NO_SIG_COLOR)
            return ph_type, remain, color
        elapsed += ph["duration"]
    return None


# 신호 대기/좌회전 규칙은 모두 util/environment.py 의 step() 안에서 통합 처리.
# 시뮬레이션은 env.step 의 결과(wait_time, abs_depart, movement, fuel_idle)를 그대로 사용.


def _load_model(name: str, cfg: dict, env: RoadNetworkEnv):
    model_dir = Path(cfg["output"]["model_dir"])
    if name == "shortest_dijkstra":
        return ShortestDijkstra(env)
    if name == "static_fuel_dijkstra":
        return StaticFuelDijkstra(env)
    mode_map = {"rl_base": "base", "rl_signal": "signal",
                "rl_signal_attention": "attention"}
    pth_map  = {"rl_base": "model_rl_base.pth",
                "rl_signal": "model_rl_signal.pth",
                "rl_signal_attention": "model_rl_signal_attention.pth"}
    tc = cfg["train"]
    agent = DQNAgent(
        mode=mode_map[name], gamma=tc["gamma"],
        epsilon_min=tc["epsilon_min"], epsilon=tc["epsilon_min"],
        epsilon_decay=1.0, lr=tc["lr"],
    )
    pth = model_dir / pth_map[name]
    if pth.exists():
        agent.load(str(pth))
        print(f"  [로드] {pth.name}")
    else:
        print(f"  [경고] {pth.name} 없음 — 미학습 모델")
    return agent


# ── 에이전트 상태 머신 ────────────────────────────────────────────────────────
class _AgentState:
    """
    Global wall-clock 기반 에이전트.

    핵심: 모든 agent 가 같은 sim_time 을 공유한다. 매 frame Simulator.sim_time 이
    sim_dt 만큼 진행하면, 각 agent.advance_to(sim_time) 가 자기 link 안에서의
    위치·모드를 갱신한다. 빠른 agent는 같은 sim_time 안에서 더 많은 link 를
    통과한다 (env.step 호출이 더 빠르게 누적).

    link 안 구성: [waiting (t_wait)] → [traveling (t_travel)]
    위치 보간: traveling 구간에서 ease-in-out (3t²−2t³) — 시작/끝 부드러운 가·감속.

    신호 준수: env.step() 의 movement-aware 대기를 그대로 신뢰. 모든 agent 가
    같은 sim_time 진행이므로 신호 phase 도 모두 동일하게 평가됨.
    """
    def __init__(self, name: str, env: RoadNetworkEnv, model,
                 start: str, goal: str, start_hour: float):
        self.name  = name
        self.env   = env
        self.model = model

        self.done    = False
        self.reached = False

        # 누적 지표 (cum_time = 자기 시뮬 시간, sim_time 과 다를 수 있음:
        # link 진행 중에는 sim_time 이 더 앞서 있을 수 있음. 대시보드에는
        # global sim_time 을 표시하므로 cum_time 은 디버그용)
        self.step_count = 0
        self.cum_fuel   = 0.0
        self.cum_wait   = 0.0
        self.cum_dist   = 0.0

        # 영구 경로
        self._path_nodes: list[str] = [start]

        # 현재 link 진행 상태
        self._link_sim_start = 0.0          # 이 link 가 sim_time 어디서 시작했는가
        self._link_t_wait    = 0.0          # 출발 신호 대기 (s)
        self._link_t_travel  = 0.0          # 링크 주행 시간 (s)
        self._link_total     = 0.0          # = t_wait + t_travel
        self._link_v_cruise  = 0.0          # 표시용 (km/h)
        self._link_fuel      = 0.0          # 이 link 의 총 연료 (mL)
        self._link_dist      = 0.0
        self._pos_from = [0.0, 0.0]
        self._pos_to   = [0.0, 0.0]

        # 시뮬 현재 위치·모드 (advance_to 가 갱신)
        self.pos       = [0.0, 0.0]
        self._mode     = "idle"             # waiting / traveling / done
        self.speed_kmh = 0.0                # 현재 순간 속도 (운동학)

        self._sim_done = False

        self.state = env.reset(start_node=start, goal_nodes=[goal],
                               start_hour=start_hour)
        self.pos     = list(env.nodes[start]["pos"])
        self._pos_to = list(self.pos)
        # 첫 link 준비 (sim_time=0 기준)
        self._prepare_next(sim_time=0.0)

    # ── 시각화 보조 ──────────────────────────────────────────────────────────
    def focus_node_id(self) -> str | None:
        """wedge 표시 대상 노드 (waiting=현재, traveling=다음 도착)."""
        if self._mode == "waiting":
            if len(self._path_nodes) >= 2:
                return self._path_nodes[-2]
            return self._path_nodes[-1] if self._path_nodes else None
        if self._mode == "traveling":
            return self._path_nodes[-1] if self._path_nodes else None
        return None

    def current_abs_t(self) -> float:
        """env wall-clock (신호 phase 평가용). env.current_time 이 이미 sim_time
        과 동기되어 진행되므로 link 안 미세 보간만 추가."""
        base = self.env.start_time_sec + self.env.current_time
        # env.current_time 은 step 완료 시점 기준 → 진행 중 link 의 sub-frame 분 보정
        # 단순화: sim_time 진행분을 그대로 더해줘도 OK 하지만 env.current_time 이
        # 이미 t_wait+t_travel 을 포함하므로 base 가 link 끝 시각. waiting/traveling
        # 중에는 base 에서 (link_total - elapsed) 만큼 뺀 것이 현재 wall-clock.
        return base - max(0.0, self._link_total - (self._sim_now - self._link_sim_start))

    # ── 다음 link 준비 ───────────────────────────────────────────────────────
    def _prepare_next(self, sim_time: float):
        self._sim_now = sim_time
        if self._sim_done:
            self._mode = "done"
            self.done  = True
            self.pos   = list(self._pos_to)
            return

        valid = self.env.get_valid_actions()
        if not valid:
            self._sim_done = True
            self._mode     = "done"
            self.done      = True
            return

        from_node      = self.env.current_node
        self._pos_from = list(self.env.nodes[from_node]["pos"])

        action = self.model.act(self.state, valid)
        self.state, reward, done_flag, info = self.env.step(action)

        to_node      = self.env.current_node
        self._pos_to = list(self.env.nodes[to_node]["pos"])

        wt        = info.get("wait_time",   0.0)
        t_travel  = info.get("travel_time", 0.0)
        fuel_tot  = info.get("fuel_total",  0.0)
        movement  = info.get("movement",    "straight")

        # link 기록
        self._link_sim_start = sim_time
        self._link_t_wait    = wt
        self._link_t_travel  = max(t_travel, 0.01)
        self._link_total     = wt + self._link_t_travel
        self._link_v_cruise  = info.get("speed_kmh", 0.0)
        self._link_fuel      = fuel_tot
        self._link_dist      = info.get("distance", 0.0)

        # 누적 지표
        self.cum_fuel  += fuel_tot
        self.cum_wait  += wt
        self.cum_dist  += self._link_dist
        self.step_count += 1
        self.reached    = info.get("reached_goal", False)

        self._path_nodes.append(to_node)

        self._mode = "waiting" if wt > 0 else "traveling"
        self.pos   = list(self._pos_from)
        self.speed_kmh = 0.0 if wt > 0 else self._link_v_cruise

        if wt > 0:
            phase = _signal_phase_at(self.env.nodes[from_node],
                                     self.env.start_time_sec + self.env.current_time - wt - t_travel)
            print(f"  [{self.name}] 스텝{self.step_count}: "
                  f"{from_node}({phase}) 에서 {movement} 위해 {wt:.1f}s 대기")

        if done_flag:
            self._sim_done = True

    # ── Global sim_time 으로 자기 상태 갱신 ──────────────────────────────────
    def advance_to(self, sim_time: float):
        """
        sim_time 시점에서의 위치·모드 갱신.
        sim_time 이 현재 link 의 끝(_link_sim_start + _link_total)을 넘으면
        다음 link 로 자동 진입 (한 frame 내 여러 link 통과 가능).
        """
        if self._mode == "done":
            return
        self._sim_now = sim_time

        # 현재 link 가 이미 끝났으면 다음 link 들로 진행 (반복)
        while not self._sim_done and sim_time >= self._link_sim_start + self._link_total - 1e-9:
            # 이 link 의 누적 시간 = self._link_sim_start + self._link_total
            # 다음 link 는 그 시점부터 시작
            next_start = self._link_sim_start + self._link_total
            self._prepare_next(next_start)
            if self._mode == "done":
                return

        # 마지막 link (목표 도착 link) 의 주행까지 모두 끝났으면 종료 처리.
        # _sim_done 이 set 되면 위 while 이 _prepare_next 를 더 호출하지 않으므로
        # 여기서 직접 done 으로 전환해 줘야 한다.
        if self._sim_done and sim_time >= self._link_sim_start + self._link_total - 1e-9:
            self._mode = "done"
            self.done  = True
            self.pos   = list(self._pos_to)
            return

        local_t = sim_time - self._link_sim_start  # 0 ~ _link_total
        if local_t < self._link_t_wait:
            # 출발 대기
            self._mode = "waiting"
            self.pos   = list(self._pos_from)
            self.speed_kmh = 0.0
        else:
            # 주행
            self._mode = "traveling"
            t_in_travel = local_t - self._link_t_wait
            r = min(1.0, max(0.0, t_in_travel / max(self._link_t_travel, 1e-6)))
            ease = r * r * (3.0 - 2.0 * r)   # smoothstep ease-in-out
            self.pos = [
                self._pos_from[0] + (self._pos_to[0] - self._pos_from[0]) * ease,
                self._pos_from[1] + (self._pos_to[1] - self._pos_from[1]) * ease,
            ]
            # 순간 속도 ≈ ease 미분 × 평균속도. smoothstep' = 6r(1-r), max=1.5 at r=0.5
            ease_prime = 6.0 * r * (1.0 - r)
            self.speed_kmh = self._link_v_cruise * ease_prime / 1.5
            # 종착 근접 → 도착 마지막 미세 정착
            if r >= 1.0 - 1e-6:
                self.pos = list(self._pos_to)
                self.speed_kmh = self._link_v_cruise

    # ── 현재 표시용 연료 (실시간 누적: 이전 link 까지 + 현재 link 진행 비율) ──
    def fuel_realtime(self) -> float:
        if self._mode == "done":
            return self.cum_fuel
        # 이미 cum_fuel 에 현재 link 의 전체 연료가 더해져 있음 → 진행 비율로 빼주기
        if self._link_total <= 0:
            return self.cum_fuel
        local_t = max(0.0, self._sim_now - self._link_sim_start)
        progress = min(1.0, local_t / self._link_total)
        return (self.cum_fuel - self._link_fuel) + self._link_fuel * progress

    def wait_realtime(self) -> float:
        if self._mode == "done":
            return self.cum_wait
        local_t = max(0.0, self._sim_now - self._link_sim_start)
        wait_progress = min(local_t, self._link_t_wait)
        return (self.cum_wait - self._link_t_wait) + wait_progress


# ── 시뮬레이터 ────────────────────────────────────────────────────────────────
class Simulator:
    def __init__(self, cfg_path: str, model_names: list[str],
                 route_name: str, time_slot: str, speed_mult: int = 1,
                 gif_path: str | None = None):
        self.cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        routes   = {r["name"]: r for r in self.cfg["experiments"]["routes"]}
        tslots   = {t["label"]: t for t in self.cfg["experiments"]["time_slots"]}
        route    = routes[route_name]
        tslot    = tslots[time_slot]

        self.start      = route["start"]
        self.goal       = route["goal"]
        self.start_hour = tslot["start_hour"]
        self.route_name = route_name
        self.time_slot  = time_slot
        self._fc        = 0
        self.gif_path   = gif_path
        self.gif_frames: list = []
        self.interval_ms: int = 33

        # speed_mult: 1배속=BASE_FRAMES_PER_LINK, N배속=//N
        self.frames_per_link = max(5, BASE_FRAMES_PER_LINK // speed_mult)

        # 속도 노이즈는 ±20% 가우시안 그대로 (학습/평가와 동일). 시간 동기화는
        # global sim_time + advance_to() 로 보장 — 모든 agent 가 같은 sim_time
        # 진행에 따라 자기 위치 갱신.
        self.agents: list[_AgentState] = []
        for name in model_names:
            env = RoadNetworkEnv(
                self.cfg["data"]["topology"], self.cfg["data"]["speed"],
                reward_cfg=self.cfg["reward"], use_signal=(name != "rl_base"),
            )
            model = _load_model(name, self.cfg, env)
            self.agents.append(
                _AgentState(name, env, model, self.start, self.goal,
                            self.start_hour)
            )

        self.ref_env = self.agents[0].env

        # Global wall-clock — 1배속에서 frame 당 sim_dt = 0.5초.
        # speed_mult N → sim_dt = 0.5 * N. (예: link 30s = 60/N frames at 30fps)
        self.sim_time = 0.0
        self.sim_dt   = 0.5 * speed_mult

        self._build_figure()

    # ── 그림 뼈대 ────────────────────────────────────────────────────────────
    def _build_figure(self):
        n     = len(self.agents)
        fig_h = max(9.5, n * 2.0 + 3.8)

        self.fig = plt.figure(figsize=(18, fig_h), layout="constrained")
        self.fig.patch.set_facecolor(BG_FIGURE)

        route_lbl = ROUTE_TYPE_LABEL.get(self.route_name, self.route_name)
        time_lbl  = TIMESLOT_LABEL.get(self.time_slot, self.time_slot)
        self.fig.suptitle(
            f"최소 연료 경로 탐색 시뮬레이션  ·  "
            f"{route_lbl} ({self.start}→{self.goal})  ·  {time_lbl}",
            fontsize=12, fontweight="bold", color=TEXT_DARK,
        )

        left_sf, right_sf = self.fig.subfigures(
            1, 2, width_ratios=[1, 2.3], wspace=0.01,
        )
        left_sf.set_facecolor(BG_FIGURE)
        right_sf.set_facecolor(BG_FIGURE)

        hr      = [1.0] * n + [0.85, 0.85]
        left_gs = left_sf.add_gridspec(n + 2, 1, height_ratios=hr, hspace=0.14)
        self.info_axes = [left_sf.add_subplot(left_gs[i]) for i in range(n)]
        self.fuel_ax   = left_sf.add_subplot(left_gs[n])
        self.wait_ax   = left_sf.add_subplot(left_gs[n + 1])

        self.map_ax = right_sf.add_subplot(1, 1, 1)

        self._init_info_cards()
        self._init_graphs()
        self._init_map()

    # ── 정보 카드 초기화 ──────────────────────────────────────────────────────
    def _init_info_cards(self):
        self._card_texts: list[dict] = []

        for ax, ag in zip(self.info_axes, self.agents):
            meta  = MODEL_META[ag.name]
            color = meta["color"]

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_facecolor(BG_CARD)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for side, sp in ax.spines.items():
                sp.set_visible(True)
                sp.set_edgecolor(color if side == "left" else GRID_CLR)
                sp.set_linewidth(3.5 if side == "left" else 0.8)

            ax.add_patch(mpatches.FancyBboxPatch(
                (0, 0.78), 1, 0.22, boxstyle="square,pad=0",
                transform=ax.transAxes,
                facecolor=BG_HDR, edgecolor="none", clip_on=False,
            ))

            ax.text(0.07, 0.89, meta["no"], transform=ax.transAxes,
                    va="center", ha="left", fontsize=12,
                    fontweight="bold", color=color)
            ax.text(0.22, 0.89, ag.name, transform=ax.transAxes,
                    va="center", ha="left", fontsize=8.5,
                    fontweight="bold", color=TEXT_DARK)

            st = ax.text(0.95, 0.89, "주행 중 ▶", transform=ax.transAxes,
                         va="center", ha="right", fontsize=8,
                         fontweight="bold", color="#d07010")

            ax.plot([0, 1], [0.78, 0.78], transform=ax.transAxes,
                    color=GRID_CLR, lw=0.8, clip_on=False)

            for lbl, y in [("연료", 0.59), ("시간", 0.42),
                           ("속도", 0.25), ("스텝", 0.08)]:
                ax.text(0.08, y, lbl, transform=ax.transAxes,
                        va="center", ha="left", fontsize=8.5, color=TEXT_LITE)

            def _vt(y, init="—"):
                return ax.text(0.95, y, init, transform=ax.transAxes,
                               va="center", ha="right", fontsize=9,
                               fontweight="bold", color=TEXT_DARK,
                               fontfamily="monospace")

            self._card_texts.append({
                "status": st,
                "fuel":   _vt(0.59, "0.0 mL"),
                "time":   _vt(0.42, "0.0 s"),
                "speed":  _vt(0.25, "0.0 km/h"),
                "step":   _vt(0.08, "0"),
            })

    def _update_info_cards(self):
        """매 frame 호출. 시간=global sim_time(모든 모델 동일), 나머지=agent별 실시간."""
        for texts, ag in zip(self._card_texts, self.agents):
            if ag.reached:
                texts["status"].set_text("도달 완료 ✓")
                texts["status"].set_color("#16a858")
            elif ag.done:
                texts["status"].set_text("타임아웃 ✗")
                texts["status"].set_color("#c83030")
            elif ag._mode == "waiting":
                texts["status"].set_text("신호 대기 ●")
                texts["status"].set_color("#e03535")
            else:
                texts["status"].set_text("주행 중 ▶")
                texts["status"].set_color("#d07010")

            # 연료/속도는 link 진행 비율로 실시간 보간 → 매 frame 매끄럽게 갱신
            texts["fuel"].set_text(f"{ag.fuel_realtime():.1f} mL")
            texts["time"].set_text(f"{self.sim_time:.1f} s")
            texts["speed"].set_text(f"{ag.speed_kmh:.1f} km/h")
            texts["step"].set_text(str(ag.step_count))

    # ── 실시간 그래프 초기화 ────────────────────────────────────────────────
    def _init_graphs(self):
        # 그래프용 시계열 — sim_time 축, 매 frame 한 점씩 추가
        self._series_t:    list[float]         = []
        self._series_fuel: dict[str, list[float]] = {}
        self._series_wait: dict[str, list[float]] = {}
        for ag in self.agents:
            self._series_fuel[ag.name] = []
            self._series_wait[ag.name] = []

        for ax, title in [(self.fuel_ax, "누적 연료 (mL)"),
                          (self.wait_ax, "누적 대기시간 (s)")]:
            ax.set_facecolor(BG_CARD)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID_CLR)
                sp.set_linewidth(0.8)
            ax.tick_params(colors=TEXT_LITE, labelsize=7)
            ax.set_title(title, fontsize=8, color=TEXT_MID, pad=3)
            ax.set_xlabel("sim time (s)", fontsize=7, color=TEXT_LITE)
            ax.grid(True, alpha=0.25, color=GRID_CLR)

        self.fuel_lines: dict[str, plt.Line2D] = {}
        self.wait_lines: dict[str, plt.Line2D] = {}
        for ag in self.agents:
            color = MODEL_META[ag.name]["color"]
            no    = MODEL_META[ag.name]["no"]
            fl, = self.fuel_ax.plot([], [], color=color, lw=1.6, label=no)
            # rl_base는 경로 탐색시 신호 미사용이지만 운행 중 신호는 준수
            wait_label = no + (" (탐색無신호)" if ag.name == "rl_base" else "")
            wl, = self.wait_ax.plot([], [], color=color, lw=1.6, label=wait_label)
            self.fuel_lines[ag.name] = fl
            self.wait_lines[ag.name] = wl

        self.fuel_ax.legend(fontsize=7, facecolor=BG_CARD, labelcolor=TEXT_DARK,
                            edgecolor=GRID_CLR, loc="upper left", borderpad=0.5)
        self.wait_ax.legend(fontsize=6.5, facecolor=BG_CARD, labelcolor=TEXT_DARK,
                            edgecolor=GRID_CLR, loc="upper left", borderpad=0.5)

    def _update_graphs(self):
        """매 frame 호출. x축 = sim_time(초), y축 = 실시간 누적치.
        autoscale 은 5 frame 마다 (성능 최적화)."""
        self._series_t.append(self.sim_time)
        for ag in self.agents:
            self._series_fuel[ag.name].append(ag.fuel_realtime())
            self._series_wait[ag.name].append(ag.wait_realtime())
        for ag in self.agents:
            self.fuel_lines[ag.name].set_data(
                self._series_t, self._series_fuel[ag.name])
            self.wait_lines[ag.name].set_data(
                self._series_t, self._series_wait[ag.name])
        if self._fc % 5 == 0:
            for ax in (self.fuel_ax, self.wait_ax):
                ax.relim()
                ax.autoscale_view()

    # ── 지도 초기화 ──────────────────────────────────────────────────────────
    def _init_map(self):
        ax  = self.map_ax
        env = self.ref_env
        ax.set_facecolor(BG_MAP)
        ax.set_aspect("equal")
        ax.axis("off")

        # 자동 스케일링 (36 노드 ~ 1000+ 노드까지 호환)
        self._vp = viz_params(env.N, env.map_diag)
        vp = self._vp

        # 링크 (고정 회색)
        for lk in env.links.values():
            p1 = env.nodes[str(lk["end1"])]["pos"]
            p2 = env.nodes[str(lk["end2"])]["pos"]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color="#d0d3e0", lw=vp["link_lw"], zorder=1,
                    solid_capstyle="round")

        # ── 노드 타입 분류 (무신호 / 신호 / 좌회전) ──────────────────────────
        self._node_ids        = sorted(env.nodes.keys())
        self._node_no_sig:    list[str] = []
        self._node_signal:    list[str] = []
        self._node_left_turn: list[str] = []

        for nid in self._node_ids:
            sig = env.nodes[nid].get("signal")
            if sig is None:
                self._node_no_sig.append(nid)
            elif any(p["type"] == "left_turn" for p in sig["phases"]):
                self._node_left_turn.append(nid)
            else:
                self._node_signal.append(nid)

        def _pos_xy(ids):
            return ([env.nodes[n]["pos"][0] for n in ids],
                    [env.nodes[n]["pos"][1] for n in ids])

        xs_ns, ys_ns = _pos_xy(self._node_no_sig)
        xs_sg, ys_sg = _pos_xy(self._node_signal)
        xs_lt, ys_lt = _pos_xy(self._node_left_turn)

        # 무신호: 작고 회색
        if xs_ns:
            ax.scatter(xs_ns, ys_ns, c=NO_SIG_COLOR,
                       s=vp["node_size_nosig"], zorder=3, edgecolors="none")

        # 신호 (비좌회전): 중간 크기
        self.scatter_signal = None
        if xs_sg:
            self.scatter_signal = ax.scatter(
                xs_sg, ys_sg, c=[NO_SIG_COLOR] * len(xs_sg),
                s=vp["node_size_signal"], zorder=3,
                edgecolors="white", linewidths=max(0.6, vp["density_scale"] * 1.2),
            )

        # 좌회전 신호: 크고 파란 테두리
        self.scatter_left = None
        if xs_lt:
            self.scatter_left = ax.scatter(
                xs_lt, ys_lt, c=[NO_SIG_COLOR] * len(xs_lt),
                s=vp["node_size_lt"], zorder=3,
                edgecolors="#1565c0",
                linewidths=max(1.2, vp["density_scale"] * 2.5),
            )

        # 좌회전 ← 표시 (대규모 노드에서는 생략 — 시각적 잡음 ↑)
        if env.N <= 200:
            for nid in self._node_left_turn:
                pos = env.nodes[nid]["pos"]
                ax.text(pos[0] + vp["lt_offset"], pos[1] + vp["lt_offset"], "←",
                        fontsize=vp["lt_font"], color="#1565c0",
                        ha="left", va="bottom", zorder=7, fontweight="bold")

        # 출발 / 도착 마커
        sp = env.nodes[self.start]["pos"]
        gp = env.nodes[self.goal]["pos"]
        ax.scatter(*sp, c="#16c45e", s=vp["star_ms"], marker="*", zorder=9)
        ax.scatter(*gp, c="#f5a623", s=vp["star_ms"], marker="*", zorder=9)

        # ── 영구 경로 (흰색 테두리 + 모델 컬러) ─────────────────────────────
        self.path_bg: list[plt.Line2D] = []
        self.path_fg: list[plt.Line2D] = []
        for ag in self.agents:
            color = MODEL_META[ag.name]["color"]
            bg, = ax.plot([], [], "-", color="white",
                          lw=vp["path_bg_lw"], zorder=5,
                          solid_capstyle="round", solid_joinstyle="round")
            fg, = ax.plot([], [], "-", color=color,
                          lw=vp["path_lw"], zorder=6,
                          solid_capstyle="round", solid_joinstyle="round",
                          alpha=0.85)
            self.path_bg.append(bg)
            self.path_fg.append(fg)

        # 에이전트 마커 + 신호 phase wedge
        # wedge 는 12시 시작·시계방향 줄어드는 호로 phase 잔여시간 표시.
        # waiting 모드: 현재 노드의 신호 phase wedge
        # traveling 모드: 다음 노드(action)의 신호 phase wedge — 다음 노드 무신호면 숨김
        # 5 agents 동시 → radius offset 으로 동심원 형태 (겹침 방지)
        self.agent_dots:    list[plt.Line2D]     = []
        self.signal_wedges: list[mpatches.Wedge] = []
        # 신호 wedge 반지름·간격은 맵 좌표 단위. 절대 하한을 두면 좌표계가
        # 작은 토폴로지(강남구 위경도, map_diag≈0.13)에서 맵보다 커져 화면을
        # 가린다 → 항상 map_diag 에 비례시킨다 (격자 토폴로지 영향 없음).
        ring_base = env.map_diag / 60.0
        self._ring_base   = ring_base
        self._ring_offset = env.map_diag / 200.0  # agent 간 동심원 간격
        wedge_lw = max(2.0, vp["density_scale"] * 3.5)
        for i, ag in enumerate(self.agents):
            color = MODEL_META[ag.name]["color"]
            dot, = ax.plot([], [], "o", color=color,
                           ms=vp["agent_ms"], zorder=8,
                           markeredgecolor="white",
                           markeredgewidth=max(0.8, vp["density_scale"] * 1.5))
            # 도넛 형 호 (width=두께). 색은 phase 에 따라 set_facecolor 로 매 frame 갱신
            wedge = mpatches.Wedge(
                center=(0, 0),
                r=ring_base + i * self._ring_offset,
                theta1=90, theta2=90,    # 시작은 0 도 (보이지 않음)
                width=ring_base * 0.18,
                facecolor="#e03535", edgecolor="white",
                linewidth=max(0.5, vp["density_scale"]),
                alpha=0, zorder=9,
            )
            ax.add_patch(wedge)
            self.agent_dots.append(dot)
            self.signal_wedges.append(wedge)

        # 신호 범례 (좌하단)
        sig_handles = [
            mlines.Line2D([], [], color="none", marker="o",
                          markerfacecolor="#16c45e", ms=9, label="직진 녹색"),
            mlines.Line2D([], [], color="none", marker="o",
                          markerfacecolor="#e03535", ms=9, label="정지 적색"),
            mlines.Line2D([], [], color="none", marker="o",
                          markerfacecolor="#1e90ff", ms=11,
                          markeredgecolor="#1565c0", markeredgewidth=2,
                          label="← 좌회전"),
            mlines.Line2D([], [], color="none", marker="o",
                          markerfacecolor=NO_SIG_COLOR, ms=7, label="무신호"),
        ]
        sig_leg = ax.legend(
            handles=sig_handles, loc="lower left", fontsize=7.5,
            facecolor=BG_CARD, labelcolor=TEXT_DARK, edgecolor=GRID_CLR,
            framealpha=0.95, borderpad=0.7, title="신호", title_fontsize=8,
        )
        ax.add_artist(sig_leg)

        # 모델 범례 (우상단)
        model_handles = [
            mlines.Line2D(
                [], [], color=MODEL_META[ag.name]["color"],
                marker="o", ms=8, lw=2,
                label=f"{MODEL_META[ag.name]['no']} {ag.name}",
            )
            for ag in self.agents
        ]
        ax.legend(
            handles=model_handles, loc="upper right", fontsize=8,
            facecolor=BG_CARD, labelcolor=TEXT_DARK, edgecolor=GRID_CLR,
            framealpha=0.95, borderpad=0.8, title="모델", title_fontsize=8,
        )
        # 명시적 축 범위 — 노드 좌표 경계 기준. autoscale 에 의존하면 (0,0)
        # 에 초기화된 신호 wedge patch 가 datalim 을 끌어당겨, 원점에서 먼
        # 좌표계(강남구 위경도)에서 도로망이 구석으로 밀린다.
        mx = max(env.map_w, 1e-9) * 0.06
        my = max(env.map_h, 1e-9) * 0.06
        ax.set_xlim(env.map_x_min - mx, env.map_x_max + mx)
        ax.set_ylim(env.map_y_min - my, env.map_y_max + my)

    # ── 지도 업데이트 (프레임마다) ────────────────────────────────────────────
    def _update_map(self):
        env = self.ref_env

        # 모든 agent 가 같은 sim_time 으로 진행 → 모두 같은 wall-clock 공유
        sim_t = self.sim_time
        for ag in self.agents:
            ag.advance_to(sim_t)

        # 전역 신호 색 평가 시각: env.start_time_sec + sim_time
        abs_t = env.start_time_sec + sim_t

        # 신호 노드 색상 업데이트 — phase 는 수십 초 단위 변경이므로 매 3 frame
        # 마다만 redraw (60fps × 1초 = 60 frame, 3 frame = 50ms 간격으로 충분)
        if self._fc % 3 == 0:
            if self.scatter_signal is not None:
                self.scatter_signal.set_color(
                    [_signal_color(env.nodes[n], abs_t) for n in self._node_signal]
                )
            if self.scatter_left is not None:
                self.scatter_left.set_color(
                    [_signal_color(env.nodes[n], abs_t) for n in self._node_left_turn]
                )

        for i, ag in enumerate(self.agents):
            pos = ag.pos

            # 에이전트 마커
            self.agent_dots[i].set_data([pos[0]], [pos[1]])

            # 영구 경로: 방문 노드 + 현재 서브프레임 위치
            valid_path = [n for n in ag._path_nodes if n in env.nodes]
            pxs = [env.nodes[n]["pos"][0] for n in valid_path] + [pos[0]]
            pys = [env.nodes[n]["pos"][1] for n in valid_path] + [pos[1]]
            self.path_bg[i].set_data(pxs, pys)
            self.path_fg[i].set_data(pxs, pys)

            # 신호 phase wedge — 모든 모델이 같은 sim_time 의 같은 phase 평가
            wedge = self.signal_wedges[i]
            focus_id = ag.focus_node_id()
            phase_state = (_node_phase_state(env.nodes[focus_id], abs_t)
                           if focus_id is not None else None)
            if phase_state is None:
                wedge.set_alpha(0)
            else:
                ph_type, remain_r, color = phase_state
                target_pos = (env.nodes[focus_id]["pos"]
                              if ag._mode == "traveling"
                              else pos)
                wedge.set_center(tuple(target_pos))
                wedge.set_theta1(90.0 - 360.0 * remain_r)
                wedge.set_theta2(90.0)
                wedge.set_facecolor(color)
                wedge.set_alpha(0.85 if ag._mode == "waiting" else 0.72)

    # ── 애니메이션 프레임 ─────────────────────────────────────────────────────
    def _animate(self, frame: int):
        self._fc = frame
        # Global wall-clock 진행: 매 frame 마다 sim_dt 초 더하기
        # (1배속 sim_dt=0.5s; speed_mult N 일 때 0.5×N)
        self.sim_time += self.sim_dt
        self._update_map()
        self._update_info_cards()
        self._update_graphs()

        # GIF 프레임 캡처 (PIL 설치된 경우에만).
        # 디코딩된 이미지 대신 PNG 바이트로 보관 → 메모리 사용량 수십 배 절감.
        if self.gif_path and _PIL_AVAILABLE:
            buf = io.BytesIO()
            self.fig.savefig(buf, format="png", dpi=72)
            self.gif_frames.append(buf.getvalue())
            buf.close()

        if all(ag.done for ag in self.agents):
            if self.anim is not None:
                self.anim.event_source.stop()
            txt = self.fig.texts[0]
            if "완료" not in txt.get_text():
                txt.set_text(txt.get_text() + "   .   시뮬레이션 완료")
                txt.set_color("#16a858")
                if self.gif_path and self.gif_frames:
                    self._save_gif()
            if self.anim is not None:
                self.fig.canvas.draw_idle()

        return []

    def _save_gif(self):
        gif_dir = Path(self.gif_path).parent
        gif_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  [GIF] 저장 중... ({len(self.gif_frames)} 프레임)")
        # PNG 바이트를 하나씩 lazily 디코딩 (generator) → 저장 시점에도
        # 한 프레임만 메모리에 올라간다.
        frames = (_PILImage.open(io.BytesIO(b)) for b in self.gif_frames)
        first  = next(frames)
        first.save(
            self.gif_path, save_all=True,
            append_images=frames,
            duration=self.interval_ms, loop=0, optimize=False,
        )
        print(f"  [GIF] 저장 완료: {self.gif_path}")

    # ── 헤드리스 GIF 렌더링 ──────────────────────────────────────────────────
    def _run_gif_only(self):
        """Agg 백엔드로 창 없이 전체 시뮬레이션을 렌더링해 GIF 저장."""
        print(f"  [GIF] 헤드리스 렌더링 시작...", flush=True)
        frame_idx = 0
        while not all(ag.done for ag in self.agents):
            self._fc = frame_idx
            self.sim_time += self.sim_dt
            self._update_map()
            self._update_info_cards()
            self._update_graphs()

            if _PIL_AVAILABLE:
                buf = io.BytesIO()
                self.fig.savefig(buf, format="png", dpi=72)
                self.gif_frames.append(buf.getvalue())
                buf.close()

            frame_idx += 1
            if frame_idx % 200 == 0:
                print(f"  [GIF] 프레임 {frame_idx}...", flush=True)
            if frame_idx > 20000:
                print("  [GIF] 안전 한도 도달 — 렌더링 종료", flush=True)
                break

        if self.gif_frames:
            self._save_gif()
        plt.close(self.fig)

    # ── 실행 ─────────────────────────────────────────────────────────────────
    def run(self, interval_ms: int = 33, gif_only: bool = False):
        self.interval_ms = interval_ms
        if gif_only:
            self.anim = None
            self._run_gif_only()
        else:
            self.anim = animation.FuncAnimation(
                self.fig, self._animate,
                interval=interval_ms, blit=False, cache_frame_data=False,
            )
            plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "최소 연료 경로 탐색 - 고주사율 시뮬레이션\n"
            "  ① shortest_dijkstra    링크 길이 최단\n"
            "  ② static_fuel_dijkstra 예상 연료 최적\n"
            "  ③ rl_base              신호 미사용 DQN (운행 중 신호 준수)\n"
            "  ④ rl_signal            신호 포함 DQN\n"
            "  ⑤ rl_signal_attention  Attention DQN\n"
            "\n"
            "신호 준수 규칙:\n"
            "  녹색(●)  → 직진·우회전 가능, 즉시 통과\n"
            "  파랑(←)  → 좌회전·우회전 가능, 즉시 통과\n"
            "  적색(●)  → 전체 정지 (우회전 포함), 녹색까지 대기\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # 경로 선택지는 config.yaml 의 routes 에서 동적으로 읽는다
    # (토폴로지 전환 시 cross_* / gangnam_* 등 자동 반영).
    try:
        _cfg_routes = yaml.safe_load(open("config/config.yaml", encoding="utf-8"))
        _route_choices = [r["name"] for r in _cfg_routes["experiments"]["routes"]]
    except Exception:
        _route_choices = ["cross_main"]
    parser.add_argument(
        "--models", nargs="+", default=["shortest_dijkstra", "rl_base"],
        metavar="MODEL",
        help=f"모델 선택 (공백 구분 / all=전체)\n선택지: {ALL_MODELS + ['all']}",
    )
    parser.add_argument("--route", default=_route_choices[0],
                        help="실험 경로 — --config 의 experiments.routes 에 정의된 경로 이름")
    parser.add_argument("--time_slot", default="off_peak",
                        choices=["off_peak", "peak"],
                        help="off_peak=07:00 한산, peak=08:00 병목")
    parser.add_argument("--config",   default="config/config.yaml")
    parser.add_argument(
        "--interval", type=int, default=33,
        help="프레임 간격 ms (기본 33 = 30fps / 빠르게: 16 ≈ 60fps / 느리게: 50 = 20fps)",
    )
    parser.add_argument(
        "--speed", type=int, default=1, choices=[1, 2, 3, 4],
        help=(
            "애니메이션 배속 (기본 1배속)\n"
            "  1 = 1배속 (25 frames/link)\n"
            "  2 = 2배속 (12 frames/link)\n"
            "  3 = 3배속 (8 frames/link)\n"
            "  4 = 4배속 (6 frames/link)\n"
        ),
    )
    parser.add_argument(
        "--save_gif", action="store_true",
        help="시뮬레이션 결과를 GIF로 저장 (output/gif/). Pillow 필요: pip install Pillow",
    )
    parser.add_argument(
        "--gif_only", action="store_true",
        help="창 없이 GIF만 저장 (헤드리스 렌더링, --save_gif 자동 포함)",
    )
    args = parser.parse_args()
    if args.gif_only:
        args.save_gif = True

    model_names = ALL_MODELS if "all" in args.models else args.models
    invalid = [m for m in model_names if m not in ALL_MODELS]
    if invalid:
        parser.error(f"알 수 없는 모델: {invalid}")

    fps = 1000 // args.interval
    fpl = max(5, BASE_FRAMES_PER_LINK // args.speed)
    link_ms = fpl * args.interval

    # GIF 경로 구성
    gif_path = None
    if args.save_gif:
        if not _PIL_AVAILABLE:
            print("[경고] Pillow 미설치 — pip install Pillow.  GIF 저장 건너뜀.")
        else:
            _abbr = {
                "rl_signal_attention": "rla", "rl_signal": "rls",
                "rl_base": "rlb", "shortest_dijkstra": "sdj",
                "static_fuel_dijkstra": "fdj",
            }
            ms = "_".join(_abbr.get(m, m) for m in model_names)
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_name = f"{now_str}_{ms}_{args.route}_{args.time_slot}_x{args.speed}.gif"
            gif_path = str(ROOT / "output" / "gif" / gif_name)

    print(f"\n{'='*55}")
    print("  최소 연료 경로 탐색 - 시뮬레이션")
    print(f"{'='*55}")
    for name in model_names:
        print(f"  {MODEL_META[name]['no']}  {name}")
    print(f"\n  경로   : {ROUTE_TYPE_LABEL.get(args.route, args.route)}")
    print(f"  시간대 : {TIMESLOT_LABEL.get(args.time_slot, args.time_slot)}")
    print(f"  FPS    : {fps}  |  배속: {args.speed}x  |  링크당: {link_ms}ms")
    if gif_path:
        print(f"  GIF    : {gif_path}")
    print(f"\n  신호 준수: 적색=정지(우회전 포함), 녹색=통과, 파랑=좌회전통과")
    print(f"  ③ rl_base: 경로 탐색 시 신호 미사용, 운행 중 교통법규 준수")
    print(f"{'='*55}\n")

    Simulator(args.config, model_names, args.route, args.time_slot,
              speed_mult=args.speed, gif_path=gif_path).run(args.interval,
                                                            gif_only=args.gif_only)


if __name__ == "__main__":
    main()
