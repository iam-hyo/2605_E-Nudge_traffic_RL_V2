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
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
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
    "short_01": "단거리 ①", "short_02": "단거리 ②",
    "long_01":  "장거리 ①", "long_02":  "장거리 ②",
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


def _calc_signal_wait(node: dict, arrive_sec: float) -> float:
    """
    교통 법규 준수용 대기 시간.
    모델 use_signal 설정과 무관하게 실제 신호 위반을 방지.

    적색/황색 → 다음 녹색(또는 좌회전) 페이즈까지 대기.
    녹색/좌회전/무신호 → 즉시 통과 (0.0).
    """
    sig = node.get("signal")
    if sig is None:
        return 0.0
    cycle   = sig["cycle_length"]
    offset  = sig["offset"]
    local_t = (arrive_sec + offset) % cycle
    elapsed = 0.0
    for ph in sig["phases"]:
        if elapsed <= local_t < elapsed + ph["duration"]:
            if ph["type"] in ("red", "yellow"):
                return elapsed + ph["duration"] - local_t
            return 0.0   # green / left_turn → 통과
        elapsed += ph["duration"]
    return 0.0


def _calc_left_turn_wait(from_node: dict, prev_pos: list, from_pos: list,
                         to_pos: list, current_sec: float) -> float:
    """
    좌회전 신호 대기 시간.

    스크린 좌표(y 하방) 기준: 외적 cross < 0 → 운전자 관점 좌회전.
    from_node에 left_turn 페이즈가 있는 경우에만 제한 적용.
    현재 페이즈가 left_turn이면 즉시 0 반환, 아니면 다음 left_turn까지 대기.
    """
    sig = from_node.get("signal")
    if sig is None:
        return 0.0
    if not any(p["type"] == "left_turn" for p in sig["phases"]):
        return 0.0  # left_turn 페이즈 없는 노드 — 제한 없음

    dx1 = from_pos[0] - prev_pos[0]
    dy1 = from_pos[1] - prev_pos[1]
    if dx1 == 0 and dy1 == 0:
        return 0.0

    dx2 = to_pos[0] - from_pos[0]
    dy2 = to_pos[1] - from_pos[1]
    if dx1 * dy2 - dy1 * dx2 >= 0:
        return 0.0  # 직진 또는 우회전

    # 좌회전 — 현재 페이즈부터 순회하여 left_turn 페이즈까지 대기 시간 계산
    local_t = (current_sec + sig["offset"]) % sig["cycle_length"]
    elapsed = 0.0
    for i, ph in enumerate(sig["phases"]):
        if elapsed <= local_t < elapsed + ph["duration"]:
            if ph["type"] == "left_turn":
                return 0.0
            wait = elapsed + ph["duration"] - local_t
            n = len(sig["phases"])
            for j in range(1, n):
                nxt = sig["phases"][(i + j) % n]
                if nxt["type"] == "left_turn":
                    return wait
                wait += nxt["duration"]
            return 0.0
        elapsed += ph["duration"]
    return 0.0


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
        action_size=env.action_size, node_list=sorted(env.nodes.keys()),
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


# ── 에이전트 상태 + 애니메이션 머신 ──────────────────────────────────────────
class _AgentState:
    """
    한 에이전트의 시뮬레이션 + 애니메이션 상태를 관리.

    모드 전이:
      idle → (prepare_next) → traveling → waiting → traveling → ...
                                                              → done

    신호 준수:
      use_signal=False(rl_base)라도 시뮬레이션 상에서는 교통 법규 적용.
      _calc_signal_wait() 가 env._calc_wait() 와 독립적으로 신호 대기 계산.
    """
    def __init__(self, name: str, env: RoadNetworkEnv, model,
                 start: str, goal: str, start_hour: float,
                 frames_per_link: int):
        self.name            = name
        self.env             = env
        self.model           = model
        self.frames_per_link = frames_per_link

        self.done    = False
        self.reached = False

        self.step_count  = 0
        self.speed_kmh   = 0.0
        self.cum_fuel    = 0.0
        self.cum_time    = 0.0
        self.cum_wait    = 0.0
        self.cum_dist    = 0.0
        self.last_reward = 0.0

        # 실시간 그래프용 히스토리
        self._step_hist: list[int]   = [0]
        self._fuel_hist: list[float] = [0.0]
        self._wait_hist: list[float] = [0.0]

        # 영구 경로 기록 (방문 노드 순서)
        self._path_nodes: list[str] = [start]

        # 신호 애니메이션용: 링크 진입 시각 + 통과 소요 시간
        self._abs_t_depart:     float = float(env.start_time_sec)
        self._link_travel_time: float = 30.0

        # 애니메이션 내부 상태
        self._mode        = "idle"
        self._t           = 0.0
        self._pos_from    = [0.0, 0.0]
        self._pos_to      = [0.0, 0.0]
        self._wait_frames = 0
        self._wait_idx    = 0
        self._sim_done    = False

        self.state = env.reset(start_node=start, goal_nodes=[goal],
                               start_hour=start_hour)
        self._pos_to = list(env.nodes[start]["pos"])
        self._prepare_next()

    # ── 다음 스텝 준비 ────────────────────────────────────────────────────────
    def _prepare_next(self):
        if self._sim_done:
            self._mode = "done"
            self.done  = True
            return

        valid = self.env.get_valid_actions()
        if not valid:
            self._sim_done = True
            self._mode     = "done"
            self.done      = True
            return

        from_node      = self.env.current_node
        self._pos_from = list(self.env.nodes[from_node]["pos"])

        # 링크 진입 시각 기록 (신호 색상 보간용)
        self._abs_t_depart = self.env.start_time_sec + self.env.current_time

        action = self.model.act(self.state, valid)

        # 좌회전 신호 준수: 출발 노드에서 left_turn 페이즈 대기
        prev_node = self.env.previous_node
        if prev_node != from_node:
            prev_pos    = list(self.env.nodes[prev_node]["pos"])
            to_pos_tmp  = list(self.env.nodes[action]["pos"])
            current_sec = self.env.start_time_sec + self.env.current_time
            lt_wait = _calc_left_turn_wait(
                self.env.nodes[from_node], prev_pos, self._pos_from,
                to_pos_tmp, current_sec,
            )
            if lt_wait > 0:
                self.env.current_time += lt_wait
                self.cum_wait         += lt_wait
                self.cum_time         += lt_wait
                self._abs_t_depart     = self.env.start_time_sec + self.env.current_time
                print(f"  [{self.name}] {from_node} 좌회전 대기 {lt_wait:.1f}s")

        self.state, reward, done_flag, info = self.env.step(action)

        to_node      = self.env.current_node
        self._pos_to = list(self.env.nodes[to_node]["pos"])

        # ── 신호 준수 (모든 모델에 교통 법규 적용) ──────────────────────────
        # env.step()이 반환한 wait_time: use_signal=False(rl_base)이면 항상 0
        info_wt = info.get("wait_time", 0.0)

        # env.current_time = T_prev + t_travel + info_wt
        # arrive_sec = env.start_time_sec + T_prev + t_travel
        #            = env.start_time_sec + env.current_time - info_wt
        arrive_sec = self.env.start_time_sec + self.env.current_time - info_wt
        sim_wt     = _calc_signal_wait(self.env.nodes[to_node], arrive_sec)

        # rl_base 등 use_signal=False 모델의 env 내부 시각도 보정
        extra_wait = sim_wt - info_wt
        if extra_wait > 0:
            self.env.current_time += extra_wait

        wt = sim_wt

        # ── 지표 갱신 ────────────────────────────────────────────────────────
        self.speed_kmh   = info.get("speed_kmh",  0.0)
        self.cum_fuel   += info.get("fuel_total",  0.0)
        self.cum_time   += info.get("travel_time", 0.0) + wt
        self.cum_wait   += wt
        self.cum_dist   += info.get("distance",    0.0)
        self.last_reward = reward
        self.step_count += 1
        self.reached     = info.get("reached_goal", False)

        self._path_nodes.append(to_node)
        self._step_hist.append(self.step_count)
        self._fuel_hist.append(self.cum_fuel)
        self._wait_hist.append(self.cum_wait)

        # 링크 통과 시간 갱신 (신호 색상 보간용)
        self._link_travel_time = info.get("travel_time", 30.0)

        if wt > 0:
            phase = _signal_phase_at(self.env.nodes[to_node], arrive_sec)
            print(f"  [{self.name}] 스텝{self.step_count}: "
                  f"{to_node}({phase}) 에서 {wt:.1f}s 대기")

        # ── 애니메이션 파라미터 ───────────────────────────────────────────────
        self._t = 0.0
        # 대기 프레임: 실제 대기 시간에 비례하되 최소 10, 최대 60 프레임 확보
        self._wait_frames = (max(10, min(MAX_WAIT_FRAMES, int(wt * 0.6)))
                             if wt > 0 else 0)
        self._wait_idx = 0
        self._mode     = "traveling"

        if done_flag:
            self._sim_done = True

    # ── 프레임 1회 진행 → 현재 위치 반환 ─────────────────────────────────────
    def advance_frame(self) -> list[float]:
        if self._mode == "done":
            return list(self._pos_to)

        if self._mode == "traveling":
            self._t += 1.0 / self.frames_per_link

            if self._t >= 1.0:
                end_pos = list(self._pos_to)
                if self._wait_frames > 0:
                    self._mode     = "waiting"
                    self._wait_idx = 0
                else:
                    self._prepare_next()
                return end_pos

            t = self._t
            if self._wait_frames > 0:
                # 신호 정차 예정: ease-in-out (자연스러운 감속)
                ease = t * t * (3.0 - 2.0 * t)
            else:
                # 무신호 통과: linear (등속, 끝에서 멈추지 않음)
                ease = t

            return [
                self._pos_from[0] + (self._pos_to[0] - self._pos_from[0]) * ease,
                self._pos_from[1] + (self._pos_to[1] - self._pos_from[1]) * ease,
            ]

        elif self._mode == "waiting":
            cur_pos = list(self._pos_to)
            self._wait_idx += 1
            if self._wait_idx >= self._wait_frames:
                self._prepare_next()
            return cur_pos

        return list(self._pos_to)


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

        self.agents: list[_AgentState] = []
        for name in model_names:
            env = RoadNetworkEnv(
                self.cfg["data"]["topology"], self.cfg["data"]["speed"],
                reward_cfg=self.cfg["reward"], use_signal=(name != "rl_base"),
            )
            model = _load_model(name, self.cfg, env)
            self.agents.append(
                _AgentState(name, env, model, self.start, self.goal,
                            self.start_hour, self.frames_per_link)
            )

        self.ref_env = self.agents[0].env
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

            texts["fuel"].set_text(f"{ag.cum_fuel:.1f} mL")
            texts["time"].set_text(f"{ag.cum_time:.1f} s")
            texts["speed"].set_text(f"{ag.speed_kmh:.1f} km/h")
            texts["step"].set_text(str(ag.step_count))

    # ── 실시간 그래프 초기화 ────────────────────────────────────────────────
    def _init_graphs(self):
        for ax, title in [(self.fuel_ax, "누적 연료 (mL)"),
                          (self.wait_ax, "누적 대기시간 (s)")]:
            ax.set_facecolor(BG_CARD)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID_CLR)
                sp.set_linewidth(0.8)
            ax.tick_params(colors=TEXT_LITE, labelsize=7)
            ax.set_title(title, fontsize=8, color=TEXT_MID, pad=3)
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
        has_data = False
        for ag in self.agents:
            if len(ag._step_hist) > 1:
                self.fuel_lines[ag.name].set_data(ag._step_hist, ag._fuel_hist)
                self.wait_lines[ag.name].set_data(ag._step_hist, ag._wait_hist)
                has_data = True
        if has_data:
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

        # 링크 (고정 회색)
        for lk in env.links.values():
            p1 = env.nodes[str(lk["end1"])]["pos"]
            p2 = env.nodes[str(lk["end2"])]["pos"]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color="#d0d3e0", lw=2.0, zorder=1, solid_capstyle="round")

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
            ax.scatter(xs_ns, ys_ns, c=NO_SIG_COLOR, s=22, zorder=3,
                       edgecolors="none")

        # 신호 (비좌회전): 중간 크기
        self.scatter_signal = None
        if xs_sg:
            self.scatter_signal = ax.scatter(
                xs_sg, ys_sg, c=[NO_SIG_COLOR] * len(xs_sg),
                s=100, zorder=3, edgecolors="white", linewidths=1.2,
            )

        # 좌회전 신호: 크고 파란 테두리
        self.scatter_left = None
        if xs_lt:
            self.scatter_left = ax.scatter(
                xs_lt, ys_lt, c=[NO_SIG_COLOR] * len(xs_lt),
                s=160, zorder=3, edgecolors="#1565c0", linewidths=2.5,
            )

        # 좌회전 ← 표시 (노드 우상단 오프셋)
        for nid in self._node_left_turn:
            pos = env.nodes[nid]["pos"]
            ax.text(pos[0] + 28, pos[1] + 28, "←",
                    fontsize=7.5, color="#1565c0",
                    ha="left", va="bottom", zorder=7, fontweight="bold")

        # 출발 / 도착 마커
        sp = env.nodes[self.start]["pos"]
        gp = env.nodes[self.goal]["pos"]
        ax.scatter(*sp, c="#16c45e", s=360, marker="*", zorder=9)
        ax.scatter(*gp, c="#f5a623", s=360, marker="*", zorder=9)

        # ── 영구 경로 (흰색 테두리 + 모델 컬러) ─────────────────────────────
        self.path_bg: list[plt.Line2D] = []
        self.path_fg: list[plt.Line2D] = []
        for ag in self.agents:
            color = MODEL_META[ag.name]["color"]
            bg, = ax.plot([], [], "-", color="white", lw=5.5, zorder=5,
                          solid_capstyle="round", solid_joinstyle="round")
            fg, = ax.plot([], [], "-", color=color, lw=2.8, zorder=6,
                          solid_capstyle="round", solid_joinstyle="round",
                          alpha=0.85)
            self.path_bg.append(bg)
            self.path_fg.append(fg)

        # 에이전트 마커 + 신호 대기 링
        self.agent_dots:  list[plt.Line2D] = []
        self.wait_rings:  list[plt.Circle] = []
        for ag in self.agents:
            color = MODEL_META[ag.name]["color"]
            dot, = ax.plot([], [], "o", color=color, ms=14, zorder=8,
                           markeredgecolor="white", markeredgewidth=1.5)
            ring = plt.Circle((0, 0), 0, color="#e03535",
                              fill=False, lw=2.5, alpha=0, zorder=9)
            ax.add_patch(ring)
            self.agent_dots.append(dot)
            self.wait_rings.append(ring)

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
        ax.margins(0.06)

    # ── 지도 업데이트 (프레임마다) ────────────────────────────────────────────
    def _update_map(self):
        env = self.ref_env
        ag0 = self.agents[0]

        # 링크 통과 중이면 진행률(t)에 비례하여 시각 보간 → 신호 색상이 실시간으로 변함
        if ag0._mode == "traveling":
            abs_t = ag0._abs_t_depart + ag0._t * ag0._link_travel_time
        elif ag0._mode == "waiting":
            abs_t = ag0._abs_t_depart + ag0._link_travel_time
        else:
            abs_t = env.start_time_sec + ag0.cum_time

        # 신호 노드 색상 업데이트
        if self.scatter_signal is not None:
            self.scatter_signal.set_color(
                [_signal_color(env.nodes[n], abs_t) for n in self._node_signal]
            )
        if self.scatter_left is not None:
            self.scatter_left.set_color(
                [_signal_color(env.nodes[n], abs_t) for n in self._node_left_turn]
            )

        for i, ag in enumerate(self.agents):
            pos = ag.advance_frame()

            # 에이전트 마커
            self.agent_dots[i].set_data([pos[0]], [pos[1]])

            # 영구 경로: 방문 노드 + 현재 서브프레임 위치
            valid_path = [n for n in ag._path_nodes if n in env.nodes]
            pxs = [env.nodes[n]["pos"][0] for n in valid_path] + [pos[0]]
            pys = [env.nodes[n]["pos"][1] for n in valid_path] + [pos[1]]
            self.path_bg[i].set_data(pxs, pys)
            self.path_fg[i].set_data(pxs, pys)

            # 신호 대기 링 (pulse)
            ring = self.wait_rings[i]
            if ag._mode == "waiting":
                pulse = 0.7 + 0.3 * abs(math.sin(self._fc * 0.22))
                ring.set_center(tuple(pos))
                ring.set_radius(32 * pulse)
                ring.set_alpha(0.85)
            else:
                ring.set_alpha(0)

    # ── 애니메이션 프레임 ─────────────────────────────────────────────────────
    def _animate(self, frame: int):
        self._fc = frame
        self._update_map()
        self._update_info_cards()
        self._update_graphs()

        # GIF 프레임 캡처 (PIL 설치된 경우에만)
        if self.gif_path and _PIL_AVAILABLE:
            buf = io.BytesIO()
            self.fig.savefig(buf, format="png", dpi=72)
            buf.seek(0)
            self.gif_frames.append(_PILImage.open(buf).copy())
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
        self.gif_frames[0].save(
            self.gif_path, save_all=True,
            append_images=self.gif_frames[1:],
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
            self._update_map()
            self._update_info_cards()
            self._update_graphs()

            if _PIL_AVAILABLE:
                buf = io.BytesIO()
                self.fig.savefig(buf, format="png", dpi=72)
                buf.seek(0)
                self.gif_frames.append(_PILImage.open(buf).copy())
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
    parser.add_argument(
        "--models", nargs="+", default=["shortest_dijkstra", "rl_base"],
        metavar="MODEL",
        help=f"모델 선택 (공백 구분 / all=전체)\n선택지: {ALL_MODELS + ['all']}",
    )
    parser.add_argument("--route", default="short_01",
                        choices=["short_01", "short_02", "long_01", "long_02"],
                        help="short_01/02=단거리, long_01/02=장거리")
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
