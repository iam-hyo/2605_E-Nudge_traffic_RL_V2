"""
fuel_calculate.py
-----------------
VT-Macro 기반 주행 연료 계산 + 공회전(Idle) 연료 계산.

주행 구간: FC(mL/s) = K0 + K1*v + K2*v^2 + K3*(a^2/v)
공회전    : IFC(mL/s) * t_wait

단독 실행 시 간단한 예시 출력.
"""

from __future__ import annotations
import math


# ── VT-Macro 계수 (일반 승용차 기준) ──────────────────────────────────────────
VT_K0 = 0.09     # 기저 연료율 (mL/s)
VT_K1 = 0.004    # 속도 1차 항
VT_K2 = 0.0002   # 속도 2차 항
VT_K3 = 0.03     # 가속도 항
IFC   = 0.50     # 공회전 연료율 (mL/s)
V_MIN_MS = 0.5   # VT-Macro 적용 최소 속도 (m/s) — 이하는 idle 처리


def fc_rate(v_ms: float, a_ms2: float) -> float:
    """
    순간 연료 소비율 (mL/s).
    v_ms  : 속도 (m/s)
    a_ms2 : 가속도 (m/s²), 음수 가능
    """
    if v_ms < V_MIN_MS:
        return IFC
    return VT_K0 + VT_K1 * v_ms + VT_K2 * v_ms ** 2 + VT_K3 * (a_ms2 ** 2) / v_ms


def fuel_segment(v_start: float, v_end: float, dist_m: float,
                 accel: float = 2.5) -> float:
    """
    단일 구간 연료 (mL).
    v_start, v_end : m/s
    dist_m         : 구간 거리 (m)
    accel          : 가속도 절댓값 (m/s²)

    등가속도 가정 → 구간 평균 속도·가속도로 적분.
    """
    if dist_m <= 0:
        return 0.0

    v_mean = (v_start + v_end) / 2.0
    if v_mean < V_MIN_MS:
        t_seg = dist_m / max(v_mean, 0.01)
        return IFC * t_seg

    a_eff  = accel if v_end >= v_start else -accel
    t_seg  = dist_m / v_mean
    rate   = fc_rate(v_mean, a_eff)
    return rate * t_seg


def fuel_idle(t_wait_sec: float) -> float:
    """공회전 연료 (mL). 신호 대기 구간 전용."""
    return IFC * max(0.0, t_wait_sec)


class SpeedProfile:
    """
    링크 주행 속도 프로파일 계산기.
    진입속도 → 목표 순항속도 → 진출 목표속도 로 구성된 3구간 프로파일 생성.

    가감속: 고정 2.5 m/s²
    링크 길이 부족 시 삼각형 프로파일(등속 구간 없음)로 자동 처리.
    """

    ACCEL = 2.5   # m/s²

    def __init__(self, v_cruise: float, v_entry: float, v_exit: float,
                 link_len: float):
        """
        v_cruise  : 순항 속도 (m/s)
        v_entry   : 진입 속도 (m/s) — 이전 링크 진출 속도
        v_exit    : 진출 목표 속도 (m/s) — 회전 시 감속 목표
        link_len  : 링크 길이 (m)
        """
        self.v_cruise  = v_cruise
        self.v_entry   = v_entry
        self.v_exit    = v_exit
        self.link_len  = link_len
        self.a         = self.ACCEL
        self.segments  = self._compute_segments()

    def _dist_to_reach(self, v_from: float, v_to: float) -> float:
        """등가속도로 v_from → v_to 에 필요한 거리 (m)."""
        return abs(v_to ** 2 - v_from ** 2) / (2 * self.a)

    def _compute_segments(self) -> list[tuple[float, float, float, float]]:
        """
        [(v_start, v_end, dist_m, accel), ...]
        """
        d_accel = self._dist_to_reach(self.v_entry,   self.v_cruise)
        d_decel = self._dist_to_reach(self.v_cruise,  self.v_exit)

        if d_accel + d_decel > self.link_len:
            # 삼각형 프로파일 — 달성 가능한 최고 속도 재계산
            v_peak = math.sqrt(
                (self.a * self.link_len
                 + 0.5 * self.v_entry ** 2
                 + 0.5 * self.v_exit ** 2)
            )
            v_peak  = max(v_peak, max(self.v_entry, self.v_exit))
            d_accel = self._dist_to_reach(self.v_entry, v_peak)
            d_decel = max(0.0, self.link_len - d_accel)
            return [
                (self.v_entry,  v_peak,       d_accel, +self.a),
                (v_peak,        self.v_exit,  d_decel, -self.a),
            ]

        d_cruise = self.link_len - d_accel - d_decel
        segs = []
        if d_accel > 0:
            segs.append((self.v_entry,   self.v_cruise, d_accel,  +self.a))
        if d_cruise > 0:
            segs.append((self.v_cruise,  self.v_cruise, d_cruise,  0.0))
        if d_decel > 0:
            segs.append((self.v_cruise,  self.v_exit,   d_decel,  -self.a))
        return segs or [(self.v_entry, self.v_exit, self.link_len, 0.0)]

    def total_fuel(self) -> float:
        """링크 주행 총 연료 (mL)."""
        return sum(fuel_segment(vs, ve, d, abs(a))
                   for vs, ve, d, a in self.segments)

    def total_time(self) -> float:
        """링크 주행 총 시간 (초)."""
        total = 0.0
        for vs, ve, d, _ in self.segments:
            v_mean = (vs + ve) / 2.0
            total += d / max(v_mean, 0.01)
        return total


if __name__ == "__main__":
    # 간단 예시: 40 km/h 순항, 직진, 400m 링크
    v = 40 / 3.6
    profile = SpeedProfile(v_cruise=v, v_entry=v * 0.6, v_exit=v * 0.5,
                           link_len=400)
    print(f"주행 연료: {profile.total_fuel():.2f} mL")
    print(f"주행 시간: {profile.total_time():.1f} s")
    print(f"공회전 20s: {fuel_idle(20):.2f} mL")
