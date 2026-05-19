"""
fuel_calculate.py
-----------------
VT-Micro 다항회귀 연료 모델 (Rakha-Ahn, NCHRP 데이터 calibrated).

  ln(F) = Σ_{i=0..3} Σ_{j=0..3} K_{ij} · a^i · s^j

  - 가속도 부호에 따라 별도 계수 셋 (가속 16개 / 감속 16개)
  - 연료식 입력 단위: 속도 km/h, 가속도 km/h/s
  - 내부 운동학 계산 단위: 속도 m/s, 가속도 m/s²
  - 시간 적분: dt = 0.1s 단위 수치 누적

이전 VT-Macro (FC = K0 + K1·v + K2·v² + K3·a²/v) 폐기 사유:
  1. 거시 평균 추정용 → step 단위 reward에 부적합
  2. K값 출처 불명 (단위·차종 미명시)
  3. 평균속도 1회 계산 → 비선형 항에 systematic bias
  4. a²/v 대칭형 → 회생제동/coasting 효율 학습 불가

API 호환성 유지: fc_rate, fuel_idle, fuel_segment, SpeedProfile
"""

from __future__ import annotations
import math


# ============================================================
# 1. Time-step setting
# ============================================================

DT = 0.1   # 수치 적분 시간 간격 (s)


# ============================================================
# 2. VT-Micro fuel model
# ============================================================

def vt_micro_fuel(s: float, a: float) -> float:
    """
    VT-Micro 다항회귀 연료식.

    매개변수
    --------
    s : 속도 (km/h)
    a : 가속도 (km/h/s)

    반환
    ----
    순간 연료 소비율 (모델 학습 단위 그대로, 통상 mL/s ~ L/s)
    """
    if a >= 0:
        ln_fuel = (
            (-7.73452 + 0.02799 * s - 0.0002228 * s**2 + 1.09e-6 * s**3)
            + a * (0.22946 + 0.0068 * s - 0.00004402 * s**2 + 4.80e-8 * s**3)
            + (a**2) * (-0.00561 - 0.00077221 * s + 7.90e-7 * s**2 + 3.27e-8 * s**3)
            + (a**3) * (9.77e-5 + 0.00000838 * s + 8.17e-7 * s**2 - 7.79e-9 * s**3)
        )
    else:
        ln_fuel = (
            (-7.73452 + 0.02804 * s - 0.00021988 * s**2 + 1.08e-6 * s**3)
            + a * (-0.01799 + 0.00772 * s - 0.00005219 * s**2 + 2.47e-7 * s**3)
            + (a**2) * (-0.00427 + 0.00083744 * s - 7.44e-6 * s**2 + 4.87e-8 * s**3)
            + (a**3) * (0.00018829 - 0.00003387 * s + 2.77e-7 * s**2 + 3.79e-10 * s**3)
        )
    return math.exp(ln_fuel)


# ============================================================
# 3. Fuel rate wrapper with unit conversion
# ============================================================

def fc_rate(v_ms: float, a_ms2: float) -> float:
    """
    순간 연료 소비율 (학습 단위).

    매개변수
    --------
    v_ms   : 속도 (m/s)
    a_ms2  : 가속도 (m/s²), 음수 가능 (감속)

    내부 단위 변환 (학습 시 사용된 단위로 맞춤):
      m/s   → km/h    (× 3.6)
      m/s²  → km/h/s  (× 3.6)
    """
    s_kmh = v_ms * 3.6
    a_kmhs = a_ms2 * 3.6
    return vt_micro_fuel(s_kmh, a_kmhs)


# ============================================================
# 4. Segment time calculation
# ============================================================

def segment_time(v_start: float, v_end: float, dist_m: float, accel: float) -> float:
    """
    세그먼트 소요 시간 (초).

    - 가속/감속 구간: Δv / |a|
    - 등속 구간    : dist / 평균속도
    """
    if dist_m <= 0:
        return 0.0
    if abs(accel) > 1e-12 and abs(v_end - v_start) > 1e-12:
        return abs(v_end - v_start) / abs(accel)
    v_mean = (v_start + v_end) / 2.0
    return dist_m / max(v_mean, 0.01)


# ============================================================
# 5. Segment fuel using time-step accumulation
# ============================================================

def fuel_segment(
    v_start: float,
    v_end:   float,
    dist_m:  float,
    accel:   float = 2.5,
    dt:      float = DT,
) -> float:
    """
    단일 주행 세그먼트 연료.

    dt 단위 수치 적분 — 비선형 연료식에 대해
    평균속도 1회 계산 방식의 systematic bias 제거.
    """
    if dist_m <= 0:
        return 0.0

    if v_end > v_start:
        a_eff = abs(accel)
    elif v_end < v_start:
        a_eff = -abs(accel)
    else:
        a_eff = 0.0

    t_total = segment_time(v_start, v_end, dist_m, a_eff)
    if t_total <= 0:
        return 0.0

    total_fuel = 0.0
    elapsed    = 0.0
    v_curr     = v_start

    while elapsed < t_total - 1e-12:
        h = min(dt, t_total - elapsed)
        if a_eff > 0:
            v_next = min(v_end, v_curr + a_eff * h)
        elif a_eff < 0:
            v_next = max(v_end, v_curr + a_eff * h)
        else:
            v_next = v_curr

        rate = fc_rate(v_next, a_eff)
        total_fuel += rate * h

        v_curr   = v_next
        elapsed += h

    return total_fuel


# ============================================================
# 6. Idle fuel
# ============================================================

def fuel_idle(t_wait_sec: float) -> float:
    """
    정차/공회전 연료.

    별도 IFC 상수 폐기 — VT-Micro 식의 v=0, a=0 값을
    학습 단위로 일관 적용해 모델 통일성 확보.
    """
    idle_rate = fc_rate(0.0, 0.0)
    return idle_rate * max(0.0, t_wait_sec)


# ============================================================
# 7. Speed profile
# ============================================================

class SpeedProfile:
    """
    링크 주행 속도 프로파일 계산기.

    진입속도 → 순항속도 → 진출 목표속도 의 3구간 등가속도 프로파일.
    링크 길이 부족 시 삼각형(등속 구간 제거) 프로파일로 자동 처리.

    고정 가감속도: 2.5 m/s²
    """

    ACCEL = 2.5   # m/s²

    def __init__(
        self,
        v_cruise: float,
        v_entry:  float,
        v_exit:   float,
        link_len: float,
    ):
        """
        v_cruise : 순항 속도 (m/s)
        v_entry  : 진입 속도 (m/s)
        v_exit   : 진출 목표 속도 (m/s)
        link_len : 링크 길이 (m)
        """
        self.v_cruise = v_cruise
        self.v_entry  = v_entry
        self.v_exit   = v_exit
        self.link_len = link_len
        self.a        = self.ACCEL
        self.segments = self._compute_segments()

    def _dist_to_reach(self, v_from: float, v_to: float) -> float:
        """등가속도로 v_from → v_to 도달에 필요한 거리."""
        return abs(v_to ** 2 - v_from ** 2) / (2 * self.a)

    def _compute_segments(self) -> list[tuple[float, float, float, float]]:
        """[(v_start, v_end, dist_m, accel_signed), ...]"""
        d_accel = self._dist_to_reach(self.v_entry,  self.v_cruise)
        d_decel = self._dist_to_reach(self.v_cruise, self.v_exit)

        if d_accel + d_decel > self.link_len:
            # 삼각형 프로파일 — 도달 가능 최고 속도 재계산
            v_peak = math.sqrt(
                self.a * self.link_len
                + 0.5 * self.v_entry ** 2
                + 0.5 * self.v_exit ** 2
            )
            v_peak  = max(v_peak, max(self.v_entry, self.v_exit))
            d_accel = self._dist_to_reach(self.v_entry, v_peak)
            d_decel = max(0.0, self.link_len - d_accel)
            return [
                (self.v_entry, v_peak,      d_accel, +self.a),
                (v_peak,       self.v_exit, d_decel, -self.a),
            ]

        d_cruise = self.link_len - d_accel - d_decel
        segs: list[tuple[float, float, float, float]] = []
        if d_accel > 0:
            segs.append((self.v_entry,  self.v_cruise, d_accel,  +self.a))
        if d_cruise > 0:
            segs.append((self.v_cruise, self.v_cruise, d_cruise,  0.0))
        if d_decel > 0:
            segs.append((self.v_cruise, self.v_exit,   d_decel,  -self.a))
        return segs or [(self.v_entry, self.v_exit, self.link_len, 0.0)]

    def total_fuel(self) -> float:
        """링크 주행 총 연료 (학습 단위)."""
        return sum(
            fuel_segment(vs, ve, d, abs(a), dt=DT)
            for vs, ve, d, a in self.segments
        )

    def total_time(self) -> float:
        """링크 주행 총 시간 (초)."""
        return sum(
            segment_time(vs, ve, d, a)
            for vs, ve, d, a in self.segments
        )


# ============================================================
# 8. Example run
# ============================================================

if __name__ == "__main__":
    # 40 km/h 순항, 400m 링크
    v = 40 / 3.6
    profile = SpeedProfile(
        v_cruise = v,
        v_entry  = v * 0.6,
        v_exit   = v * 0.5,
        link_len = 400,
    )
    print(f"주행 연료: {profile.total_fuel():.6f}")
    print(f"주행 시간: {profile.total_time():.1f} s")
    print(f"공회전 20s: {fuel_idle(20):.6f}")
    print(f"idle rate: {fc_rate(0.0, 0.0):.6f} per second")
