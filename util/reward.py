"""
reward.py
---------
보상 계산기 — 이론적 minimum viable reward 구조.

R = -α · fuel_mL                       ← 연료 패널티 (매 스텝)
  + arrival_bonus · 𝟙_goal              ← 도착 보너스 (1회)
  - penalty_timeout · 𝟙_timeout         ← 학습 종료 신호 (선택)
  - penalty_dead · 𝟙_dead               ← 학습 종료 신호 (선택)

설계 원칙 (Sutton, Reward Hypothesis):
  목표가 "최소 연료로 도착"이면 -fuel + goal_bonus 만으로 충분 (sufficient).
  시간/대기시간/재방문 등은 proxy reward → 진짜 목표를 왜곡할 위험.

  - 대기시간: fuel_idle 이 이미 fuel_mL 에 포함되어 있어 별도 패널티 불필요
  - 총 시간 : 연료 자체가 시간×rate 의 적분이라 부분적으로 시간 정보 포함
  - 재방문 : 연료 추가가 자연 패널티

  Distance shaping 은 _train_common.py 에서 potential-based 형태로 적용 가능.
  Ng et al. (1999) 이론적 invariance 보장 — 단, 가중치가 너무 크면 fuel 신호를
  압도해 최적 정책을 사실상 왜곡하므로 0~소수로 유지 권장.

단위:
  fuel_mL : mL 단위 (VT-Micro 출력 L/s → environment.py에서 × 1000 환산)
"""

from __future__ import annotations


class RewardCalculator:
    def __init__(
        self,
        alpha:           float = 1.0,    # 연료 패널티 계수 (mL 기준)
        arrival_bonus:   float = 500.0,  # 도착 보너스
        penalty_timeout: float = 0.0,    # 기본 0 — 도착 실패 시 보너스 미수령이 자연 패널티
        penalty_dead:    float = 0.0,
    ):
        self.alpha           = alpha
        self.arrival_bonus   = arrival_bonus
        self.penalty_timeout = penalty_timeout
        self.penalty_dead    = penalty_dead

    # ── 매 스텝 호출 ──────────────────────────────────────────────────────────
    def step_reward(self, fuel_ml: float) -> float:
        """주행 + 공회전 연료 합산 패널티."""
        return -self.alpha * fuel_ml

    # ── 에피소드 종료 시 호출 ─────────────────────────────────────────────────
    def terminal_reward(
        self,
        reached_goal: bool,
        is_timeout:   bool,
        is_dead:      bool,
    ) -> float:
        if reached_goal:
            return self.arrival_bonus
        if is_dead:
            return -self.penalty_dead
        if is_timeout:
            return -self.penalty_timeout
        return 0.0

    def total(
        self,
        fuel_ml:      float,
        reached_goal: bool = False,
        is_timeout:   bool = False,
        is_dead:      bool = False,
    ) -> float:
        return (self.step_reward(fuel_ml)
                + self.terminal_reward(reached_goal, is_timeout, is_dead))
