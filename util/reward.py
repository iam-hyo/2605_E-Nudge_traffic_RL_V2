"""
reward.py
---------
보상 계산기.

R = -α * (FC_VTmacro + IFC * t_wait)   ← 연료 패널티 (매 스텝)
  + arrival_bonus * 𝟙_goal              ← 도착 보너스 (1회)
  - penalty_timeout * 𝟙_timeout         ← 타임아웃 (1회)
  - penalty_dead * 𝟙_dead               ← 막다른 길 (1회)

시간 패널티 없음 — 목표: 최소 연료 경로 탐색
"""

from __future__ import annotations


class RewardCalculator:
    def __init__(
        self,
        alpha:            float = 1.0,    # 연료 패널티 계수
        arrival_bonus:    float = 500.0,  # 도착 보너스 (고정)
        penalty_timeout:  float = 30.0,
        penalty_dead:     float = 20.0,
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
