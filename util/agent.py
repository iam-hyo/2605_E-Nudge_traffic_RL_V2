"""
agent.py
--------
DQNAgent — Double DQN + Dueling + Experience Replay.

모든 RL 모델(base / signal / attention)에서 공통 사용.
모델 종류는 build_model(mode) 로 주입.

행동 공간 — 엣지-상대적 (edge-relative):
  모델 출력은 ACTION_DIM(=K_HOP1) 개의 슬롯 Q값. 슬롯 k 는 env.get_valid_actions()
  (방위순 정렬) 의 k 번째 엣지에 대응한다. 전역 노드 ID/인덱스를 쓰지 않으므로
  토폴로지가 달라도 동일 모델·동일 가중치가 그대로 동작 → 다중 토폴로지 학습 가능.

  · act()    : 슬롯 Q값 argmax → 해당 엣지의 노드 ID 반환 (외부 인터페이스는 노드 ID 유지)
  · remember(): 행동을 '슬롯 인덱스'로 저장, next state 의 유효 슬롯 개수도 저장
  · replay() : 타깃 텐서 shape (B, ACTION_DIM), 슬롯 인덱스에만 타깃 주입
"""

from __future__ import annotations

import copy
import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from util.model import build_model, ACTION_DIM


class DQNAgent:
    def __init__(
        self,
        mode:           str   = "signal",   # 'base' | 'signal' | 'attention'
        gamma:          float = 0.95,
        epsilon:        float = 1.0,
        epsilon_min:    float = 0.05,
        epsilon_decay:  float = 0.995,
        lr:             float = 1e-3,
        memory_size:    int   = 10_000,
        batch_size:     int   = 64,
        target_update:  int   = 15,        # 에피소드 단위
        device:         Optional[str] = None,
    ):
        self.action_dim    = ACTION_DIM

        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model        = build_model(mode).to(self.device)
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.optimizer    = optim.Adam(self.model.parameters(), lr=lr)
        self.memory: deque = deque(maxlen=memory_size)

        self._episode_count = 0

    # ── 행동 선택 ─────────────────────────────────────────────────────────────
    def act(self, state: np.ndarray, valid_actions: list[str]) -> str:
        """
        valid_actions : env.get_valid_actions() — 방위순 정렬, len ≤ ACTION_DIM.
                        슬롯 k ↔ valid_actions[k].
        반환          : 선택한 엣지의 다음 노드 ID (외부 인터페이스 호환).
        """
        if not valid_actions:
            raise ValueError("valid_actions is empty")

        n = len(valid_actions)
        if random.random() <= self.epsilon:
            return random.choice(valid_actions)

        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model(s)[0]               # (ACTION_DIM,)

        best_slot = int(torch.argmax(q[:n]).item())
        return valid_actions[best_slot]

    # ── 메모리 저장 ───────────────────────────────────────────────────────────
    def remember(self, state, action_slot: int, reward, next_state, done,
                 n_next_valid: int):
        """
        action_slot  : 선택한 슬롯 인덱스 (0..ACTION_DIM-1)
        n_next_valid : next_state 의 유효 슬롯 개수 (= len(next get_valid_actions))
        """
        self.memory.append(
            (state, int(action_slot), reward, next_state, done, int(n_next_valid)))

    # ── 학습 ──────────────────────────────────────────────────────────────────
    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, slots, rewards, next_states, dones, n_next_valids = zip(*batch)

        S  = torch.FloatTensor(np.array(states)).to(self.device)
        NS = torch.FloatTensor(np.array(next_states)).to(self.device)
        R  = torch.FloatTensor(rewards).to(self.device)
        D  = torch.FloatTensor(dones).to(self.device)

        # Double DQN: online 모델로 슬롯 선택, target 모델로 값 평가
        with torch.no_grad():
            online_q_next = self.model(NS)
            target_q_next = self.target_model(NS)

        targets = self.model(S).detach().clone()   # (B, ACTION_DIM)

        for i in range(len(slots)):
            if D[i]:
                t = R[i]
            else:
                n = n_next_valids[i]
                if n > 0:
                    # 유효 슬롯 범위 [0, n) 안에서 online argmax → target 값 평가
                    best_slot = int(online_q_next[i, :n].argmax().item())
                    t = R[i] + self.gamma * target_q_next[i, best_slot]
                else:
                    t = R[i]
            targets[i, slots[i]] = t

        self.optimizer.zero_grad()
        loss = nn.MSELoss()(self.model(S), targets)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    # ── 타깃 네트워크 동기화 ──────────────────────────────────────────────────
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def end_episode(self):
        self._episode_count += 1
        if self._episode_count % self.target_update == 0:
            self.update_target()

    # ── 저장 / 로드 ───────────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            "model":   self.model.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.target_model.load_state_dict(ckpt["model"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_min)
        self.model.eval()
