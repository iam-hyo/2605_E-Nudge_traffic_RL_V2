"""
agent.py
--------
DQNAgent — Double DQN + Dueling + Experience Replay.

모든 RL 모델(base / signal / attention)에서 공통 사용.
모델 종류는 build_model(mode) 로 주입.
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

from util.model import build_model


class DQNAgent:
    def __init__(
        self,
        action_size:    int,
        node_list:      list[str],
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
        self.action_size   = action_size
        self.node_to_idx   = {n: i for i, n in enumerate(node_list)}
        self.idx_to_node   = {i: n for i, n in enumerate(node_list)}

        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model        = build_model(mode, action_size).to(self.device)
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.optimizer    = optim.Adam(self.model.parameters(), lr=lr)
        self.memory: deque = deque(maxlen=memory_size)

        self._episode_count = 0

    # ── 행동 선택 ─────────────────────────────────────────────────────────────
    def act(self, state: np.ndarray, valid_actions: list[str]) -> str:
        if not valid_actions:
            raise ValueError("valid_actions is empty")

        if random.random() <= self.epsilon:
            return random.choice(valid_actions)

        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model(s)[0]

        best = max(valid_actions,
                   key=lambda n: q[self.node_to_idx[n]].item())
        return best

    # ── 메모리 저장 ───────────────────────────────────────────────────────────
    def remember(self, state, action, reward, next_state, done,
                 next_valid: list[str]):
        self.memory.append((state, action, reward, next_state, done, next_valid))

    # ── 학습 ──────────────────────────────────────────────────────────────────
    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, next_valids = zip(*batch)

        S  = torch.FloatTensor(np.array(states)).to(self.device)
        NS = torch.FloatTensor(np.array(next_states)).to(self.device)
        R  = torch.FloatTensor(rewards).to(self.device)
        D  = torch.FloatTensor(dones).to(self.device)

        # Double DQN: online 모델로 행동 선택, target 모델로 값 평가
        with torch.no_grad():
            online_q_next = self.model(NS)
            target_q_next = self.target_model(NS)

        targets = self.model(S).detach().clone()

        for i, (action, nv) in enumerate(zip(actions, next_valids)):
            if D[i]:
                t = R[i]
            else:
                if nv:
                    valid_idx   = [self.node_to_idx[n] for n in nv]
                    best_idx    = valid_idx[
                        online_q_next[i, valid_idx].argmax().item()
                    ]
                    t = R[i] + self.gamma * target_q_next[i, best_idx]
                else:
                    t = R[i]
            targets[i, self.node_to_idx[action]] = t

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
