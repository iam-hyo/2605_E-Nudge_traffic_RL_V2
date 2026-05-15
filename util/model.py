"""
model.py
--------
QNetwork 세 가지 변형.

  QNetworkBase      — 신호 State 없음, Linear only
  QNetworkSignal    — 신호 State 포함, Linear only
  QNetworkAttention — 신호 State 포함, Self-Attention (1-hop/2-hop 이웃)

모두 Dueling DQN 구조 (Value + Advantage stream).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# State 구조 상수
GLOBAL_DIM  = 15    # s[0–14]: 위치(7) + 시간(3) + 현재신호(5)
K_HOP1      = 4
M_HOP2      = 4
HOP1_FEAT   = 8     # 1-hop 이웃 1개당 피처 수
HOP2_FEAT   = 3     # 2-hop 이웃 1개당 피처 수
STATE_SIZE  = 59

# Base 모델용: 신호 피처 제외한 차원
# 위치(7) + 시간(3) + 1-hop 위치·속도·길이만(4×4=16) + 2-hop 속도만(4×1=4) = 30
BASE_DIM    = 30
EMBED_DIM   = 64


def _dueling_head(in_dim: int, action_size: int) -> tuple[nn.Module, nn.Module]:
    value = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 1))
    adv   = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(),
                          nn.Linear(64, action_size))
    return value, adv


def _q_from_dueling(value: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    return value + (adv - adv.mean(dim=1, keepdim=True))


# ── Base 모델 ─────────────────────────────────────────────────────────────────
class QNetworkBase(nn.Module):
    """신호 정보 미사용. 위치·시간·1-hop 속도/거리만 사용."""

    def __init__(self, action_size: int):
        super().__init__()
        # 신호 관련 피처(s[10–14], s[20–22 per hop1, s[47–58]) 제외한 간소화 State 사용
        # 실제로는 전체 State 59d 입력 후 신호 차원을 0으로 마스킹 — 인터페이스 통일
        self.fc1 = nn.Linear(STATE_SIZE, 256)
        self.fc2 = nn.Linear(256, 128)
        self.value, self.adv = _dueling_head(128, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # 신호 관련 인덱스를 0으로 마스킹 (s[10–14], 1-hop sin/cos/remain/has_lt)
        x = state.clone()
        x[:, 10:15] = 0.0          # 현재 신호
        for k in range(K_HOP1):
            base = GLOBAL_DIM + k * HOP1_FEAT
            x[:, base + 4:base + 8] = 0.0   # sin, cos, remain, has_lt
        x[:, GLOBAL_DIM + K_HOP1 * HOP1_FEAT:] = 0.0  # 2-hop 전체

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return _q_from_dueling(self.value(x), self.adv(x))


# ── Signal 모델 ───────────────────────────────────────────────────────────────
class QNetworkSignal(nn.Module):
    """신호 State 포함, 단순 MLP + Dueling."""

    def __init__(self, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 256)
        self.fc2 = nn.Linear(256, 128)
        self.value, self.adv = _dueling_head(128, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return _q_from_dueling(self.value(x), self.adv(x))


# ── Attention 모델 ────────────────────────────────────────────────────────────
class QNetworkAttention(nn.Module):
    """
    신호 State 포함 + Self-Attention.

    Query  = 글로벌 컨텍스트 s[0–14] → Linear → 64d
    Key/Value = 1-hop × 4 + 2-hop × 4 토큰 → Linear → 64d each
    """

    def __init__(self, action_size: int):
        super().__init__()
        self.global_enc = nn.Sequential(
            nn.Linear(GLOBAL_DIM, EMBED_DIM), nn.ReLU()
        )
        self.hop1_enc = nn.Linear(HOP1_FEAT, EMBED_DIM)
        self.hop2_enc = nn.Linear(HOP2_FEAT, EMBED_DIM)

        self.attn = nn.MultiheadAttention(
            embed_dim=EMBED_DIM, num_heads=4, batch_first=True
        )

        fused = EMBED_DIM * 2   # global(64) + attn_out(64)
        self.value, self.adv = _dueling_head(fused, action_size)

    def _pad_mask(self, state: torch.Tensor, B: int) -> torch.Tensor:
        """패딩 토큰(모든 값 0) True → attention에서 -inf."""
        h1 = state[:, GLOBAL_DIM: GLOBAL_DIM + K_HOP1 * HOP1_FEAT]
        h1 = h1.view(B, K_HOP1, HOP1_FEAT)
        pad1 = (h1.abs().sum(-1) == 0)

        h2 = state[:, GLOBAL_DIM + K_HOP1 * HOP1_FEAT:]
        h2 = h2.view(B, M_HOP2, HOP2_FEAT)
        pad2 = (h2.abs().sum(-1) == 0)

        return torch.cat([pad1, pad2], dim=1)   # (B, K+M)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]

        g_enc = self.global_enc(state[:, :GLOBAL_DIM])   # (B, 64)

        h1 = state[:, GLOBAL_DIM: GLOBAL_DIM + K_HOP1 * HOP1_FEAT]
        h1 = F.relu(self.hop1_enc(h1.view(B, K_HOP1, HOP1_FEAT)))  # (B,4,64)

        h2 = state[:, GLOBAL_DIM + K_HOP1 * HOP1_FEAT:]
        h2 = F.relu(self.hop2_enc(h2.view(B, M_HOP2, HOP2_FEAT)))  # (B,4,64)

        query   = g_enc.unsqueeze(1)                           # (B,1,64)
        context = torch.cat([h1, h2], dim=1)                   # (B,8,64)
        mask    = self._pad_mask(state, B)

        attn_out, _ = self.attn(query, context, context,
                                key_padding_mask=mask)
        attn_out = attn_out.squeeze(1)                         # (B,64)

        fused = torch.cat([g_enc, attn_out], dim=1)            # (B,128)
        return _q_from_dueling(self.value(fused), self.adv(fused))


# ── 팩토리 함수 ───────────────────────────────────────────────────────────────
def build_model(mode: str, action_size: int) -> nn.Module:
    """
    mode: 'base' | 'signal' | 'attention'
    """
    if mode == "base":
        return QNetworkBase(action_size)
    elif mode == "signal":
        return QNetworkSignal(action_size)
    elif mode == "attention":
        return QNetworkAttention(action_size)
    raise ValueError(f"Unknown mode: {mode}")
