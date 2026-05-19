"""
model.py
--------
QNetwork 세 가지 변형 (State 229d 기준).

  QNetworkBase      — 신호 State 미사용 (마스킹)
  QNetworkSignal    — 신호 State 포함, 단순 MLP + Dueling
  QNetworkAttention — 신호 State 포함, 노드 13토큰 × 14d Self-Attention

모두 Dueling DQN 구조 (Value + Advantage stream).

State 229d 구조 (environment.py와 동기):
  s[0:5]      위치 (5d)
  s[5:8]      시간 (3d)
  s[8:17]     현재 신호 (9d)
  s[17:61]    1-hop 노드 4×11=44d  (각 11d = pos 2 + sig 9)
  s[61:69]    1-hop 링크 4×2=8d    (각 2d = len + speed)
  s[69:157]   2-hop 노드 8×11=88d  (각 11d = pos 2 + sig 9)
  s[157:229]  2-hop 링크 12×6=72d  (각 6d = len + speed + parent_onehot[4])
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── State 구조 상수 (environment.py와 일치) ──────────────────────────────────
K_HOP1     = 4
N_HOP2     = 8
L_HOP2     = 12

POS_DIM    = 5
TIME_DIM   = 3
SIG_DIM    = 9
NODE_DIM   = 11       # pos(2) + sig(9)
LINK1_DIM  = 2        # len + speed
LINK2_DIM  = 6        # len + speed + parent_onehot[4]

# 슬라이스 인덱스
IDX_POS_END        = POS_DIM                                  # 5
IDX_TIME_END       = IDX_POS_END + TIME_DIM                   # 8
IDX_CUR_SIG_END    = IDX_TIME_END + SIG_DIM                   # 17
IDX_HOP1_NODES_END = IDX_CUR_SIG_END + K_HOP1 * NODE_DIM      # 61
IDX_HOP1_LINKS_END = IDX_HOP1_NODES_END + K_HOP1 * LINK1_DIM  # 69
IDX_HOP2_NODES_END = IDX_HOP1_LINKS_END + N_HOP2 * NODE_DIM   # 157
IDX_HOP2_LINKS_END = IDX_HOP2_NODES_END + L_HOP2 * LINK2_DIM  # 229
STATE_SIZE         = IDX_HOP2_LINKS_END                        # 229

# Attention 모델용
NODE_TOK_DIM   = NODE_DIM + 3        # +hop_onehot[3] → 14
GLOBAL_FEAT    = POS_DIM + TIME_DIM  # 8
LINK_BLOCK_DIM = K_HOP1 * LINK1_DIM + L_HOP2 * LINK2_DIM   # 8 + 72 = 80

EMBED_DIM      = 64
N_NODE_TOKENS  = 1 + K_HOP1 + N_HOP2   # cur + hop1 + hop2 = 13


def _dueling_head(in_dim: int, action_size: int) -> tuple[nn.Module, nn.Module]:
    value = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 1))
    adv   = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(),
                          nn.Linear(128, action_size))
    return value, adv


def _q_from_dueling(value: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    return value + (adv - adv.mean(dim=1, keepdim=True))


# ── Base 모델 ─────────────────────────────────────────────────────────────────
class QNetworkBase(nn.Module):
    """
    신호 정보 미사용. 위치·시간·1-hop/2-hop 위치/링크 정보만 사용.
    State 차원은 통일 (229d) 하되, forward 진입 시 신호 차원을 0으로 마스킹.
    """

    def __init__(self, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 512)
        self.fc2 = nn.Linear(512, 256)
        self.value, self.adv = _dueling_head(256, action_size)

    def _mask_signal(self, state: torch.Tensor) -> torch.Tensor:
        x = state.clone()
        # 현재 신호
        x[:, IDX_TIME_END:IDX_CUR_SIG_END] = 0.0
        # 1-hop 노드 각각의 신호 부분 (pos 뒤 9d)
        for k in range(K_HOP1):
            base = IDX_CUR_SIG_END + k * NODE_DIM
            x[:, base + 2 : base + NODE_DIM] = 0.0
        # 2-hop 노드 각각의 신호 부분
        for n in range(N_HOP2):
            base = IDX_HOP1_LINKS_END + n * NODE_DIM
            x[:, base + 2 : base + NODE_DIM] = 0.0
        return x

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self._mask_signal(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return _q_from_dueling(self.value(x), self.adv(x))


# ── Signal 모델 ───────────────────────────────────────────────────────────────
class QNetworkSignal(nn.Module):
    """신호 State 포함, 단순 MLP + Dueling."""

    def __init__(self, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 512)
        self.fc2 = nn.Linear(512, 256)
        self.value, self.adv = _dueling_head(256, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return _q_from_dueling(self.value(x), self.adv(x))


# ── Attention 모델 ────────────────────────────────────────────────────────────
class QNetworkAttention(nn.Module):
    """
    신호 State 포함 + Self-Attention.

    아키텍처:
      1. flat 229d → 토큰 분해
         · 노드 토큰 13개 × 14d (pos 2 + sig 9 + hop_onehot 3)
         · 글로벌 컨텍스트 (위치 + 시간 + 현재 신호) → 쿼리
         · 링크 블록 80d → MLP 압축 (별도 보조 feature)
      2. 쿼리(글로벌)로 노드 토큰에 cross-attention
      3. fused = [global, attn_out, link_emb] → Dueling head

    노드 토큰 구조:
      [0]      = cur 노드 (hop_onehot=[1,0,0])
      [1..4]   = 1-hop K=4 (hop_onehot=[0,1,0])
      [5..12]  = 2-hop N=8 (hop_onehot=[0,0,1])
    """

    def __init__(self, action_size: int):
        super().__init__()
        # 쿼리: 위치(5) + 시간(3) + 현재 신호(9) = 17d
        self.query_enc = nn.Sequential(
            nn.Linear(POS_DIM + TIME_DIM + SIG_DIM, EMBED_DIM), nn.ReLU(),
        )
        # 노드 토큰 인코더 (14d → 64)
        self.node_enc = nn.Sequential(
            nn.Linear(NODE_TOK_DIM, EMBED_DIM), nn.ReLU(),
        )
        # 링크 블록 인코더 (80d → 64)
        self.link_enc = nn.Sequential(
            nn.Linear(LINK_BLOCK_DIM, 128), nn.ReLU(),
            nn.Linear(128, EMBED_DIM), nn.ReLU(),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=EMBED_DIM, num_heads=4, batch_first=True,
        )

        # 전역 글로벌(위치+시간만, 신호 제외 — 신호는 쿼리에 포함됨)
        self.global_enc = nn.Sequential(
            nn.Linear(GLOBAL_FEAT, EMBED_DIM), nn.ReLU(),
        )

        fused = EMBED_DIM * 3   # global(64) + attn_out(64) + link_emb(64)
        self.value, self.adv = _dueling_head(fused, action_size)

    @staticmethod
    def _build_node_tokens(state: torch.Tensor) -> torch.Tensor:
        """
        flat state → 노드 토큰 (B, 13, 14).
        토큰 = [cur, hop1×4, hop2×8], 각 14d = pos(2) + sig(9) + hop_onehot(3).
        """
        B   = state.shape[0]
        dev = state.device

        pos     = state[:, 0:2]                                  # cur 좌표만
        cur_sig = state[:, IDX_TIME_END:IDX_CUR_SIG_END]         # (B, 9)

        hop1_nodes = state[:, IDX_CUR_SIG_END:IDX_HOP1_NODES_END]\
                     .view(B, K_HOP1, NODE_DIM)                  # (B, 4, 11)
        hop2_nodes = state[:, IDX_HOP1_LINKS_END:IDX_HOP2_NODES_END]\
                     .view(B, N_HOP2, NODE_DIM)                  # (B, 8, 11)

        # 1) cur 토큰: pos(2) + cur_sig(9) + [1,0,0]
        cur_onehot = torch.zeros(B, 3, device=dev)
        cur_onehot[:, 0] = 1.0
        cur_tok = torch.cat([pos, cur_sig, cur_onehot], dim=1).unsqueeze(1)   # (B,1,14)

        # 2) hop1 토큰: 11d + [0,1,0]
        hop1_oh = torch.zeros(B, K_HOP1, 3, device=dev)
        hop1_oh[:, :, 1] = 1.0
        hop1_tok = torch.cat([hop1_nodes, hop1_oh], dim=-1)                   # (B,4,14)

        # 3) hop2 토큰: 11d + [0,0,1]
        hop2_oh = torch.zeros(B, N_HOP2, 3, device=dev)
        hop2_oh[:, :, 2] = 1.0
        hop2_tok = torch.cat([hop2_nodes, hop2_oh], dim=-1)                   # (B,8,14)

        return torch.cat([cur_tok, hop1_tok, hop2_tok], dim=1)                # (B,13,14)

    @staticmethod
    def _pad_mask(node_tokens: torch.Tensor) -> torch.Tensor:
        """
        패딩 토큰 감지: environment에서 패딩 노드는 pos=(-1,-1) sentinel 사용.
        실제 노드는 pos ∈ [0,1] 이므로 pos_x < 0 으로 안전하게 구분 가능.
        (이전 'all-zero' 방식은 좌상단 코너 + 비신호 노드를 패딩으로 오인식)
        cur 토큰은 항상 실제 (pos 비제로) → 마스크 False 보장.
        """
        pos_x = node_tokens[..., 0]                       # (B, 13)
        mask  = pos_x < 0.0                                # (B, 13) — padding 표식
        mask[:, 0] = False                                 # cur 토큰 보호
        return mask

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]

        # ── 분해 ──────────────────────────────────────────────────────────────
        pos_block   = state[:, 0:IDX_POS_END]                                # (B, 5)
        time_block  = state[:, IDX_POS_END:IDX_TIME_END]                     # (B, 3)
        cur_sig     = state[:, IDX_TIME_END:IDX_CUR_SIG_END]                 # (B, 9)
        hop1_links  = state[:, IDX_HOP1_NODES_END:IDX_HOP1_LINKS_END]        # (B, 8)
        hop2_links  = state[:, IDX_HOP2_NODES_END:IDX_HOP2_LINKS_END]        # (B, 72)

        # ── 쿼리: 위치 + 시간 + 현재 신호 ─────────────────────────────────────
        q_in   = torch.cat([pos_block, time_block, cur_sig], dim=1)          # (B, 17)
        q_emb  = self.query_enc(q_in).unsqueeze(1)                           # (B, 1, 64)

        # ── 노드 토큰 + Attention ────────────────────────────────────────────
        node_tok  = self._build_node_tokens(state)                           # (B, 13, 14)
        pad_mask  = self._pad_mask(node_tok)                                 # (B, 13)
        node_emb  = self.node_enc(node_tok)                                  # (B, 13, 64)

        attn_out, _ = self.attn(q_emb, node_emb, node_emb,
                                key_padding_mask=pad_mask)
        attn_out = attn_out.squeeze(1)                                       # (B, 64)

        # ── 링크 보조 feature ─────────────────────────────────────────────────
        link_block = torch.cat([hop1_links, hop2_links], dim=1)              # (B, 80)
        link_emb   = self.link_enc(link_block)                                # (B, 64)

        # ── 전역 컨텍스트 ─────────────────────────────────────────────────────
        global_emb = self.global_enc(
            torch.cat([pos_block, time_block], dim=1)                        # (B, 8)
        )                                                                     # (B, 64)

        fused = torch.cat([global_emb, attn_out, link_emb], dim=1)           # (B, 192)
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
