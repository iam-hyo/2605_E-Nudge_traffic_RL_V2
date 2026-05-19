"""
viz_scale.py
------------
시각화 자동 스케일 유틸 — 노드 수 / 맵 크기에 따라 마커·도로·폰트 크기를 자동 조정.

6x6 (36노드) 부터 강남구 1000+ 노드까지 같은 시각화 코드로 호환.
"""

from __future__ import annotations
import math


def viz_params(n_nodes: int, map_diag: float) -> dict:
    """
    매개변수
    --------
    n_nodes  : 노드 수
    map_diag : 맵 대각선 길이 (env.map_diag, 좌표 단위)

    반환: 시각화 파라미터 dict
      node_size_signal : 신호 노드 마커 크기 (matplotlib s=)
      node_size_lt     : 좌회전 신호 마커 크기
      node_size_nosig  : 무신호 노드 마커 크기
      link_lw          : 도로 선 굵기
      path_lw          : 경로 선 굵기 (foreground)
      path_bg_lw       : 경로 선 굵기 (white background)
      agent_ms         : 에이전트 마커 크기
      star_ms          : 출발/도착 별 크기
      lt_font          : 좌회전 ← 텍스트 폰트 크기
      lt_offset        : ← 텍스트 위치 오프셋 (좌표 단위)
      legend_fontsize  : 범례 폰트
    """
    # 노드 밀도에 따른 스케일 (격자 36노드 → 1.0, 1000노드 → 0.25)
    density_scale = math.sqrt(36.0 / max(n_nodes, 1))
    density_scale = max(0.18, min(1.6, density_scale))

    # 맵 크기에 따른 텍스트 오프셋 (좌표 단위)
    lt_offset = map_diag * 0.012

    return {
        "node_size_signal": max(20,  int(100 * density_scale)),
        "node_size_lt":     max(28,  int(160 * density_scale)),
        "node_size_nosig":  max(8,   int(22  * density_scale)),
        "link_lw":          max(0.4, 2.0 * density_scale),
        "path_lw":          max(1.8, 2.8 * density_scale),
        "path_bg_lw":       max(3.5, 5.5 * density_scale),
        "agent_ms":         max(7,   int(14  * density_scale)),
        "star_ms":          max(150, int(360 * density_scale)),
        "lt_font":          max(5.0, 7.5 * density_scale),
        "lt_offset":        lt_offset,
        "legend_fontsize":  max(6.0, 8.0 * density_scale),
        "density_scale":    density_scale,
    }
