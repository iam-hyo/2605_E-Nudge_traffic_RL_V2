"""OD-2 학습곡선 그래프 — model_rl_*_history.json 3종을 한 장에.

reward / reach-rate(rolling) / fuel / wait 4개 패널, RL 3종 비교.
한글 폰트 설정 필수 (CLAUDE.md).

사용: venv/bin/python util/plot_learning_od2.py [out_dir]
"""
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
from matplotlib import rcParams


def _pick_korean_font():
    import matplotlib.font_manager as fm
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic",
                 "Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans CJK"]:
        for f in fm.fontManager.ttflist:
            if name.lower() in f.name.lower():
                return f.name
    return None


_kf = _pick_korean_font()
if _kf:
    rcParams["font.family"] = _kf
rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np

MODELS = [
    ("model_rl_base",             "rl_base",             "#888888"),
    ("model_rl_signal",           "rl_signal",           "#2176e8"),
    ("model_rl_signal_attention", "rl_signal_attention", "#e8821f"),
]


def _roll(x, w=200):
    x = np.asarray(x, float)
    if len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[w:] - c[:-w]) / w


def main(out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    md = ROOT / "models"
    fig, ax = plt.subplots(2, 2, figsize=(15, 10), dpi=140)
    W = 200
    stats = {}
    for fname, label, col in MODELS:
        p = md / f"{fname}_history.json"
        if not p.exists():
            print(f"  [skip] {p} 없음"); continue
        data = json.load(open(p, encoding="utf-8"))
        hist = data["history"] if isinstance(data, dict) and "history" in data else data
        ep   = [h["episode"] for h in hist]
        rew  = [h["reward"]  for h in hist]
        fuel = [h["fuel"]    for h in hist]
        wait = [h["wait"]    for h in hist]
        reach = [1.0 if h["reached"] else 0.0 for h in hist]
        epx = ep[W-1:] if len(ep) >= W else ep
        ax[0,0].plot(epx, _roll(rew, W),   color=col, lw=1.6, label=label)
        ax[0,1].plot(epx, _roll(reach, W), color=col, lw=1.6, label=label)
        ax[1,0].plot(epx, _roll(fuel, W),  color=col, lw=1.6, label=label)
        ax[1,1].plot(epx, _roll(wait, W),  color=col, lw=1.6, label=label)
        meta = data.get("metadata", {}) if isinstance(data, dict) else {}
        stats[label] = {
            "episodes": len(hist),
            "final_reach": round(float(np.mean(reach[-200:])), 3),
            "final_fuel":  round(float(np.mean(fuel[-200:])), 1),
            "final_wait":  round(float(np.mean(wait[-200:])), 1),
            "elapsed_sec": meta.get("elapsed_sec"),
        }

    ax[0,0].set_title(f"학습 보상 (rolling-{W} 평균)", fontweight="bold")
    ax[0,0].set_xlabel("episode"); ax[0,0].set_ylabel("reward"); ax[0,0].legend(); ax[0,0].grid(alpha=.3)
    ax[0,1].set_title(f"도달률 (rolling-{W})", fontweight="bold")
    ax[0,1].set_xlabel("episode"); ax[0,1].set_ylabel("reach rate"); ax[0,1].set_ylim(-.02,1.02); ax[0,1].legend(); ax[0,1].grid(alpha=.3)
    ax[1,0].set_title(f"에피소드 연료 (rolling-{W}, mL)", fontweight="bold")
    ax[1,0].set_xlabel("episode"); ax[1,0].set_ylabel("fuel mL"); ax[1,0].legend(); ax[1,0].grid(alpha=.3)
    ax[1,1].set_title(f"에피소드 신호대기 (rolling-{W}, s)", fontweight="bold")
    ax[1,1].set_xlabel("episode"); ax[1,1].set_ylabel("wait s"); ax[1,1].legend(); ax[1,1].grid(alpha=.3)
    fig.suptitle("강남구 OD-2 재실험 — RL 3종 학습곡선 (gangnam_topology_add, 15000 ep)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0,0,1,0.97))
    op = out / "gn_od2_learning_curves.png"
    fig.savefig(op, dpi=140, bbox_inches="tight"); plt.close(fig)
    json.dump(stats, open(out / "gn_od2_learning_stats.json","w",encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  saved → {op}")
    print(f"  saved → {out/'gn_od2_learning_stats.json'}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    od = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "output")
    main(od)
