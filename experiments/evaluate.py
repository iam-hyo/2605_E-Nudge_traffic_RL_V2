"""
experiments/evaluate.py
-----------------------
output/{timestamp}/results.csv 를 읽어 KPI 통계 출력.
단독 실행: python experiments/evaluate.py output/20240101_120000/results.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def evaluate(csv_path: str):
    import csv
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        for k in r:
            try:
                r[k] = float(r[k]) if '.' in str(r[k]) else (
                    int(r[k]) if r[k].lstrip('-').isdigit() else r[k])
            except (ValueError, AttributeError):
                pass

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["model"], r["route"], r["time_slot"])].append(r)

    print(f"\n{'='*90}")
    print(f" KPI 요약 — {csv_path}")
    print(f"{'='*90}")

    kpi_cols = [
        ("연료(mL)",    "fuel_total",  ".1f"),
        ("시간(s)",     lambda r: r["travel_time"]+r["wait_time"], ".1f"),
        ("거리(m)",     "distance",    ".0f"),
        ("대기시간(s)", "wait_time",   ".1f"),
        ("도달률",      "reached",     ".0%"),
    ]

    header = f"{'모델':<30} {'경로':<12} {'시간대':<10}"
    for col_name, _, _ in kpi_cols:
        header += f" {col_name:>12}"
    print(header)
    print("─" * 90)

    for (model, route, ts), reps in sorted(grouped.items()):
        row_str = f"{model:<30} {route:<12} {ts:<10}"
        for _, key, fmt in kpi_cols:
            if callable(key):
                vals = [key(r) for r in reps]
            else:
                vals = [r[key] for r in reps]
            mean = float(np.mean(vals))
            std  = float(np.std(vals))
            if fmt == ".0%":
                row_str += f" {mean:>11.0%} "
            else:
                row_str += f" {mean:>6{fmt}}±{std:<5{fmt}}"
        print(row_str)

    print(f"\n총 레코드: {len(rows)}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        # 최신 결과 자동 탐색
        results = sorted(Path("output").glob("*/results.csv"), reverse=True)
        if not results:
            print("output/ 에 results.csv 없음. run_experiment.py 를 먼저 실행하세요.")
            sys.exit(1)
        path = str(results[0])
        print(f"최신 결과 사용: {path}")
    evaluate(path)
