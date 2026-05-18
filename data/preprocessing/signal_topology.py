"""
topology_강남구.json 에 신호 데이터 추가
────────────────────────────────────────────────────────
로직:

1. 신호 유형 결정 (TRA_LIGHT)
   - TRA_LIGHT=0 → signal=null
   - TRA_LIGHT=3 → 2현시 (직진 + 적색)
   - TRA_LIGHT=4 → 3현시 (직진 + 좌회전 + 적색)

2. 사이클 샘플링 (정규분포)
   - 실측 대표값을 평균으로 하는 정규분포에서 샘플링
   - Sun & Liu (2015) Stochastic Eco-routing 방법론:
     실시간 신호가 모든 교차로에서 수집되지 않으므로
     실측 대표값 기반 확률 분포에서 사이클을 샘플링
   - TRA_LIGHT=3: N(131, 30²), clip [60, 180]
   - TRA_LIGHT=4: N(180, 30²), clip [80, 180]
   - 평균: 낮 시간대 실측 default (3구=131s, 4구=180s)
   - σ=30: 실측값 분포 범위(70~212s)에서 추정
   - clip 범위: 실측 최솟값 기반 하한, 실측 최댓값 기반 상한

3. 현시 시간 결정 (실측 비율 적용)
   - T-DATA SPAT 실측 default에서 도출한 현시 비율:
       TRA_LIGHT=3: green/cycle = 42/131 = 32.1%
       TRA_LIGHT=4: green/cycle = 46/180 = 25.6%
                    left/cycle  = 26/180 = 14.4%
   - 샘플링된 cycle에 비율을 곱해 각 현시 시간 산출
   - green + left + red = cycle 정규화

4. 옵셋 결정
   - random.randint(0, cycle // 2)
   - 교차로 간 연동 정보가 없으므로 사이클 절반 범위에서 균등 샘플링
"""

import json
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# 현시 비율 (T-DATA SPAT 낮 시간대 실측 default)
GREEN_RATIO = {3: 42 / 131, 4: 46 / 180}
LEFT_RATIO  = {4: 26 / 180}

# 정규분포 파라미터
CYCLE_DIST = {
    3: {"mean": 131, "std": 30, "lo": 60, "hi": 180},
    4: {"mean": 180, "std": 30, "lo": 80, "hi": 180},
}

def sample_cycle(tra_light):
    p = CYCLE_DIST[tra_light]
    cycle = round(np.random.normal(p["mean"], p["std"]))
    return int(np.clip(cycle, p["lo"], p["hi"]))

def build_signal(info):
    tra_light = info.get("TRA_LIGHT", 0)
    if tra_light not in [3, 4]:
        return None, False

    left_turn_allowed = (tra_light == 4)

    cycle = sample_cycle(tra_light)

    green = round(cycle * GREEN_RATIO[tra_light])
    left  = round(cycle * LEFT_RATIO[4]) if tra_light == 4 else None
    red   = cycle - green - (left or 0)

    if red < 5:
        red   = 5
        green = max(cycle - (left or 0) - red, 5)

    offset = random.randint(0, cycle // 2)

    phases = [{"type": "green", "duration": green}]
    if left is not None:
        phases.append({"type": "left", "duration": left})
    phases.append({"type": "red", "duration": red})

    return {"cycle_length": cycle, "offset": offset, "phases": phases}, left_turn_allowed

# 데이터 로드
with open("data/topology_강남구.json", encoding="utf-8") as f:
    topology = json.load(f)

node_df = pd.read_csv("node_link_lengths_with_lanes.csv", encoding="utf-8-sig")
node_df["NODE_ID"] = node_df["NODE_ID"].astype(str)
node_info = node_df.set_index("NODE_ID").to_dict("index")

# 신호 추가
stats = {"signal_3": 0, "signal_4": 0, "no_signal": 0, "not_found": 0}
cycles_3, cycles_4 = [], []

for node in topology["nodes"]:
    node_id = str(node["id"])
    info = node_info.get(node_id)

    if info is None:
        node["left_turn_allowed"] = False
        node["signal"] = None
        stats["not_found"] += 1
        continue

    signal, left_turn = build_signal(info)
    node["left_turn_allowed"] = left_turn
    node["signal"] = signal

    if signal is None:
        stats["no_signal"] += 1
    elif left_turn:
        stats["signal_4"] += 1
        cycles_4.append(signal["cycle_length"])
    else:
        stats["signal_3"] += 1
        cycles_3.append(signal["cycle_length"])

with open("data/topology_강남구_signal.json", "w", encoding="utf-8") as f:
    json.dump(topology, f, ensure_ascii=False, indent=2)

print("=== 신호 데이터 추가 완료 ===")
print(f"  직진 신호 노드  (TRA_LIGHT=3): {stats['signal_3']}개")
print(f"  좌회전 신호 노드(TRA_LIGHT=4): {stats['signal_4']}개")
print(f"  신호 없음       (TRA_LIGHT=0): {stats['no_signal']}개")
print(f"  CSV 미매칭 노드              : {stats['not_found']}개")
if cycles_3:
    print(f"\nTRA_LIGHT=3 사이클: 평균={np.mean(cycles_3):.1f}s, min={min(cycles_3)}s, max={max(cycles_3)}s")
if cycles_4:
    print(f"TRA_LIGHT=4 사이클: 평균={np.mean(cycles_4):.1f}s, min={min(cycles_4)}s, max={max(cycles_4)}s")
print(f"\n저장 완료: topology_강남구_signal.json")

print("\n=== 신호 노드 샘플 ===")
for s in [n for n in topology["nodes"] if n.get("signal")][:3]:
    print(json.dumps(s, ensure_ascii=False, indent=2))
