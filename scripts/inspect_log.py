import json
with open("data/self_play/games_20260421_175525.jsonl", encoding="utf-8-sig") as f:
    d = json.loads(f.readline())
kyoku0 = d["kyoku_logs"][0]
print("=== kyoku0 last 6 events ===")
for e in kyoku0[-6:]:
    print(json.dumps(e, ensure_ascii=False))
kyoku1 = d["kyoku_logs"][1]
print("=== kyoku1 last 6 events ===")
for e in kyoku1[-6:]:
    print(json.dumps(e, ensure_ascii=False))
print(f"total kyoku: {len(d['kyoku_logs'])}")
print(f"agent_names: {d['agent_names']}")
print(f"final_scores: {d['final_scores']}")
