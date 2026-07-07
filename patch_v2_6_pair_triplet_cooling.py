from pathlib import Path

PAIR_MODEL = Path("models/pair_model.py")
TRIPLET_MODEL = Path("models/triplet_model.py")
APP_PATH = Path("app.py")

PAIR_MODEL.write_text('\nfrom __future__ import annotations\n\nimport itertools\nimport pandas as pd\n\nfrom models.common import NUMBERS, NUMBER_COLS, minmax\n\n\ndef _recent_pair_sets(df: pd.DataFrame, depth: int = 3):\n    recent = []\n    for i in range(1, min(depth, len(df)) + 1):\n        vals = sorted(map(int, df.iloc[-i][NUMBER_COLS].tolist()))\n        recent.append(set(tuple(p) for p in itertools.combinations(vals, 2)))\n    return recent\n\n\ndef score_pairs(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:\n    counts = {pair: 0 for pair in itertools.combinations(NUMBERS, 2)}\n\n    for row in df[NUMBER_COLS].values.tolist():\n        for pair in itertools.combinations(sorted(map(int, row)), 2):\n            counts[pair] += 1\n\n    norm = minmax(counts)\n    recent_sets = _recent_pair_sets(df, 3)\n\n    rows = []\n    for pair, count in counts.items():\n        raw_score = round(norm[pair] * 100, 2)\n\n        if len(recent_sets) >= 1 and pair in recent_sets[0]:\n            recent_hit_penalty = 30.0\n            recency_note = "appeared in latest draw"\n        elif len(recent_sets) >= 2 and pair in recent_sets[1]:\n            recent_hit_penalty = 18.0\n            recency_note = "appeared 2 draws ago"\n        elif len(recent_sets) >= 3 and pair in recent_sets[2]:\n            recent_hit_penalty = 9.0\n            recency_note = "appeared 3 draws ago"\n        else:\n            recent_hit_penalty = 0.0\n            recency_note = "not in last 3 draws"\n\n        selection_score = round(max(0, raw_score - recent_hit_penalty), 2)\n\n        rows.append({\n            "rank": None,\n            "pair": f"{pair[0]}-{pair[1]}",\n            "score": raw_score,\n            "appeared": count,\n            "recent_hit_penalty": recent_hit_penalty,\n            "selection_score": selection_score,\n            "recency_note": recency_note,\n        })\n\n    raw = pd.DataFrame(rows).sort_values(["score", "pair"], ascending=[False, True]).reset_index(drop=True)\n    raw["rank"] = raw.index + 1\n\n    selected = raw.sort_values(["selection_score", "pair"], ascending=[False, True]).reset_index(drop=True)\n    selected["selection_rank"] = selected.index + 1\n\n    out = raw.merge(selected[["pair", "selection_rank"]], on="pair", how="left")\n    out = out.sort_values(["selection_rank", "pair"], ascending=[True, True]).head(top_n).reset_index(drop=True)\n    return out\n', encoding="utf-8")
TRIPLET_MODEL.write_text('\nfrom __future__ import annotations\n\nimport itertools\nimport pandas as pd\n\nfrom models.common import NUMBER_COLS\n\n\ndef _recent_triplet_sets(df: pd.DataFrame, depth: int = 3):\n    recent = []\n    for i in range(1, min(depth, len(df)) + 1):\n        vals = sorted(map(int, df.iloc[-i][NUMBER_COLS].tolist()))\n        recent.append(set(tuple(t) for t in itertools.combinations(vals, 3)))\n    return recent\n\n\ndef score_triplets(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:\n    counts = {}\n\n    for row in df[NUMBER_COLS].values.tolist():\n        for triplet in itertools.combinations(sorted(map(int, row)), 3):\n            counts[triplet] = counts.get(triplet, 0) + 1\n\n    if not counts:\n        return pd.DataFrame(columns=[\n            "rank", "triplet", "score", "appeared",\n            "recent_hit_penalty", "selection_score", "selection_rank", "recency_note"\n        ])\n\n    max_count = max(counts.values())\n    recent_sets = _recent_triplet_sets(df, 3)\n\n    rows = []\n    for triplet, count in counts.items():\n        raw_score = round((count / max_count) * 100, 2)\n\n        if len(recent_sets) >= 1 and triplet in recent_sets[0]:\n            recent_hit_penalty = 45.0\n            recency_note = "appeared in latest draw"\n        elif len(recent_sets) >= 2 and triplet in recent_sets[1]:\n            recent_hit_penalty = 25.0\n            recency_note = "appeared 2 draws ago"\n        elif len(recent_sets) >= 3 and triplet in recent_sets[2]:\n            recent_hit_penalty = 12.0\n            recency_note = "appeared 3 draws ago"\n        else:\n            recent_hit_penalty = 0.0\n            recency_note = "not in last 3 draws"\n\n        selection_score = round(max(0, raw_score - recent_hit_penalty), 2)\n\n        rows.append({\n            "rank": None,\n            "triplet": "-".join(map(str, triplet)),\n            "score": raw_score,\n            "appeared": count,\n            "recent_hit_penalty": recent_hit_penalty,\n            "selection_score": selection_score,\n            "recency_note": recency_note,\n        })\n\n    raw = pd.DataFrame(rows).sort_values(["score", "triplet"], ascending=[False, True]).reset_index(drop=True)\n    raw["rank"] = raw.index + 1\n\n    selected = raw.sort_values(["selection_score", "triplet"], ascending=[False, True]).reset_index(drop=True)\n    selected["selection_rank"] = selected.index + 1\n\n    out = raw.merge(selected[["triplet", "selection_rank"]], on="triplet", how="left")\n    out = out.sort_values(["selection_rank", "triplet"], ascending=[True, True]).head(top_n).reset_index(drop=True)\n    return out\n', encoding="utf-8")

app = APP_PATH.read_text(encoding="utf-8")

# Sort pair/triplet outputs by selection_rank if present after loading CSVs.
load_marker = """pairs = read_csv("pair_predictions.csv")
triplets = read_csv("triplet_predictions.csv")
combos = read_csv("combination_predictions.csv")
"""

load_replacement = """pairs = read_csv("pair_predictions.csv")
if "selection_rank" in pairs.columns:
    pairs = pairs.sort_values(["selection_rank", "pair"]).reset_index(drop=True)
triplets = read_csv("triplet_predictions.csv")
if "selection_rank" in triplets.columns:
    triplets = triplets.sort_values(["selection_rank", "triplet"]).reset_index(drop=True)
combos = read_csv("combination_predictions.csv")
"""

if load_marker in app and "pairs = pairs.sort_values" not in app:
    app = app.replace(load_marker, load_replacement)

notice_marker = """    if "repeat_penalty" in numbers.columns:
        st.info("v2.5 uses learned recency cooling: final_score is raw strength, repeat_penalty is based on historical repeat behaviour, and selection_rank drives today’s recommendations.")
"""

notice_replacement = """    if "repeat_penalty" in numbers.columns:
        st.info("v2.6 uses selection_rank across numbers, pairs, and triplets. Raw scores remain visible, while recent-hit penalties cool repeated selections.")
"""

if notice_marker in app:
    app = app.replace(notice_marker, notice_replacement)

APP_PATH.write_text(app, encoding="utf-8")

print("EDGE AI v2.6 pair/triplet cooling patch applied.")
print("Next run: python scripts\\run_model_update.py")
