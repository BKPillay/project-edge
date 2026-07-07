
from __future__ import annotations

import itertools
import pandas as pd

from models.common import NUMBERS, NUMBER_COLS, minmax


def _recent_pair_sets(df: pd.DataFrame, depth: int = 3):
    recent = []
    for i in range(1, min(depth, len(df)) + 1):
        vals = sorted(map(int, df.iloc[-i][NUMBER_COLS].tolist()))
        recent.append(set(tuple(p) for p in itertools.combinations(vals, 2)))
    return recent


def score_pairs(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    counts = {pair: 0 for pair in itertools.combinations(NUMBERS, 2)}

    for row in df[NUMBER_COLS].values.tolist():
        for pair in itertools.combinations(sorted(map(int, row)), 2):
            counts[pair] += 1

    norm = minmax(counts)
    recent_sets = _recent_pair_sets(df, 3)

    rows = []
    for pair, count in counts.items():
        raw_score = round(norm[pair] * 100, 2)

        if len(recent_sets) >= 1 and pair in recent_sets[0]:
            recent_hit_penalty = 30.0
            recency_note = "appeared in latest draw"
        elif len(recent_sets) >= 2 and pair in recent_sets[1]:
            recent_hit_penalty = 18.0
            recency_note = "appeared 2 draws ago"
        elif len(recent_sets) >= 3 and pair in recent_sets[2]:
            recent_hit_penalty = 9.0
            recency_note = "appeared 3 draws ago"
        else:
            recent_hit_penalty = 0.0
            recency_note = "not in last 3 draws"

        selection_score = round(max(0, raw_score - recent_hit_penalty), 2)

        rows.append({
            "rank": None,
            "pair": f"{pair[0]}-{pair[1]}",
            "score": raw_score,
            "appeared": count,
            "recent_hit_penalty": recent_hit_penalty,
            "selection_score": selection_score,
            "recency_note": recency_note,
        })

    raw = pd.DataFrame(rows).sort_values(["score", "pair"], ascending=[False, True]).reset_index(drop=True)
    raw["rank"] = raw.index + 1

    selected = raw.sort_values(["selection_score", "pair"], ascending=[False, True]).reset_index(drop=True)
    selected["selection_rank"] = selected.index + 1

    out = raw.merge(selected[["pair", "selection_rank"]], on="pair", how="left")
    out = out.sort_values(["selection_rank", "pair"], ascending=[True, True]).head(top_n).reset_index(drop=True)
    return out
