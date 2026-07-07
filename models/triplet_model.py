
from __future__ import annotations

import itertools
import pandas as pd

from models.common import NUMBER_COLS


def _recent_triplet_sets(df: pd.DataFrame, depth: int = 3):
    recent = []
    for i in range(1, min(depth, len(df)) + 1):
        vals = sorted(map(int, df.iloc[-i][NUMBER_COLS].tolist()))
        recent.append(set(tuple(t) for t in itertools.combinations(vals, 3)))
    return recent


def score_triplets(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    counts = {}

    for row in df[NUMBER_COLS].values.tolist():
        for triplet in itertools.combinations(sorted(map(int, row)), 3):
            counts[triplet] = counts.get(triplet, 0) + 1

    if not counts:
        return pd.DataFrame(columns=[
            "rank", "triplet", "score", "appeared",
            "recent_hit_penalty", "selection_score", "selection_rank", "recency_note"
        ])

    max_count = max(counts.values())
    recent_sets = _recent_triplet_sets(df, 3)

    rows = []
    for triplet, count in counts.items():
        raw_score = round((count / max_count) * 100, 2)

        if len(recent_sets) >= 1 and triplet in recent_sets[0]:
            recent_hit_penalty = 45.0
            recency_note = "appeared in latest draw"
        elif len(recent_sets) >= 2 and triplet in recent_sets[1]:
            recent_hit_penalty = 25.0
            recency_note = "appeared 2 draws ago"
        elif len(recent_sets) >= 3 and triplet in recent_sets[2]:
            recent_hit_penalty = 12.0
            recency_note = "appeared 3 draws ago"
        else:
            recent_hit_penalty = 0.0
            recency_note = "not in last 3 draws"

        selection_score = round(max(0, raw_score - recent_hit_penalty), 2)

        rows.append({
            "rank": None,
            "triplet": "-".join(map(str, triplet)),
            "score": raw_score,
            "appeared": count,
            "recent_hit_penalty": recent_hit_penalty,
            "selection_score": selection_score,
            "recency_note": recency_note,
        })

    raw = pd.DataFrame(rows).sort_values(["score", "triplet"], ascending=[False, True]).reset_index(drop=True)
    raw["rank"] = raw.index + 1

    selected = raw.sort_values(["selection_score", "triplet"], ascending=[False, True]).reset_index(drop=True)
    selected["selection_rank"] = selected.index + 1

    out = raw.merge(selected[["triplet", "selection_rank"]], on="triplet", how="left")
    out = out.sort_values(["selection_rank", "triplet"], ascending=[True, True]).head(top_n).reset_index(drop=True)
    return out
