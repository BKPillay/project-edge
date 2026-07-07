from __future__ import annotations

import itertools
import pandas as pd

from models.common import NUMBER_COLS


def score_triplets(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    counts = {}

    for row in df[NUMBER_COLS].values.tolist():
        for triplet in itertools.combinations(sorted(map(int, row)), 3):
            counts[triplet] = counts.get(triplet, 0) + 1

    if not counts:
        return pd.DataFrame(columns=["rank", "triplet", "score", "appeared"])

    rows = [{"rank": None, "triplet": "-".join(map(str, k)), "score": v, "appeared": v} for k, v in counts.items()]
    out = pd.DataFrame(rows).sort_values(["score", "triplet"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    out["score"] = (out["score"] / out["score"].max() * 100).round(2)
    out["rank"] = out.index + 1
    return out
