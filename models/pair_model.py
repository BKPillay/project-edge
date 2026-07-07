from __future__ import annotations

import itertools
import pandas as pd

from models.common import NUMBERS, NUMBER_COLS, minmax


def score_pairs(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    counts = {pair: 0 for pair in itertools.combinations(NUMBERS, 2)}

    for row in df[NUMBER_COLS].values.tolist():
        for pair in itertools.combinations(sorted(map(int, row)), 2):
            counts[pair] += 1

    norm = minmax(counts)
    rows = []
    for pair, count in counts.items():
        rows.append({
            "rank": None,
            "pair": f"{pair[0]}-{pair[1]}",
            "score": round(norm[pair] * 100, 2),
            "appeared": count,
        })

    out = pd.DataFrame(rows).sort_values(["score", "pair"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out
