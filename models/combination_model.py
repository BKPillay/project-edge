from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from models.common import PRIMES
from models.structural_model import combo_structure_score


def score_combinations(numbers_df: pd.DataFrame, top_n: int = 100, pool_size: int = 15) -> pd.DataFrame:
    pool = numbers_df.head(pool_size).copy()
    score_map = dict(zip(pool["number"].astype(int), pool["final_score"].astype(float)))

    rows = []
    for combo in itertools.combinations(sorted(score_map.keys()), 5):
        total = sum(combo)
        odd = sum(n % 2 for n in combo)
        low = sum(n <= 18 for n in combo)
        prime = sum(n in PRIMES for n in combo)

        avg_score = float(np.mean([score_map[n] for n in combo]))
        structural = combo_structure_score(combo)
        final = round(avg_score + structural, 2)

        rows.append({
            "rank": None,
            "combination": "-".join(map(str, combo)),
            "score": final,
            "avg_number_score": round(avg_score, 2),
            "structural_bonus": structural,
            "sum": total,
            "odd": odd,
            "even": 5 - odd,
            "low": low,
            "high": 5 - low,
            "prime": prime,
        })

    out = pd.DataFrame(rows).sort_values(["score", "combination"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out
