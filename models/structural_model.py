from __future__ import annotations

import pandas as pd

from models.common import add_features


def structural_summary(df: pd.DataFrame) -> dict:
    feat = add_features(df)
    structure_counts = feat["structure"].value_counts(normalize=True).head(10).to_dict()
    sum_stats = {
        "mean_sum": round(float(feat["sum"].mean()), 2),
        "median_sum": round(float(feat["sum"].median()), 2),
        "std_sum": round(float(feat["sum"].std()), 2),
        "over_92_5_rate": round(float(feat["over_92_5"].mean()) * 100, 2),
    }
    return {
        "sum_stats": sum_stats,
        "top_structures": {str(k): round(float(v) * 100, 2) for k, v in structure_counts.items()},
    }


def combo_structure_score(combo) -> float:
    s = sum(combo)
    odd = sum(n % 2 for n in combo)
    low = sum(n <= 18 for n in combo)
    score = 0
    score += 5 if 65 <= s <= 120 else 0
    score += 4 if odd in (2, 3) else 0
    score += 4 if low in (2, 3) else 0
    return score
