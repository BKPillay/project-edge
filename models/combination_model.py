
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from models.common import PRIMES
from models.structural_model import combo_structure_score


def _safe_score_map(df: pd.DataFrame, key_col: str, preferred: str = "selection_score", fallback: str = "score"):
    if df is None or len(df) == 0 or key_col not in df.columns:
        return {}
    score_col = preferred if preferred in df.columns else fallback
    if score_col not in df.columns:
        return {}
    return dict(zip(df[key_col].astype(str), df[score_col].astype(float)))


def _pair_key(a: int, b: int) -> str:
    a, b = sorted([int(a), int(b)])
    return f"{a}-{b}"


def _triplet_key(values) -> str:
    return "-".join(map(str, sorted(map(int, values))))


def score_combinations(
    numbers_df: pd.DataFrame,
    pairs_df: pd.DataFrame | None = None,
    triplets_df: pd.DataFrame | None = None,
    top_n: int = 100,
    pool_size: int = 15,
) -> pd.DataFrame:
    '''
    EDGE AI v2.7 Combination Ensemble.

    The combination score now uses the cooled selection layer:
    - individual number selection_score
    - pair selection_score
    - triplet selection_score
    - structural bonus

    This prevents the app from saying "cool 10 and 25" in the individual
    model while still recommending many combinations containing them.
    '''

    if numbers_df is None or len(numbers_df) == 0:
        return pd.DataFrame()

    number_score_col = "selection_score" if "selection_score" in numbers_df.columns else "final_score"
    number_rank_col = "selection_rank" if "selection_rank" in numbers_df.columns else "rank"

    pool = numbers_df.sort_values([number_rank_col, "number"], ascending=[True, True]).head(pool_size).copy()
    number_score_map = dict(zip(pool["number"].astype(int), pool[number_score_col].astype(float)))

    pair_score_map = _safe_score_map(pairs_df, "pair", preferred="selection_score", fallback="score")
    triplet_score_map = _safe_score_map(triplets_df, "triplet", preferred="selection_score", fallback="score")

    pair_default = float(np.mean(list(pair_score_map.values()))) if pair_score_map else 50.0
    triplet_default = float(np.mean(list(triplet_score_map.values()))) if triplet_score_map else 50.0

    rows = []

    for combo in itertools.combinations(sorted(number_score_map.keys()), 5):
        total = sum(combo)
        odd = sum(n % 2 for n in combo)
        low = sum(n <= 18 for n in combo)
        prime = sum(n in PRIMES for n in combo)

        avg_number_score = float(np.mean([number_score_map[n] for n in combo]))

        pair_scores = []
        for a, b in itertools.combinations(combo, 2):
            pair_scores.append(pair_score_map.get(_pair_key(a, b), pair_default))
        avg_pair_score = float(np.mean(pair_scores)) if pair_scores else pair_default

        triplet_scores = []
        for triplet in itertools.combinations(combo, 3):
            triplet_scores.append(triplet_score_map.get(_triplet_key(triplet), triplet_default))
        avg_triplet_score = float(np.mean(triplet_scores)) if triplet_scores else triplet_default

        structural_bonus = combo_structure_score(combo)

        # Convert structural bonus to a mild score contribution.
        structural_score = min(100.0, structural_bonus * 7.5)

        # Current ensemble weights:
        # 55% individual selection strength
        # 20% pair selection strength
        # 10% triplet selection strength
        # 15% structural suitability
        ensemble_score = (
            0.55 * avg_number_score
            + 0.20 * avg_pair_score
            + 0.10 * avg_triplet_score
            + 0.15 * structural_score
        )

        # Extra cooling for combinations containing multiple latest-hit numbers
        # is already mostly handled by individual selection_score, but this
        # discourages stacked repeat-heavy lines.
        repeat_penalty = 0.0
        if "repeat_penalty" in pool.columns:
            repeat_map = dict(zip(pool["number"].astype(int), pool["repeat_penalty"].astype(float)))
            cooled_count = sum(1 for n in combo if repeat_map.get(n, 0) > 0)
            if cooled_count >= 3:
                repeat_penalty = 5.0
            elif cooled_count == 2:
                repeat_penalty = 2.5

        final_score = round(max(0, ensemble_score - repeat_penalty), 2)

        rows.append({
            "rank": None,
            "combination": "-".join(map(str, combo)),
            "score": final_score,
            "avg_number_selection_score": round(avg_number_score, 2),
            "avg_pair_selection_score": round(avg_pair_score, 2),
            "avg_triplet_selection_score": round(avg_triplet_score, 2),
            "structural_bonus": structural_bonus,
            "repeat_stack_penalty": repeat_penalty,
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
