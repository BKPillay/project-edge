from __future__ import annotations

import itertools
import pandas as pd

from models.common import NUMBER_COLS
from models.number_model import score_numbers
from models.over_under_model import predict_over_under
from models.pair_model import score_pairs
from models.triplet_model import score_triplets


def build_latest_draw_review(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(df) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    pre = df.iloc[:-1].copy()
    latest = df.iloc[-1]

    pre_numbers = score_numbers(pre)
    pre_ou = predict_over_under(pre)
    pre_pairs = score_pairs(pre, 100)
    pre_triplets = score_triplets(pre, 100)

    nmap = pre_numbers.set_index("number").to_dict(orient="index")
    pair_map = pre_pairs.set_index("pair").to_dict(orient="index") if len(pre_pairs) else {}
    triplet_map = pre_triplets.set_index("triplet").to_dict(orient="index") if len(pre_triplets) else {}

    latest_numbers = sorted([int(latest[c]) for c in NUMBER_COLS])
    actual_sum = sum(latest_numbers)
    actual_ou = "Over 92.5" if actual_sum > 92.5 else "Under 92.5"

    ball_rows = []
    for n in latest_numbers:
        info = nmap[n]
        rank = int(info["rank"])

        if rank <= 3:
            insight = "Elite pre-draw pick"
        elif rank <= 10:
            insight = "Strong pre-draw pick"
        elif rank <= 20:
            insight = "Mid-ranked pick"
        else:
            insight = "Model underweighted this ball"

        ball_rows.append({
            "draw_number": int(latest["draw_number"]),
            "draw_date": str(latest["draw_date"].date()),
            "ball": n,
            "pre_draw_rank": rank,
            "model_score": info["final_score"],
            "frequency_score": info["frequency_score"],
            "momentum_score": info["momentum_score"],
            "gap_score": info["gap_score"],
            "last_20": info["last_20"],
            "last_50": info["last_50"],
            "current_gap": info["current_gap"],
            "actual_sum": actual_sum,
            "actual_ou": actual_ou,
            "ou_prediction_before_draw": pre_ou["prediction"],
            "ou_correct": pre_ou["prediction"] == actual_ou,
            "insight": insight,
        })

    pair_rows = []
    for a, b in itertools.combinations(latest_numbers, 2):
        key = f"{a}-{b}"
        info = pair_map.get(key)
        if info:
            pair_rows.append({
                "pair": key,
                "pre_draw_pair_rank": int(info["rank"]),
                "score": info["score"],
                "appeared": info["appeared"],
            })

    triplet_rows = []
    for t in itertools.combinations(latest_numbers, 3):
        key = "-".join(map(str, t))
        info = triplet_map.get(key)
        if info:
            triplet_rows.append({
                "triplet": key,
                "pre_draw_triplet_rank": int(info["rank"]),
                "score": info["score"],
                "appeared": info["appeared"],
            })

    return pd.DataFrame(ball_rows), pd.DataFrame(pair_rows), pd.DataFrame(triplet_rows)
