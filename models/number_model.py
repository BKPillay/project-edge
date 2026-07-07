from __future__ import annotations

import numpy as np
import pandas as pd

from models.common import NUMBERS, NUMBER_COLS, draw_sets, minmax


def count_numbers(draws):
    counts = {n: 0 for n in NUMBERS}
    for draw in draws:
        for n in draw:
            counts[n] += 1
    return counts


def gap_counts(draws):
    gaps = {}
    for n in NUMBERS:
        gap = 0
        for draw in reversed(draws):
            if n in draw:
                break
            gap += 1
        gaps[n] = min(gap, 500)
    return gaps


def gap_history_fast(df: pd.DataFrame) -> pd.DataFrame:
    values = df[NUMBER_COLS].to_numpy(dtype=int)
    rows = []

    for n in NUMBERS:
        positions = np.where((values == n).any(axis=1))[0]
        if len(positions) >= 2:
            gaps = np.diff(positions)
            avg_gap = round(float(np.mean(gaps)), 3)
            gap_std = round(float(np.std(gaps)), 3)
            min_gap = int(np.min(gaps))
            max_gap = int(np.max(gaps))
        else:
            avg_gap = None
            gap_std = None
            min_gap = None
            max_gap = None

        rows.append({
            "number": n,
            "avg_gap": avg_gap,
            "gap_std": gap_std,
            "min_gap": min_gap,
            "max_gap": max_gap,
            "appearances": int(len(positions)),
        })

    return pd.DataFrame(rows)


def score_numbers(df: pd.DataFrame) -> pd.DataFrame:
    draws = draw_sets(df)

    long_counts = count_numbers(draws)
    r20 = count_numbers(draws[-20:])
    r50 = count_numbers(draws[-50:])
    r100 = count_numbers(draws[-100:])
    r250 = count_numbers(draws[-250:])
    r500 = count_numbers(draws[-500:])
    gaps = gap_counts(draws)
    gap_stats = gap_history_fast(df).set_index("number").to_dict(orient="index")

    long_s = minmax(long_counts)
    r20_s = minmax(r20)
    r50_s = minmax(r50)
    r100_s = minmax(r100)
    r250_s = minmax(r250)
    r500_s = minmax(r500)
    gap_s = minmax(gaps)

    rows = []
    for n in NUMBERS:
        momentum_score = 0.45 * r20_s[n] + 0.30 * r50_s[n] + 0.25 * r100_s[n]
        frequency_score = 0.45 * long_s[n] + 0.25 * r250_s[n] + 0.30 * r500_s[n]
        gap_score = gap_s[n]

        final_score = (
            0.45 * frequency_score
            + 0.35 * momentum_score
            + 0.20 * gap_score
        )

        current_gap = gaps[n]
        avg_gap = gap_stats[n]["avg_gap"]
        gap_ratio = round(current_gap / avg_gap, 3) if avg_gap else None

        if r20[n] >= 4:
            bucket = "hot"
        elif current_gap >= 20:
            bucket = "cold/overdue"
        elif r50[n] >= 8:
            bucket = "steady"
        else:
            bucket = "neutral"

        rows.append({
            "rank": None,
            "number": n,
            "final_score": round(final_score * 100, 2),
            "frequency_score": round(frequency_score * 100, 2),
            "momentum_score": round(momentum_score * 100, 2),
            "gap_score": round(gap_score * 100, 2),
            "long_count": long_counts[n],
            "last_20": r20[n],
            "last_50": r50[n],
            "last_100": r100[n],
            "last_250": r250[n],
            "last_500": r500[n],
            "current_gap": current_gap,
            "avg_gap": avg_gap,
            "gap_ratio": gap_ratio,
            "bucket": bucket,
        })

    out = pd.DataFrame(rows).sort_values(["final_score", "number"], ascending=[False, True]).reset_index(drop=True)
    out["rank"] = out.index + 1

    # EDGE AI v2.4: post-hit cooling layer.
    # final_score remains the raw model score.
    # selection_score is the next-draw recommendation score.
    recent_draws = []
    if len(df) >= 1:
        recent_draws.append(set(map(int, df.iloc[-1][NUMBER_COLS].tolist())))
    if len(df) >= 2:
        recent_draws.append(set(map(int, df.iloc[-2][NUMBER_COLS].tolist())))
    if len(df) >= 3:
        recent_draws.append(set(map(int, df.iloc[-3][NUMBER_COLS].tolist())))

    def recency_penalty(number):
        number = int(number)
        if len(recent_draws) >= 1 and number in recent_draws[0]:
            return 25.0
        if len(recent_draws) >= 2 and number in recent_draws[1]:
            return 15.0
        if len(recent_draws) >= 3 and number in recent_draws[2]:
            return 8.0
        return 0.0

    out["recency_penalty"] = out["number"].apply(recency_penalty)
    out["selection_score"] = (out["final_score"] * (1 - out["recency_penalty"] / 100)).round(2)

    selection_order = out.sort_values(["selection_score", "number"], ascending=[False, True]).reset_index(drop=True)
    selection_order["selection_rank"] = selection_order.index + 1
    out = out.merge(selection_order[["number", "selection_rank"]], on="number", how="left")

    # Keep raw rank visible, but order the output by today's selection rank.
    out = out.sort_values(["selection_rank", "number"], ascending=[True, True]).reset_index(drop=True)
    return out


def number_model_components(df: pd.DataFrame) -> pd.DataFrame:
    return score_numbers(df)
