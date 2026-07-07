
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


def repeat_after_rates(df: pd.DataFrame) -> pd.DataFrame:
    values = df[NUMBER_COLS].to_numpy(dtype=int)
    draw_sets_array = [set(map(int, row)) for row in values]
    baseline = 5 / 36

    rows = []
    for n in NUMBERS:
        appeared_positions = [i for i, s in enumerate(draw_sets_array) if n in s]

        def rate_after(k: int):
            eligible = [i for i in appeared_positions if i + k < len(draw_sets_array)]
            if not eligible:
                return None, 0, 0
            hits = sum(1 for i in eligible if n in draw_sets_array[i + k])
            return hits / len(eligible), hits, len(eligible)

        r1, h1, e1 = rate_after(1)
        r2, h2, e2 = rate_after(2)
        r3, h3, e3 = rate_after(3)

        valid_parts = []
        if r1 is not None:
            valid_parts.append((0.55, r1))
        if r2 is not None:
            valid_parts.append((0.30, r2))
        if r3 is not None:
            valid_parts.append((0.15, r3))

        if valid_parts:
            total_w = sum(w for w, _ in valid_parts)
            repeat_tendency = sum(w * r for w, r in valid_parts) / total_w
        else:
            repeat_tendency = baseline

        repeat_index = repeat_tendency / baseline if baseline else 1

        rows.append({
            "number": n,
            "repeat_after_1_rate": round((r1 or 0) * 100, 2),
            "repeat_after_2_rate": round((r2 or 0) * 100, 2),
            "repeat_after_3_rate": round((r3 or 0) * 100, 2),
            "repeat_after_1_hits": h1,
            "repeat_after_1_tests": e1,
            "repeat_after_2_hits": h2,
            "repeat_after_2_tests": e2,
            "repeat_after_3_hits": h3,
            "repeat_after_3_tests": e3,
            "repeat_tendency": round(repeat_tendency * 100, 2),
            "repeat_index": round(repeat_index, 3),
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
    repeat_stats_df = repeat_after_rates(df)
    repeat_stats = repeat_stats_df.set_index("number").to_dict(orient="index")

    long_s = minmax(long_counts)
    r20_s = minmax(r20)
    r50_s = minmax(r50)
    r100_s = minmax(r100)
    r250_s = minmax(r250)
    r500_s = minmax(r500)
    gap_s = minmax(gaps)

    latest_draw = set(map(int, df.iloc[-1][NUMBER_COLS].tolist())) if len(df) >= 1 else set()
    previous_draw = set(map(int, df.iloc[-2][NUMBER_COLS].tolist())) if len(df) >= 2 else set()
    third_last_draw = set(map(int, df.iloc[-3][NUMBER_COLS].tolist())) if len(df) >= 3 else set()

    rows = []
    for n in NUMBERS:
        momentum_score = 0.45 * r20_s[n] + 0.30 * r50_s[n] + 0.25 * r100_s[n]
        frequency_score = 0.45 * long_s[n] + 0.25 * r250_s[n] + 0.30 * r500_s[n]
        gap_score = gap_s[n]

        raw_final = 0.45 * frequency_score + 0.35 * momentum_score + 0.20 * gap_score

        current_gap = gaps[n]
        avg_gap = gap_stats[n]["avg_gap"]
        gap_ratio = round(current_gap / avg_gap, 3) if avg_gap else None

        repeat_info = repeat_stats[n]
        repeat_index = float(repeat_info["repeat_index"])
        repeat_score = max(0, min(100, 50 + ((repeat_index - 1) * 100)))

        if n in latest_draw:
            base_penalty = 22.0
            repeat_penalty = base_penalty * max(0.35, min(1.50, 1.0 / max(repeat_index, 0.25)))
        elif n in previous_draw:
            base_penalty = 12.0
            repeat_penalty = base_penalty * max(0.35, min(1.35, 1.0 / max(repeat_index, 0.25)))
        elif n in third_last_draw:
            base_penalty = 6.0
            repeat_penalty = base_penalty * max(0.35, min(1.20, 1.0 / max(repeat_index, 0.25)))
        else:
            repeat_penalty = 0.0

        final_score = round(raw_final * 100, 2)
        repeat_penalty = round(float(repeat_penalty), 2)
        selection_score = round(final_score - repeat_penalty, 2)

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
            "final_score": final_score,
            "frequency_score": round(frequency_score * 100, 2),
            "momentum_score": round(momentum_score * 100, 2),
            "gap_score": round(gap_score * 100, 2),
            "repeat_score": round(repeat_score, 2),
            "repeat_index": repeat_info["repeat_index"],
            "repeat_after_1_rate": repeat_info["repeat_after_1_rate"],
            "repeat_after_2_rate": repeat_info["repeat_after_2_rate"],
            "repeat_after_3_rate": repeat_info["repeat_after_3_rate"],
            "repeat_penalty": repeat_penalty,
            "selection_score": selection_score,
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

    raw_order = pd.DataFrame(rows).sort_values(["final_score", "number"], ascending=[False, True]).reset_index(drop=True)
    raw_order["rank"] = raw_order.index + 1

    selection_order = raw_order.sort_values(["selection_score", "number"], ascending=[False, True]).reset_index(drop=True)
    selection_order["selection_rank"] = selection_order.index + 1

    out = raw_order.merge(selection_order[["number", "selection_rank"]], on="number", how="left")
    out = out.sort_values(["selection_rank", "number"], ascending=[True, True]).reset_index(drop=True)
    return out


def number_model_components(df: pd.DataFrame) -> pd.DataFrame:
    return score_numbers(df)
