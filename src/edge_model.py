# EDGE AI v8.1 MODEL FIX
from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

NUMBERS = range(1, 37)
NUMBER_COLS = ["n1", "n2", "n3", "n4", "n5"]
COMBINATIONS = list(itertools.combinations(NUMBERS, 5))
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}


def load_history(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    for col in NUMBER_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    df = df.sort_values("draw_date").drop_duplicates(subset=["draw_date"], keep="last").reset_index(drop=True)
    validate_history(df)
    return df


def validate_history(df: pd.DataFrame) -> None:
    for idx, row in df.iterrows():
        vals = [int(row[c]) for c in NUMBER_COLS]
        if len(set(vals)) != 5 or any(v < 1 or v > 36 for v in vals):
            raise ValueError(f"Invalid Daily Lotto numbers on row {idx + 2}: {vals}")


def minmax(values: Dict) -> Dict:
    if not values:
        return {}
    mn = min(values.values())
    mx = max(values.values())
    if math.isclose(mx, mn):
        return {k: 0.5 for k in values}
    return {k: (v - mn) / (mx - mn) for k, v in values.items()}


def count_consecutive_pairs(numbers) -> int:
    numbers = sorted([int(n) for n in numbers])
    return sum(1 for a, b in zip(numbers, numbers[1:]) if b == a + 1)


def add_draw_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    nums = out[NUMBER_COLS]
    out["sum"] = nums.sum(axis=1)
    out["over_92_5"] = out["sum"] > 92.5
    out["odd_count"] = nums.apply(lambda r: sum(int(x) % 2 for x in r), axis=1)
    out["even_count"] = 5 - out["odd_count"]
    out["low_count"] = nums.apply(lambda r: sum(int(x) <= 18 for x in r), axis=1)
    out["high_count"] = 5 - out["low_count"]
    out["range"] = nums.max(axis=1) - nums.min(axis=1)
    out["consecutive_pairs"] = nums.apply(lambda r: count_consecutive_pairs(r.tolist()), axis=1)
    out["prime_count"] = nums.apply(lambda r: sum(int(x) in PRIMES for x in r), axis=1)
    out["structure"] = out["odd_count"].astype(str) + " Odd / " + out["even_count"].astype(str) + " Even · " + out["low_count"].astype(str) + " Low / " + out["high_count"].astype(str) + " High"
    repeats = [0]
    for i in range(1, len(out)):
        prev = set(out.loc[i - 1, NUMBER_COLS].tolist())
        cur = set(out.loc[i, NUMBER_COLS].tolist())
        repeats.append(len(prev & cur))
    out["repeat_from_previous"] = repeats
    for w in [5, 10, 20, 50, 100]:
        out[f"rolling_sum_{w}"] = out["sum"].rolling(w, min_periods=1).mean()
        out[f"rolling_over_rate_{w}"] = out["over_92_5"].rolling(w, min_periods=1).mean()
    return out


def draw_sets(df: pd.DataFrame):
    return [set(map(int, row)) for row in df[NUMBER_COLS].values.tolist()]


def count_numbers(draws) -> Dict[int, int]:
    counts = {n: 0 for n in NUMBERS}
    for draw in draws:
        for n in draw:
            counts[n] += 1
    return counts


def gap_counts(draws) -> Dict[int, int]:
    gaps = {}
    for n in NUMBERS:
        gap = 0
        for draw in reversed(draws):
            if n in draw:
                break
            gap += 1
        gaps[n] = min(gap, 60)
    return gaps


def number_reason(long_count: int, r20: int, r50: int, gap: int) -> str:
    if r20 >= 4:
        return "Strong recent momentum"
    if r50 >= 9:
        return "Strong 50-draw form"
    if long_count >= 390:
        return "Strong long-term frequency"
    if gap >= 15:
        return "Overdue factor"
    return "Balanced profile"


def number_scores(df: pd.DataFrame) -> pd.DataFrame:
    draws = draw_sets(df)
    long_counts = count_numbers(draws)
    r20_counts = count_numbers(draws[-20:])
    r50_counts = count_numbers(draws[-50:])
    r100_counts = count_numbers(draws[-100:])
    gaps = gap_counts(draws)
    long_s = minmax(long_counts); r20_s = minmax(r20_counts); r50_s = minmax(r50_counts); r100_s = minmax(r100_counts); gap_s = minmax(gaps)
    rows = []
    for n in NUMBERS:
        score = 0.35 * long_s[n] + 0.20 * r20_s[n] + 0.18 * r50_s[n] + 0.17 * r100_s[n] + 0.10 * gap_s[n]
        rows.append({"Rank": None, "Number": n, "Score": round(score * 100, 2), "Long Count": long_counts[n], "Last 20": r20_counts[n], "Last 50": r50_counts[n], "Last 100": r100_counts[n], "Gap": gaps[n], "Reason": number_reason(long_counts[n], r20_counts[n], r50_counts[n], gaps[n])})
    out = pd.DataFrame(rows).sort_values(["Score", "Number"], ascending=[False, True]).reset_index(drop=True)
    out["Rank"] = out.index + 1
    return out


def pair_counts(df: pd.DataFrame) -> Dict[Tuple[int, int], int]:
    counts = {pair: 0 for pair in itertools.combinations(NUMBERS, 2)}
    for row in df[NUMBER_COLS].values.tolist():
        for pair in itertools.combinations(sorted(map(int, row)), 2):
            counts[pair] += 1
    return counts


def pair_last_seen(df: pd.DataFrame, pair: Tuple[int, int]) -> int:
    for gap, (_, row) in enumerate(df.iloc[::-1].iterrows()):
        vals = set(row[NUMBER_COLS].tolist())
        if pair[0] in vals and pair[1] in vals:
            return gap
    return len(df)


def pair_predictions(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    pc = pair_counts(df)
    norm = minmax(pc)
    percentile_90 = float(np.percentile(list(pc.values()), 90))
    rows = []
    for pair, count in pc.items():
        gap = pair_last_seen(df, pair)
        gap_score = min(gap, 60) / 60
        score = 0.80 * norm[pair] + 0.20 * gap_score
        rows.append({"Rank": None, "Pair": f"{pair[0]}-{pair[1]}", "Score": round(score * 100, 2), "Appeared": count, "Last Seen": "latest draw" if gap == 0 else f"{gap} draws ago", "Reason": "Frequent pair" if count >= percentile_90 else "Pair + gap balance"})
    out = pd.DataFrame(rows).sort_values(["Score", "Pair"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    out["Rank"] = out.index + 1
    return out


def triplet_counts(df: pd.DataFrame) -> Dict[Tuple[int, int, int], int]:
    counts = {}
    for row in df[NUMBER_COLS].values.tolist():
        for triplet in itertools.combinations(sorted(map(int, row)), 3):
            counts[triplet] = counts.get(triplet, 0) + 1
    return counts


def triplet_last_seen(df: pd.DataFrame, triplet: Tuple[int, int, int]) -> int:
    for gap, (_, row) in enumerate(df.iloc[::-1].iterrows()):
        vals = set(row[NUMBER_COLS].tolist())
        if all(n in vals for n in triplet):
            return gap
    return len(df)


def triplet_predictions(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    tc = triplet_counts(df)
    if not tc:
        return pd.DataFrame(columns=["Rank", "Triplet", "Score", "Appeared", "Last Seen", "Reason"])
    norm = minmax(tc)
    percentile_95 = float(np.percentile(list(tc.values()), 95))
    rows = []
    for triplet, count in tc.items():
        gap = triplet_last_seen(df, triplet)
        gap_score = min(gap, 120) / 120
        score = 0.85 * norm[triplet] + 0.15 * gap_score
        rows.append({"Rank": None, "Triplet": "-".join(map(str, triplet)), "Score": round(score * 100, 2), "Appeared": count, "Last Seen": "latest draw" if gap == 0 else f"{gap} draws ago", "Reason": "Recurring triplet" if count >= percentile_95 else "Triplet + gap balance"})
    out = pd.DataFrame(rows).sort_values(["Score", "Triplet"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    out["Rank"] = out.index + 1
    return out


def structure_score(combo) -> float:
    s = sum(combo); odd = sum(n % 2 for n in combo); low = sum(n <= 18 for n in combo); consecutive = count_consecutive_pairs(combo); prime = sum(n in PRIMES for n in combo)
    score = 0.0
    score += 0.30 if 65 <= s <= 120 else 0.10
    score += 0.25 if odd in (2, 3) else 0.08
    score += 0.25 if low in (2, 3) else 0.08
    score += 0.12 if consecutive <= 1 else 0.04
    score += 0.08 if prime in (1, 2, 3) else 0.03
    return score


def combination_scores(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    ns = number_scores(df)
    nscore = dict(zip(ns["Number"], ns["Score"]))
    pc = pair_counts(df)
    pair_norm = minmax(pc)
    rows = []
    for combo in COMBINATIONS:
        avg_number = float(np.mean([nscore[n] for n in combo])) / 100
        avg_pair = float(np.mean([pair_norm[tuple(sorted(p))] for p in itertools.combinations(combo, 2)]))
        edge_score = (0.68 * avg_number + 0.17 * avg_pair + 0.15 * structure_score(combo)) * 100
        odd = sum(n % 2 for n in combo); low = sum(n <= 18 for n in combo)
        rows.append({"Rank": None, "Combination": "-".join(map(str, combo)), "Score": round(edge_score, 2), "Sum": sum(combo), "Odd": odd, "Even": 5 - odd, "Low": low, "High": 5 - low, "Prime": sum(n in PRIMES for n in combo)})
    out = pd.DataFrame(rows).sort_values(["Score", "Combination"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    out["Rank"] = out.index + 1
    return out


def over_under_prediction(df: pd.DataFrame, threshold: float = 92.5) -> dict:
    feat = add_draw_features(df)
    all_rate = float((feat["sum"] > threshold).mean())
    r20_rate = float((feat.tail(min(20, len(feat)))["sum"] > threshold).mean())
    r50_rate = float((feat.tail(min(50, len(feat)))["sum"] > threshold).mean())
    r100_rate = float((feat.tail(min(100, len(feat)))["sum"] > threshold).mean())
    avg10 = float(feat.tail(min(10, len(feat)))["sum"].mean())
    prob_over = 0.40 * 0.50 + 0.20 * all_rate + 0.15 * r20_rate + 0.15 * r50_rate + 0.10 * r100_rate
    prob_over += 0.025 if avg10 > threshold else -0.025
    prob_over = max(0.05, min(0.95, prob_over))
    prediction = "Over 92.5" if prob_over >= 0.5 else "Under 92.5"
    confidence = prob_over if prediction.startswith("Over") else 1 - prob_over
    return {"prediction": prediction, "confidence": round(confidence * 100, 2), "prob_over": round(prob_over * 100, 2), "prob_under": round((1 - prob_over) * 100, 2), "all_time_over_rate": round(all_rate * 100, 2), "recent_20_over_rate": round(r20_rate * 100, 2), "recent_50_over_rate": round(r50_rate * 100, 2), "recent_100_over_rate": round(r100_rate * 100, 2), "recent_avg_sum_10": round(avg10, 2)}


def latest_draw_review(df: pd.DataFrame) -> dict:
    if len(df) < 2:
        return {"available": False, "reason": "Need at least two draws."}
    pre = df.iloc[:-1].copy()
    latest = df.iloc[-1].copy()
    latest_numbers = sorted([int(latest[c]) for c in NUMBER_COLS])
    actual_set = set(latest_numbers)
    ns = number_scores(pre)
    ns_map = ns.set_index("Number").to_dict(orient="index")
    pairs_full = pair_predictions(pre, top_n=666)
    pair_rank = {row["Pair"]: row for _, row in pairs_full.iterrows()}
    triplets_full = triplet_predictions(pre, top_n=10000)
    triplet_rank = {row["Triplet"]: row for _, row in triplets_full.iterrows()}
    combos = combination_scores(pre, top_n=10)
    top3 = set(ns.head(3)["Number"].astype(int).tolist())
    ou_pred = over_under_prediction(pre)
    actual_sum = sum(latest_numbers)
    actual_ou = "Over 92.5" if actual_sum > 92.5 else "Under 92.5"
    ball_rows = []
    for n in latest_numbers:
        info = ns_map[n]
        rank = int(info["Rank"]); score = float(info["Score"])
        insight = "Elite pre-draw pick" if rank <= 3 else "Strong pre-draw pick" if rank <= 10 else "Mid-table model pick" if rank <= 20 else "Model underweighted this ball"
        ball_rows.append({"Ball": n, "Pre-draw Rank": rank, "Model Score": score, "Long Count": int(info["Long Count"]), "Last 20": int(info["Last 20"]), "Last 50": int(info["Last 50"]), "Gap Before Draw": int(info["Gap"]), "Insight": insight})
    pair_rows = []
    for a, b in itertools.combinations(latest_numbers, 2):
        key = f"{a}-{b}"; row = pair_rank.get(key)
        if row is not None:
            pair_rows.append({"Pair": key, "Pre-draw Pair Rank": int(row["Rank"]), "Score": float(row["Score"]), "Appeared Before": int(row["Appeared"]), "Insight": "High-ranked pair hit" if int(row["Rank"]) <= 20 else "Lower-ranked pair hit"})
    triplet_rows = []
    for t in itertools.combinations(latest_numbers, 3):
        key = "-".join(map(str, t)); row = triplet_rank.get(key)
        if row is not None:
            triplet_rows.append({"Triplet": key, "Pre-draw Triplet Rank": int(row["Rank"]), "Score": float(row["Score"]), "Appeared Before": int(row["Appeared"]), "Insight": "Recurring triplet hit" if int(row["Rank"]) <= 50 else "Rare/lower-ranked triplet hit"})
    best_combo_matches = -1; best_combo = None
    for _, row in combos.iterrows():
        combo_set = set(map(int, row["Combination"].split("-")))
        matches = len(combo_set & actual_set)
        if matches > best_combo_matches:
            best_combo_matches = matches; best_combo = row["Combination"]
    return {"available": True, "draw_date": str(latest["draw_date"].date()), "latest_numbers": latest_numbers, "actual_sum": actual_sum, "actual_ou": actual_ou, "ou_prediction": ou_pred["prediction"], "ou_correct": ou_pred["prediction"] == actual_ou, "ou_confidence": ou_pred["confidence"], "top3_hits": len(actual_set & top3), "best_top10_combo": best_combo, "best_top10_combo_matches": best_combo_matches, "ball_review": pd.DataFrame(ball_rows), "pair_review": pd.DataFrame(pair_rows).sort_values("Pre-draw Pair Rank").reset_index(drop=True) if pair_rows else pd.DataFrame(), "triplet_review": pd.DataFrame(triplet_rows).sort_values("Pre-draw Triplet Rank").reset_index(drop=True) if triplet_rows else pd.DataFrame()}


def walk_forward_over_under_backtest(df: pd.DataFrame, min_train: int = 200, step: int = 1) -> pd.DataFrame:
    rows = []
    for i in range(min_train, len(df), step):
        train = df.iloc[:i].copy(); pred = over_under_prediction(train); actual_sum = int(df.iloc[i][NUMBER_COLS].sum()); actual = "Over 92.5" if actual_sum > 92.5 else "Under 92.5"
        rows.append({"draw_date": df.iloc[i]["draw_date"], "prediction": pred["prediction"], "confidence": pred["confidence"], "actual_sum": actual_sum, "actual": actual, "correct": pred["prediction"] == actual})
    return pd.DataFrame(rows)


def simple_top3_backtest(df: pd.DataFrame, min_train: int = 200, step: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(min_train, len(df), step):
        train = df.iloc[:i].copy(); actual = set(map(int, df.iloc[i][NUMBER_COLS].tolist())); top3 = number_scores(train).head(3)["Number"].astype(int).tolist()
        rows.append({"draw_date": df.iloc[i]["draw_date"], "top3": "-".join(map(str, top3)), "matches": len(actual & set(top3))})
    return pd.DataFrame(rows)


def walk_forward_over_under_backtest_fast(df: pd.DataFrame, min_train: int = 200, step: int = 1) -> pd.DataFrame:
    return walk_forward_over_under_backtest(df, min_train=min_train, step=step)


def simple_top3_backtest_fast(df: pd.DataFrame, min_train: int = 200, step: int = 5) -> pd.DataFrame:
    return simple_top3_backtest(df, min_train=min_train, step=step)


def performance_summary(df: pd.DataFrame) -> dict:
    ou = walk_forward_over_under_backtest(df); t3 = simple_top3_backtest(df)
    return {"over_under_accuracy": round(float(ou["correct"].mean()) * 100, 2) if len(ou) else None, "over_under_tests": int(len(ou)), "top3_avg_matches": round(float(t3["matches"].mean()), 3) if len(t3) else None, "top3_2plus_hit_rate": round(float((t3["matches"] >= 2).mean()) * 100, 2) if len(t3) else None, "top3_tests": int(len(t3))}
