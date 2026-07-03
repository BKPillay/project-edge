from __future__ import annotations
import itertools, math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

NUMBERS = range(1, 37)
COMBINATIONS = list(itertools.combinations(NUMBERS, 5))

@dataclass
class EdgeConfig:
    long_term_weight: float = 0.35
    recent_20_weight: float = 0.20
    recent_50_weight: float = 0.15
    gap_weight: float = 0.10
    pair_weight: float = 0.10
    structure_weight: float = 0.10
    over_under_threshold: float = 92.5

def load_history(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    for col in ["n1","n2","n3","n4","n5"]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    return df.sort_values("draw_date").reset_index(drop=True)

def count_consecutive_pairs(numbers):
    return sum(1 for a, b in zip(numbers, numbers[1:]) if b == a + 1)

def add_draw_features(df):
    out = df.copy()
    cols = ["n1","n2","n3","n4","n5"]
    nums = out[cols]
    out["sum"] = nums.sum(axis=1)
    out["over_92_5"] = out["sum"] > 92.5
    out["odd_count"] = nums.apply(lambda r: sum(x % 2 for x in r), axis=1)
    out["low_count"] = nums.apply(lambda r: sum(x <= 18 for x in r), axis=1)
    out["range"] = nums.max(axis=1) - nums.min(axis=1)
    out["consecutive_pairs"] = nums.apply(lambda r: count_consecutive_pairs(sorted(r.tolist())), axis=1)
    repeats = [0]
    for i in range(1, len(out)):
        prev = set(out.loc[i-1, cols].tolist())
        cur = set(out.loc[i, cols].tolist())
        repeats.append(len(prev & cur))
    out["repeat_from_previous"] = repeats
    out["rolling_sum_5"] = out["sum"].rolling(5, min_periods=1).mean()
    out["rolling_sum_10"] = out["sum"].rolling(10, min_periods=1).mean()
    return out

def minmax(values):
    mn, mx = min(values.values()), max(values.values())
    if math.isclose(mx, mn):
        return {k: 0.5 for k in values}
    return {k: (v - mn) / (mx - mn) for k, v in values.items()}

def number_scores(df, cfg=EdgeConfig()):
    cols = ["n1","n2","n3","n4","n5"]
    draws = [set(row) for row in df[cols].values.tolist()]
    long_counts = {n: 0 for n in NUMBERS}
    for draw in draws:
        for n in draw:
            long_counts[n] += 1
    r20 = draws[-20:]
    r50 = draws[-50:]
    r20_counts = {n: sum(n in d for d in r20) for n in NUMBERS}
    r50_counts = {n: sum(n in d for d in r50) for n in NUMBERS}
    gaps = {}
    for n in NUMBERS:
        gap = 0
        for draw in reversed(draws):
            if n in draw:
                break
            gap += 1
        gaps[n] = min(gap, 30)
    long_s, r20_s, r50_s, gap_s = map(minmax, [long_counts, r20_counts, r50_counts, gaps])
    rows = []
    for n in NUMBERS:
        score = (
            cfg.long_term_weight * long_s[n]
            + cfg.recent_20_weight * r20_s[n]
            + cfg.recent_50_weight * r50_s[n]
            + cfg.gap_weight * gap_s[n]
        )
        rows.append({
            "number": n,
            "long_count": long_counts[n],
            "recent_20_count": r20_counts[n],
            "recent_50_count": r50_counts[n],
            "gap": gaps[n],
            "score": round(score * 100, 3)
        })
    return pd.DataFrame(rows).sort_values(["score","number"], ascending=[False, True]).reset_index(drop=True)

def pair_scores(df):
    cols = ["n1","n2","n3","n4","n5"]
    counts = {pair: 0 for pair in itertools.combinations(NUMBERS, 2)}
    for row in df[cols].values.tolist():
        for pair in itertools.combinations(sorted(row), 2):
            counts[pair] += 1
    return minmax(counts)

def structure_score(combo):
    s = sum(combo)
    odd = sum(n % 2 for n in combo)
    low = sum(n <= 18 for n in combo)
    consecutive = count_consecutive_pairs(list(combo))
    score = 0
    score += 0.35 if 65 <= s <= 120 else 0.10
    score += 0.25 if odd in (2,3) else 0.10
    score += 0.25 if low in (2,3) else 0.10
    score += 0.15 if consecutive <= 1 else 0.05
    return score

def combination_scores(df, cfg=EdgeConfig(), top_n=10):
    ns = number_scores(df, cfg)
    nscore = dict(zip(ns["number"], ns["score"]))
    ps = pair_scores(df)
    rows = []
    for combo in COMBINATIONS:
        avg_number = np.mean([nscore[n] for n in combo]) / 100
        avg_pair = np.mean([ps[tuple(sorted(p))] for p in itertools.combinations(combo, 2)])
        struct = structure_score(combo)
        score = (0.65 * avg_number + cfg.pair_weight * avg_pair + cfg.structure_weight * struct) * 100
        rows.append({
            "combination": "-".join(map(str, combo)),
            "sum": sum(combo),
            "odd_count": sum(n % 2 for n in combo),
            "low_count": sum(n <= 18 for n in combo),
            "score": round(score, 3)
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

def over_under_prediction(df, cfg=EdgeConfig()):
    feat = add_draw_features(df)
    if len(feat) == 0:
        return {"prediction": "Unknown", "confidence": 0}
    all_rate = float(feat["over_92_5"].mean())
    recent_rate = float(feat.tail(min(20, len(feat)))["over_92_5"].mean())
    rolling_sum = float(feat["sum"].tail(min(10, len(feat))).mean())
    prob_over = 0.55 * 0.50 + 0.25 * all_rate + 0.20 * recent_rate
    prob_over += 0.03 if rolling_sum > cfg.over_under_threshold else -0.03
    prob_over = max(0.05, min(0.95, prob_over))
    pred = "Over 92.5" if prob_over >= 0.5 else "Under 92.5"
    conf = prob_over if pred.startswith("Over") else 1 - prob_over
    return {
        "prediction": pred,
        "confidence": round(conf * 100, 2),
        "prob_over": round(prob_over * 100, 2),
        "prob_under": round((1 - prob_over) * 100, 2),
        "recent_avg_sum": round(rolling_sum, 2)
    }
