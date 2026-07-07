from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

NUMBER_COLS = ["n1", "n2", "n3", "n4", "n5"]
NUMBERS = range(1, 37)
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
FIBONACCI = {1, 2, 3, 5, 8, 13, 21, 34}


def load_history(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"draw_date", "n1", "n2", "n3", "n4", "n5"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "draw_number" not in df.columns:
        df.insert(0, "draw_number", range(1, len(df) + 1))

    df["draw_number"] = pd.to_numeric(df["draw_number"], errors="raise").astype(int)
    df["draw_date"] = pd.to_datetime(df["draw_date"])

    for c in NUMBER_COLS:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(int)

    df = df.sort_values(["draw_number", "draw_date"]).drop_duplicates("draw_number", keep="last")
    df = df.drop_duplicates("draw_date", keep="last").reset_index(drop=True)

    validate_history(df)
    return df


def validate_history(df: pd.DataFrame) -> None:
    bad_rows = []
    for idx, row in df.iterrows():
        values = [int(row[c]) for c in NUMBER_COLS]
        if len(set(values)) != 5 or any(v < 1 or v > 36 for v in values):
            bad_rows.append((idx + 2, values))

    if bad_rows:
        raise ValueError(f"Invalid Daily Lotto rows: {bad_rows[:10]}")

    if df["draw_number"].duplicated().any():
        raise ValueError("Duplicate draw_number values found.")

    if df["draw_date"].duplicated().any():
        raise ValueError("Duplicate draw_date values found.")


def draw_sets(df: pd.DataFrame):
    return [set(map(int, row)) for row in df[NUMBER_COLS].values.tolist()]


def minmax(values: Dict) -> Dict:
    if not values:
        return {}
    mn, mx = min(values.values()), max(values.values())
    if mx == mn:
        return {k: 0.5 for k in values}
    return {k: (v - mn) / (mx - mn) for k, v in values.items()}


def zscore_series(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([0] * len(series), index=series.index)
    return (series - series.mean()) / std


def count_consecutive_pairs(numbers) -> int:
    nums = sorted(map(int, numbers))
    return sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    nums = out[NUMBER_COLS]

    out["sum"] = nums.sum(axis=1)
    out["over_92_5"] = out["sum"] > 92.5
    out["odd_count"] = nums.apply(lambda r: sum(int(x) % 2 for x in r), axis=1)
    out["even_count"] = 5 - out["odd_count"]
    out["low_count"] = nums.apply(lambda r: sum(int(x) <= 18 for x in r), axis=1)
    out["high_count"] = 5 - out["low_count"]
    out["prime_count"] = nums.apply(lambda r: sum(int(x) in PRIMES for x in r), axis=1)
    out["fibonacci_count"] = nums.apply(lambda r: sum(int(x) in FIBONACCI for x in r), axis=1)
    out["consecutive_pairs"] = nums.apply(lambda r: count_consecutive_pairs(r.tolist()), axis=1)
    out["range"] = nums.max(axis=1) - nums.min(axis=1)

    repeats = [0]
    for i in range(1, len(out)):
        prev = set(out.loc[i - 1, NUMBER_COLS].tolist())
        cur = set(out.loc[i, NUMBER_COLS].tolist())
        repeats.append(len(prev & cur))
    out["repeat_from_previous"] = repeats

    out["decade_01_09"] = nums.apply(lambda r: sum(1 <= int(x) <= 9 for x in r), axis=1)
    out["decade_10_19"] = nums.apply(lambda r: sum(10 <= int(x) <= 19 for x in r), axis=1)
    out["decade_20_29"] = nums.apply(lambda r: sum(20 <= int(x) <= 29 for x in r), axis=1)
    out["decade_30_36"] = nums.apply(lambda r: sum(30 <= int(x) <= 36 for x in r), axis=1)

    for w in [20, 50, 100, 250, 500]:
        out[f"rolling_sum_{w}"] = out["sum"].rolling(w, min_periods=1).mean()
        out[f"rolling_over_rate_{w}"] = out["over_92_5"].rolling(w, min_periods=1).mean()

    out["structure"] = (
        out["odd_count"].astype(str) + "O/" + out["even_count"].astype(str) + "E · "
        + out["low_count"].astype(str) + "L/" + out["high_count"].astype(str) + "H"
    )

    return out
