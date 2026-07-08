from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd

from models.common import NUMBER_COLS, NUMBERS
from models.number_model import score_numbers
from models.pair_model import score_pairs
from models.triplet_model import score_triplets
from models.combination_model import score_combinations


def _combo_from_row(row) -> tuple[int, ...]:
    return tuple(sorted(int(row[c]) for c in NUMBER_COLS))


def _history_combos(df: pd.DataFrame) -> list[tuple[int, ...]]:
    return [_combo_from_row(row) for _, row in df.iterrows()]


def _matches(predicted: tuple[int, ...], actual: tuple[int, ...]) -> int:
    return len(set(predicted) & set(actual))


def _score_row(predicted: tuple[int, ...], actual: tuple[int, ...]) -> dict:
    m = _matches(predicted, actual)
    return {
        "matches": m,
        "hit_2": int(m >= 2),
        "hit_3": int(m >= 3),
        "hit_4": int(m >= 4),
        "hit_5": int(m == 5),
    }


def _numbers_from_combo_string(value: str) -> tuple[int, ...]:
    return tuple(sorted(int(x) for x in str(value).split("-") if str(x).strip()))


def edge_ai_strategy(train: pd.DataFrame) -> tuple[int, ...]:
    """Uses the actual EDGE ensemble path: numbers -> pairs -> triplets -> combinations."""
    numbers = score_numbers(train)
    pairs = score_pairs(train, 100)
    triplets = score_triplets(train, 100)
    combos = score_combinations(numbers, pairs, triplets, top_n=1)

    if combos.empty:
        return hot_numbers_strategy(train)

    return _numbers_from_combo_string(combos.iloc[0]["combination"])


def random_strategy(train: pd.DataFrame, seed: int) -> tuple[int, ...]:
    rng = np.random.default_rng(seed)
    return tuple(sorted(int(x) for x in rng.choice(list(NUMBERS), 5, replace=False)))


def hot_numbers_strategy(train: pd.DataFrame) -> tuple[int, ...]:
    counts = Counter()
    for combo in _history_combos(train):
        counts.update(combo)
    return tuple(sorted(n for n, _ in counts.most_common(5)))


def cold_numbers_strategy(train: pd.DataFrame) -> tuple[int, ...]:
    counts = Counter()
    for combo in _history_combos(train):
        counts.update(combo)
    ranked = sorted(NUMBERS, key=lambda n: (counts[n], n))
    return tuple(sorted(ranked[:5]))


def overdue_strategy(train: pd.DataFrame) -> tuple[int, ...]:
    combos = _history_combos(train)
    last_seen = {n: -1 for n in NUMBERS}
    for idx, combo in enumerate(combos):
        for n in combo:
            last_seen[n] = idx
    current = len(combos)
    ranked = sorted(NUMBERS, key=lambda n: (current - last_seen[n], -n), reverse=True)
    return tuple(sorted(ranked[:5]))


def recent_repeat_strategy(train: pd.DataFrame, lookback: int = 3) -> tuple[int, ...]:
    counts = Counter()
    for combo in _history_combos(train)[-lookback:]:
        counts.update(combo)
    ranked = sorted(NUMBERS, key=lambda n: (counts[n], -n), reverse=True)
    return tuple(sorted(ranked[:5]))


def pair_frequency_strategy(train: pd.DataFrame) -> tuple[int, ...]:
    pair_counts = Counter()
    for combo in _history_combos(train):
        for pair in combinations(combo, 2):
            pair_counts[pair] += 1

    number_scores = defaultdict(int)
    for pair, count in pair_counts.items():
        for n in pair:
            number_scores[n] += count

    ranked = sorted(NUMBERS, key=lambda n: (number_scores[n], -n), reverse=True)
    return tuple(sorted(ranked[:5]))


def triplet_frequency_strategy(train: pd.DataFrame) -> tuple[int, ...]:
    triplet_counts = Counter()
    for combo in _history_combos(train):
        for triplet in combinations(combo, 3):
            triplet_counts[triplet] += 1

    number_scores = defaultdict(int)
    for triplet, count in triplet_counts.items():
        for n in triplet:
            number_scores[n] += count

    ranked = sorted(NUMBERS, key=lambda n: (number_scores[n], -n), reverse=True)
    return tuple(sorted(ranked[:5]))


def balanced_hot_strategy(train: pd.DataFrame) -> tuple[int, ...]:
    counts = Counter()
    for combo in _history_combos(train):
        counts.update(combo)
    odds = sorted([n for n in NUMBERS if n % 2 == 1], key=lambda n: (counts[n], -n), reverse=True)
    evens = sorted([n for n in NUMBERS if n % 2 == 0], key=lambda n: (counts[n], -n), reverse=True)
    return tuple(sorted(odds[:3] + evens[:2]))


def run_strategy_benchmark(
    df: pd.DataFrame,
    min_train: int = 300,
    step: int = 20,
    include_edge_ai: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Sampled walk-forward benchmark.

    Each test trains only on draws before the target draw. The `step` parameter keeps
    the default run fast enough for regular model updates. Use step=1 for a heavier
    full audit.
    """
    if len(df) <= min_train:
        raise ValueError(f"Need more than {min_train} draws to benchmark. Found {len(df)}.")

    strategies: list[tuple[str, Callable]] = [
        ("Random", None),
        ("Hot Numbers", hot_numbers_strategy),
        ("Cold Numbers", cold_numbers_strategy),
        ("Overdue", overdue_strategy),
        ("Recent Repeat", recent_repeat_strategy),
        ("Pair Frequency", pair_frequency_strategy),
        ("Triplet Frequency", triplet_frequency_strategy),
        ("Balanced Hot", balanced_hot_strategy),
    ]
    if include_edge_ai:
        strategies.insert(0, ("EDGE AI", edge_ai_strategy))

    rows = []
    test_indices = list(range(min_train, len(df), step))

    for test_no, idx in enumerate(test_indices, start=1):
        train = df.iloc[:idx].copy()
        actual = _combo_from_row(df.iloc[idx])
        draw_date = df.iloc[idx]["draw_date"]
        draw_number = int(df.iloc[idx]["draw_number"])

        for strategy_name, func in strategies:
            if strategy_name == "Random":
                predicted = random_strategy(train, seed=draw_number)
            else:
                predicted = func(train)

            score = _score_row(predicted, actual)
            rows.append({
                "test_no": test_no,
                "draw_number": draw_number,
                "draw_date": draw_date,
                "strategy": strategy_name,
                "prediction": "-".join(map(str, predicted)),
                "actual": "-".join(map(str, actual)),
                **score,
            })

    results = pd.DataFrame(rows)
    summary = (
        results.groupby("strategy")
        .agg(
            tests=("draw_number", "count"),
            avg_matches=("matches", "mean"),
            max_matches=("matches", "max"),
            hit_2_rate=("hit_2", "mean"),
            hit_3_rate=("hit_3", "mean"),
            hit_4_rate=("hit_4", "mean"),
            hit_5_rate=("hit_5", "mean"),
            total_2_plus=("hit_2", "sum"),
            total_3_plus=("hit_3", "sum"),
            total_4_plus=("hit_4", "sum"),
            jackpot_hits=("hit_5", "sum"),
        )
        .reset_index()
    )

    for col in ["avg_matches", "hit_2_rate", "hit_3_rate", "hit_4_rate", "hit_5_rate"]:
        summary[col] = summary[col].round(4)

    summary = summary.sort_values(
        ["hit_5_rate", "hit_4_rate", "hit_3_rate", "avg_matches", "hit_2_rate"],
        ascending=False,
    ).reset_index(drop=True)
    summary.insert(0, "rank", range(1, len(summary) + 1))

    metadata = {
        "purpose": "Strategy Benchmark Suite: compare EDGE AI against simple baseline strategies.",
        "method": "Sampled walk-forward validation. Every prediction uses only prior draws.",
        "min_train": int(min_train),
        "step": int(step),
        "tests_per_strategy": int(len(test_indices)),
        "include_edge_ai": bool(include_edge_ai),
        "warning": "Lottery outcomes are random. Treat this as model validation, not guaranteed prediction power.",
    }
    if len(summary):
        metadata["leader"] = summary.iloc[0].to_dict()

    return results, summary, metadata
