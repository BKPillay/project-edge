from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd

from models.common import NUMBER_COLS, NUMBERS, PRIMES
from models.number_model import score_numbers
from models.pair_model import score_pairs
from models.triplet_model import score_triplets
from models.combination_model import score_combinations
from models.structural_model import combo_structure_score

NUM_LIST = list(NUMBERS)
MAX_NUM = max(NUM_LIST)
PAIR_LIST = list(combinations(NUM_LIST, 2))
TRIPLET_LIST = list(combinations(NUM_LIST, 3))
PAIR_INDEX = {p: i for i, p in enumerate(PAIR_LIST)}
TRIPLET_INDEX = {t: i for i, t in enumerate(TRIPLET_LIST)}


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


# ---------------------------------------------------------------------------
# Original exact EDGE path retained for parity checks / small sampled audits.
# ---------------------------------------------------------------------------

def edge_ai_strategy(train: pd.DataFrame) -> tuple[int, ...]:
    """Uses the original EDGE ensemble path: numbers -> pairs -> triplets -> combinations."""
    numbers = score_numbers(train)
    pairs = score_pairs(train, 100)
    triplets = score_triplets(train, 100)
    combos = score_combinations(numbers, pairs, triplets, top_n=1)

    if combos.empty:
        return hot_numbers_strategy(train)

    return _numbers_from_combo_string(combos.iloc[0]["combination"])


# ---------------------------------------------------------------------------
# Legacy simple strategies. Kept for compatibility, but the benchmark now uses
# StrategyBenchmarkCache because repeatedly scanning the dataframe is too slow.
# ---------------------------------------------------------------------------

def random_strategy(train: pd.DataFrame, seed: int) -> tuple[int, ...]:
    rng = np.random.default_rng(seed)
    return tuple(sorted(int(x) for x in rng.choice(NUM_LIST, 5, replace=False)))


def hot_numbers_strategy(train: pd.DataFrame) -> tuple[int, ...]:
    counts = Counter()
    for combo in _history_combos(train):
        counts.update(combo)
    return tuple(sorted(n for n, _ in counts.most_common(5)))


def cold_numbers_strategy(train: pd.DataFrame) -> tuple[int, ...]:
    counts = Counter()
    for combo in _history_combos(train):
        counts.update(combo)
    ranked = sorted(NUM_LIST, key=lambda n: (counts[n], n))
    return tuple(sorted(ranked[:5]))


def overdue_strategy(train: pd.DataFrame) -> tuple[int, ...]:
    combos = _history_combos(train)
    last_seen = {n: -1 for n in NUM_LIST}
    for idx, combo in enumerate(combos):
        for n in combo:
            last_seen[n] = idx
    current = len(combos)
    ranked = sorted(NUM_LIST, key=lambda n: (current - last_seen[n], -n), reverse=True)
    return tuple(sorted(ranked[:5]))


def recent_repeat_strategy(train: pd.DataFrame, lookback: int = 3) -> tuple[int, ...]:
    counts = Counter()
    for combo in _history_combos(train)[-lookback:]:
        counts.update(combo)
    ranked = sorted(NUM_LIST, key=lambda n: (counts[n], -n), reverse=True)
    return tuple(sorted(ranked[:5]))


def pair_frequency_strategy(train: pd.DataFrame) -> tuple[int, ...]:
    pair_counts = Counter()
    for combo in _history_combos(train):
        for pair in combinations(combo, 2):
            pair_counts[pair] += 1

    number_scores = Counter()
    for pair, count in pair_counts.items():
        for n in pair:
            number_scores[n] += count

    ranked = sorted(NUM_LIST, key=lambda n: (number_scores[n], -n), reverse=True)
    return tuple(sorted(ranked[:5]))


def triplet_frequency_strategy(train: pd.DataFrame) -> tuple[int, ...]:
    triplet_counts = Counter()
    for combo in _history_combos(train):
        for triplet in combinations(combo, 3):
            triplet_counts[triplet] += 1

    number_scores = Counter()
    for triplet, count in triplet_counts.items():
        for n in triplet:
            number_scores[n] += count

    ranked = sorted(NUM_LIST, key=lambda n: (number_scores[n], -n), reverse=True)
    return tuple(sorted(ranked[:5]))


def balanced_hot_strategy(train: pd.DataFrame) -> tuple[int, ...]:
    counts = Counter()
    for combo in _history_combos(train):
        counts.update(combo)
    odds = sorted([n for n in NUM_LIST if n % 2 == 1], key=lambda n: (counts[n], -n), reverse=True)
    evens = sorted([n for n in NUM_LIST if n % 2 == 0], key=lambda n: (counts[n], -n), reverse=True)
    return tuple(sorted(odds[:3] + evens[:2]))


def _minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    mn = min(values.values())
    mx = max(values.values())
    if mx == mn:
        return {k: 0.5 for k in values}
    return {k: (v - mn) / (mx - mn) for k, v in values.items()}


@dataclass
class StrategyBenchmarkCache:
    """
    Prefix-cache for full walk-forward benchmarking.

    The previous benchmark rebuilt counters from scratch for every test draw.
    This cache precomputes cumulative number/pair/triplet counts and repeat/gap
    statistics once, then answers each strategy query in near-constant time.
    """

    combos: list[tuple[int, ...]]
    draw_numbers: list[int]
    draw_dates: list
    number_prefix: np.ndarray
    pair_prefix: np.ndarray
    triplet_prefix: np.ndarray
    pair_number_prefix: np.ndarray
    triplet_number_prefix: np.ndarray
    last_seen_prefix: np.ndarray
    gap_sum_prefix: np.ndarray
    gap_count_prefix: np.ndarray
    repeat_hit_prefix: dict[int, np.ndarray]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "StrategyBenchmarkCache":
        combos = [_combo_from_row(row) for _, row in df.iterrows()]
        draw_numbers = [int(x) for x in df["draw_number"].tolist()]
        draw_dates = df["draw_date"].tolist()
        n_draws = len(combos)

        number_prefix = np.zeros((n_draws + 1, MAX_NUM + 1), dtype=np.int32)
        pair_prefix = np.zeros((n_draws + 1, len(PAIR_LIST)), dtype=np.int16)
        triplet_prefix = np.zeros((n_draws + 1, len(TRIPLET_LIST)), dtype=np.int16)
        pair_number_prefix = np.zeros((n_draws + 1, MAX_NUM + 1), dtype=np.int32)
        triplet_number_prefix = np.zeros((n_draws + 1, MAX_NUM + 1), dtype=np.int32)
        last_seen_prefix = np.full((n_draws + 1, MAX_NUM + 1), -1, dtype=np.int32)
        gap_sum_prefix = np.zeros((n_draws + 1, MAX_NUM + 1), dtype=np.float64)
        gap_count_prefix = np.zeros((n_draws + 1, MAX_NUM + 1), dtype=np.int32)

        last_seen = np.full(MAX_NUM + 1, -1, dtype=np.int32)
        gap_sum = np.zeros(MAX_NUM + 1, dtype=np.float64)
        gap_count = np.zeros(MAX_NUM + 1, dtype=np.int32)

        for i, combo in enumerate(combos, start=1):
            prev_i = i - 1
            number_prefix[i] = number_prefix[prev_i]
            pair_prefix[i] = pair_prefix[prev_i]
            triplet_prefix[i] = triplet_prefix[prev_i]
            pair_number_prefix[i] = pair_number_prefix[prev_i]
            triplet_number_prefix[i] = triplet_number_prefix[prev_i]

            for num in combo:
                number_prefix[i, num] += 1
                if last_seen[num] >= 0:
                    gap_sum[num] += prev_i - int(last_seen[num])
                    gap_count[num] += 1
                last_seen[num] = prev_i

            for pair in combinations(combo, 2):
                pidx = PAIR_INDEX[pair]
                pair_prefix[i, pidx] += 1
                for num in pair:
                    pair_number_prefix[i, num] += 1

            for triplet in combinations(combo, 3):
                tidx = TRIPLET_INDEX[triplet]
                triplet_prefix[i, tidx] += 1
                for num in triplet:
                    triplet_number_prefix[i, num] += 1

            last_seen_prefix[i] = last_seen
            gap_sum_prefix[i] = gap_sum
            gap_count_prefix[i] = gap_count

        repeat_hit_prefix: dict[int, np.ndarray] = {}
        for k in (1, 2, 3):
            hit_prefix = np.zeros((n_draws + 1, MAX_NUM + 1), dtype=np.int32)
            for p in range(0, n_draws - k):
                hit_prefix[p + 1] = hit_prefix[p]
                inter = set(combos[p]) & set(combos[p + k])
                for num in inter:
                    hit_prefix[p + 1, num] += 1
            for p in range(max(0, n_draws - k), n_draws):
                hit_prefix[p + 1] = hit_prefix[p]
            repeat_hit_prefix[k] = hit_prefix

        return cls(
            combos=combos,
            draw_numbers=draw_numbers,
            draw_dates=draw_dates,
            number_prefix=number_prefix,
            pair_prefix=pair_prefix,
            triplet_prefix=triplet_prefix,
            pair_number_prefix=pair_number_prefix,
            triplet_number_prefix=triplet_number_prefix,
            last_seen_prefix=last_seen_prefix,
            gap_sum_prefix=gap_sum_prefix,
            gap_count_prefix=gap_count_prefix,
            repeat_hit_prefix=repeat_hit_prefix,
        )

    def actual(self, idx: int) -> tuple[int, ...]:
        return self.combos[idx]

    def random(self, idx: int) -> tuple[int, ...]:
        rng = np.random.default_rng(self.draw_numbers[idx])
        return tuple(sorted(int(x) for x in rng.choice(NUM_LIST, 5, replace=False)))

    def hot(self, idx: int) -> tuple[int, ...]:
        counts = self.number_prefix[idx]
        ranked = sorted(NUM_LIST, key=lambda n: (int(counts[n]), -n), reverse=True)
        return tuple(sorted(ranked[:5]))

    def cold(self, idx: int) -> tuple[int, ...]:
        counts = self.number_prefix[idx]
        ranked = sorted(NUM_LIST, key=lambda n: (int(counts[n]), n))
        return tuple(sorted(ranked[:5]))

    def overdue(self, idx: int) -> tuple[int, ...]:
        last_seen = self.last_seen_prefix[idx]
        ranked = sorted(NUM_LIST, key=lambda n: (idx - int(last_seen[n]), -n), reverse=True)
        return tuple(sorted(ranked[:5]))

    def recent_repeat(self, idx: int, lookback: int = 3) -> tuple[int, ...]:
        start = max(0, idx - lookback)
        counts = self.number_prefix[idx] - self.number_prefix[start]
        ranked = sorted(NUM_LIST, key=lambda n: (int(counts[n]), -n), reverse=True)
        return tuple(sorted(ranked[:5]))

    def pair_frequency(self, idx: int) -> tuple[int, ...]:
        scores = self.pair_number_prefix[idx]
        ranked = sorted(NUM_LIST, key=lambda n: (int(scores[n]), -n), reverse=True)
        return tuple(sorted(ranked[:5]))

    def triplet_frequency(self, idx: int) -> tuple[int, ...]:
        scores = self.triplet_number_prefix[idx]
        ranked = sorted(NUM_LIST, key=lambda n: (int(scores[n]), -n), reverse=True)
        return tuple(sorted(ranked[:5]))

    def balanced_hot(self, idx: int) -> tuple[int, ...]:
        counts = self.number_prefix[idx]
        odds = sorted([n for n in NUM_LIST if n % 2 == 1], key=lambda n: (int(counts[n]), -n), reverse=True)
        evens = sorted([n for n in NUM_LIST if n % 2 == 0], key=lambda n: (int(counts[n]), -n), reverse=True)
        return tuple(sorted(odds[:3] + evens[:2]))

    def _recent_pair_sets(self, idx: int, depth: int = 3):
        recent = []
        for back in range(1, min(depth, idx) + 1):
            recent.append(set(combinations(self.combos[idx - back], 2)))
        return recent

    def _recent_triplet_sets(self, idx: int, depth: int = 3):
        recent = []
        for back in range(1, min(depth, idx) + 1):
            recent.append(set(combinations(self.combos[idx - back], 3)))
        return recent

    def _edge_number_rows(self, idx: int) -> pd.DataFrame:
        counts = self.number_prefix[idx]

        def window_counts(w: int) -> dict[int, int]:
            start = max(0, idx - w)
            arr = self.number_prefix[idx] - self.number_prefix[start]
            return {n: int(arr[n]) for n in NUM_LIST}

        long_counts = {n: int(counts[n]) for n in NUM_LIST}
        r20 = window_counts(20)
        r50 = window_counts(50)
        r100 = window_counts(100)
        r250 = window_counts(250)
        r500 = window_counts(500)

        last_seen = self.last_seen_prefix[idx]
        gaps = {n: min(idx - int(last_seen[n]) if int(last_seen[n]) >= 0 else idx, 500) for n in NUM_LIST}

        long_s = _minmax(long_counts)
        r20_s = _minmax(r20)
        r50_s = _minmax(r50)
        r100_s = _minmax(r100)
        r250_s = _minmax(r250)
        r500_s = _minmax(r500)
        gap_s = _minmax(gaps)

        latest_draw = set(self.combos[idx - 1]) if idx >= 1 else set()
        previous_draw = set(self.combos[idx - 2]) if idx >= 2 else set()
        third_last_draw = set(self.combos[idx - 3]) if idx >= 3 else set()

        rows = []
        baseline = 5 / 36
        for n in NUM_LIST:
            momentum_score = 0.45 * r20_s[n] + 0.30 * r50_s[n] + 0.25 * r100_s[n]
            frequency_score = 0.45 * long_s[n] + 0.25 * r250_s[n] + 0.30 * r500_s[n]
            gap_score = gap_s[n]
            raw_final = 0.45 * frequency_score + 0.35 * momentum_score + 0.20 * gap_score

            gap_count = int(self.gap_count_prefix[idx, n])
            avg_gap = float(self.gap_sum_prefix[idx, n] / gap_count) if gap_count else None
            current_gap = gaps[n]
            gap_ratio = round(current_gap / avg_gap, 3) if avg_gap else None

            repeat_rates = []
            repeat_out = {}
            for k, w in ((1, 0.55), (2, 0.30), (3, 0.15)):
                eligible_idx = max(0, idx - k)
                eligible = int(self.number_prefix[eligible_idx, n])
                hits = int(self.repeat_hit_prefix[k][eligible_idx, n])
                rate = hits / eligible if eligible else 0.0
                repeat_out[k] = (rate, hits, eligible)
                if eligible:
                    repeat_rates.append((w, rate))

            if repeat_rates:
                total_w = sum(w for w, _ in repeat_rates)
                repeat_tendency = sum(w * r for w, r in repeat_rates) / total_w
            else:
                repeat_tendency = baseline
            repeat_index = repeat_tendency / baseline if baseline else 1
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
                "repeat_index": round(repeat_index, 3),
                "repeat_after_1_rate": round(repeat_out[1][0] * 100, 2),
                "repeat_after_2_rate": round(repeat_out[2][0] * 100, 2),
                "repeat_after_3_rate": round(repeat_out[3][0] * 100, 2),
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
        return out.sort_values(["selection_rank", "number"], ascending=[True, True]).reset_index(drop=True)

    def _edge_pairs_rows(self, idx: int, top_n: int = 100) -> pd.DataFrame:
        counts = self.pair_prefix[idx]
        max_count = int(counts.max()) if len(counts) else 0
        recent_sets = self._recent_pair_sets(idx, 3)
        rows = []
        for pair, pidx in PAIR_INDEX.items():
            count = int(counts[pidx])
            raw_score = round((count / max_count) * 100, 2) if max_count else 50.0
            if len(recent_sets) >= 1 and pair in recent_sets[0]:
                penalty = 30.0
                note = "appeared in latest draw"
            elif len(recent_sets) >= 2 and pair in recent_sets[1]:
                penalty = 18.0
                note = "appeared 2 draws ago"
            elif len(recent_sets) >= 3 and pair in recent_sets[2]:
                penalty = 9.0
                note = "appeared 3 draws ago"
            else:
                penalty = 0.0
                note = "not in last 3 draws"
            rows.append({
                "rank": None,
                "pair": f"{pair[0]}-{pair[1]}",
                "score": raw_score,
                "appeared": count,
                "recent_hit_penalty": penalty,
                "selection_score": round(max(0, raw_score - penalty), 2),
                "recency_note": note,
            })
        raw = pd.DataFrame(rows).sort_values(["score", "pair"], ascending=[False, True]).reset_index(drop=True)
        raw["rank"] = raw.index + 1
        selected = raw.sort_values(["selection_score", "pair"], ascending=[False, True]).reset_index(drop=True)
        selected["selection_rank"] = selected.index + 1
        out = raw.merge(selected[["pair", "selection_rank"]], on="pair", how="left")
        return out.sort_values(["selection_rank", "pair"], ascending=[True, True]).head(top_n).reset_index(drop=True)

    def _edge_triplets_rows(self, idx: int, top_n: int = 100) -> pd.DataFrame:
        counts = self.triplet_prefix[idx]
        nonzero = counts[counts > 0]
        if len(nonzero) == 0:
            return pd.DataFrame(columns=[
                "rank", "triplet", "score", "appeared", "recent_hit_penalty",
                "selection_score", "selection_rank", "recency_note"
            ])
        max_count = int(nonzero.max())
        recent_sets = self._recent_triplet_sets(idx, 3)
        rows = []
        for triplet, tidx in TRIPLET_INDEX.items():
            count = int(counts[tidx])
            if count <= 0:
                continue
            raw_score = round((count / max_count) * 100, 2)
            if len(recent_sets) >= 1 and triplet in recent_sets[0]:
                penalty = 45.0
                note = "appeared in latest draw"
            elif len(recent_sets) >= 2 and triplet in recent_sets[1]:
                penalty = 25.0
                note = "appeared 2 draws ago"
            elif len(recent_sets) >= 3 and triplet in recent_sets[2]:
                penalty = 12.0
                note = "appeared 3 draws ago"
            else:
                penalty = 0.0
                note = "not in last 3 draws"
            rows.append({
                "rank": None,
                "triplet": "-".join(map(str, triplet)),
                "score": raw_score,
                "appeared": count,
                "recent_hit_penalty": penalty,
                "selection_score": round(max(0, raw_score - penalty), 2),
                "recency_note": note,
            })
        raw = pd.DataFrame(rows).sort_values(["score", "triplet"], ascending=[False, True]).reset_index(drop=True)
        raw["rank"] = raw.index + 1
        selected = raw.sort_values(["selection_score", "triplet"], ascending=[False, True]).reset_index(drop=True)
        selected["selection_rank"] = selected.index + 1
        out = raw.merge(selected[["triplet", "selection_rank"]], on="triplet", how="left")
        return out.sort_values(["selection_rank", "triplet"], ascending=[True, True]).head(top_n).reset_index(drop=True)


    def _edge_number_score_maps(self, idx: int):
        counts = self.number_prefix[idx]

        def window_arr(w: int):
            start = max(0, idx - w)
            return self.number_prefix[idx] - self.number_prefix[start]

        arr20 = window_arr(20); arr50 = window_arr(50); arr100 = window_arr(100)
        arr250 = window_arr(250); arr500 = window_arr(500)
        long_counts = {n: int(counts[n]) for n in NUM_LIST}
        r20 = {n: int(arr20[n]) for n in NUM_LIST}
        r50 = {n: int(arr50[n]) for n in NUM_LIST}
        r100 = {n: int(arr100[n]) for n in NUM_LIST}
        r250 = {n: int(arr250[n]) for n in NUM_LIST}
        r500 = {n: int(arr500[n]) for n in NUM_LIST}

        last_seen = self.last_seen_prefix[idx]
        gaps = {n: min(idx - int(last_seen[n]) if int(last_seen[n]) >= 0 else idx, 500) for n in NUM_LIST}

        long_s = _minmax(long_counts); r20_s = _minmax(r20); r50_s = _minmax(r50); r100_s = _minmax(r100)
        r250_s = _minmax(r250); r500_s = _minmax(r500); gap_s = _minmax(gaps)

        latest_draw = set(self.combos[idx - 1]) if idx >= 1 else set()
        previous_draw = set(self.combos[idx - 2]) if idx >= 2 else set()
        third_last_draw = set(self.combos[idx - 3]) if idx >= 3 else set()

        baseline = 5 / 36
        selection_scores: dict[int, float] = {}
        repeat_penalties: dict[int, float] = {}
        final_scores: dict[int, float] = {}

        for n in NUM_LIST:
            momentum_score = 0.45 * r20_s[n] + 0.30 * r50_s[n] + 0.25 * r100_s[n]
            frequency_score = 0.45 * long_s[n] + 0.25 * r250_s[n] + 0.30 * r500_s[n]
            gap_score = gap_s[n]
            raw_final = 0.45 * frequency_score + 0.35 * momentum_score + 0.20 * gap_score

            repeat_rates = []
            for k, w in ((1, 0.55), (2, 0.30), (3, 0.15)):
                eligible_idx = max(0, idx - k)
                eligible = int(self.number_prefix[eligible_idx, n])
                hits = int(self.repeat_hit_prefix[k][eligible_idx, n])
                if eligible:
                    repeat_rates.append((w, hits / eligible))
            if repeat_rates:
                total_w = sum(w for w, _ in repeat_rates)
                repeat_tendency = sum(w * r for w, r in repeat_rates) / total_w
            else:
                repeat_tendency = baseline
            repeat_index = repeat_tendency / baseline if baseline else 1

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
            selection_scores[n] = round(final_score - repeat_penalty, 2)
            repeat_penalties[n] = repeat_penalty
            final_scores[n] = final_score

        pool = [n for n, _ in sorted(selection_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:15]]
        return selection_scores, repeat_penalties, final_scores, pool

    def _edge_pair_score_map_fast(self, idx: int, top_n: int = 100):
        counts = self.pair_prefix[idx]
        max_count = int(counts.max()) if len(counts) else 0
        recent_sets = self._recent_pair_sets(idx, 3)
        scored = []
        for pair, pidx in PAIR_INDEX.items():
            count = int(counts[pidx])
            raw = round((count / max_count) * 100, 2) if max_count else 50.0
            if len(recent_sets) >= 1 and pair in recent_sets[0]:
                penalty = 30.0
            elif len(recent_sets) >= 2 and pair in recent_sets[1]:
                penalty = 18.0
            elif len(recent_sets) >= 3 and pair in recent_sets[2]:
                penalty = 9.0
            else:
                penalty = 0.0
            sel = round(max(0, raw - penalty), 2)
            scored.append((pair, sel))
        scored.sort(key=lambda x: (-x[1], x[0]))
        top = scored[:top_n]
        score_map = dict(top)
        default = float(np.mean([v for _, v in top])) if top else 50.0
        return score_map, default

    def _edge_triplet_score_map_fast(self, idx: int, top_n: int = 100):
        counts = self.triplet_prefix[idx]
        max_count = int(counts.max()) if len(counts) else 0
        if max_count <= 0:
            return {}, 50.0
        recent_sets = self._recent_triplet_sets(idx, 3)
        scored = []
        nonzero_indices = np.nonzero(counts)[0]
        for tidx in nonzero_indices:
            triplet = TRIPLET_LIST[int(tidx)]
            count = int(counts[tidx])
            raw = round((count / max_count) * 100, 2)
            if len(recent_sets) >= 1 and triplet in recent_sets[0]:
                penalty = 45.0
            elif len(recent_sets) >= 2 and triplet in recent_sets[1]:
                penalty = 25.0
            elif len(recent_sets) >= 3 and triplet in recent_sets[2]:
                penalty = 12.0
            else:
                penalty = 0.0
            sel = round(max(0, raw - penalty), 2)
            scored.append((triplet, sel))
        scored.sort(key=lambda x: (-x[1], x[0]))
        top = scored[:top_n]
        score_map = dict(top)
        default = float(np.mean([v for _, v in top])) if top else 50.0
        return score_map, default

    def edge_ai_fast_direct(self, idx: int) -> tuple[int, ...]:
        number_scores, repeat_penalties, _final_scores, pool = self._edge_number_score_maps(idx)
        if len(pool) < 5:
            return self.hot(idx)
        pair_map, pair_default = self._edge_pair_score_map_fast(idx, 100)
        triplet_map, triplet_default = self._edge_triplet_score_map_fast(idx, 100)

        best_combo = None
        best_score = -1.0
        best_key = None
        sorted_pool = sorted(pool)
        ns = number_scores
        rp = repeat_penalties
        pm = pair_map
        tm = triplet_map
        pdflt = pair_default
        tdflt = triplet_default

        # Manual nested loops are much faster here than repeatedly building
        # temporary lists and calling numpy.mean inside itertools loops.
        m = len(sorted_pool)
        for ia in range(m - 4):
            a = sorted_pool[ia]
            for ib in range(ia + 1, m - 3):
                b = sorted_pool[ib]
                for ic in range(ib + 1, m - 2):
                    c = sorted_pool[ic]
                    for idd in range(ic + 1, m - 1):
                        d = sorted_pool[idd]
                        for ie in range(idd + 1, m):
                            e = sorted_pool[ie]
                            combo = (a, b, c, d, e)
                            avg_number_score = (ns[a] + ns[b] + ns[c] + ns[d] + ns[e]) / 5.0
                            avg_pair_score = (
                                pm.get((a, b), pdflt) + pm.get((a, c), pdflt) + pm.get((a, d), pdflt) + pm.get((a, e), pdflt)
                                + pm.get((b, c), pdflt) + pm.get((b, d), pdflt) + pm.get((b, e), pdflt)
                                + pm.get((c, d), pdflt) + pm.get((c, e), pdflt) + pm.get((d, e), pdflt)
                            ) / 10.0
                            avg_triplet_score = (
                                tm.get((a, b, c), tdflt) + tm.get((a, b, d), tdflt) + tm.get((a, b, e), tdflt)
                                + tm.get((a, c, d), tdflt) + tm.get((a, c, e), tdflt) + tm.get((a, d, e), tdflt)
                                + tm.get((b, c, d), tdflt) + tm.get((b, c, e), tdflt) + tm.get((b, d, e), tdflt)
                                + tm.get((c, d, e), tdflt)
                            ) / 10.0
                            total = a + b + c + d + e
                            odd = (a % 2) + (b % 2) + (c % 2) + (d % 2) + (e % 2)
                            low = int(a <= 18) + int(b <= 18) + int(c <= 18) + int(d <= 18) + int(e <= 18)
                            structural_bonus = (5 if 65 <= total <= 120 else 0) + (4 if odd in (2, 3) else 0) + (4 if low in (2, 3) else 0)
                            structural_score = min(100.0, structural_bonus * 7.5)
                            ensemble_score = (
                                0.55 * avg_number_score
                                + 0.20 * avg_pair_score
                                + 0.10 * avg_triplet_score
                                + 0.15 * structural_score
                            )
                            cooled_count = int(rp.get(a, 0) > 0) + int(rp.get(b, 0) > 0) + int(rp.get(c, 0) > 0) + int(rp.get(d, 0) > 0) + int(rp.get(e, 0) > 0)
                            if cooled_count >= 3:
                                repeat_penalty = 5.0
                            elif cooled_count == 2:
                                repeat_penalty = 2.5
                            else:
                                repeat_penalty = 0.0
                            score = round(max(0, ensemble_score - repeat_penalty), 2)
                            key = f"{a}-{b}-{c}-{d}-{e}"
                            if score > best_score or (score == best_score and (best_key is None or key < best_key)):
                                best_score = score
                                best_combo = combo
                                best_key = key
        return tuple(sorted(best_combo)) if best_combo else self.hot(idx)

    def edge_ai_fast(self, idx: int) -> tuple[int, ...]:
        numbers = self._edge_number_rows(idx)
        pairs = self._edge_pairs_rows(idx, 100)
        triplets = self._edge_triplets_rows(idx, 100)
        combos = score_combinations(numbers, pairs, triplets, top_n=1)
        if combos.empty:
            return self.hot(idx)
        return _numbers_from_combo_string(combos.iloc[0]["combination"])


def run_strategy_benchmark(
    df: pd.DataFrame,
    min_train: int = 300,
    step: int = 20,
    include_edge_ai: bool = True,
    edge_mode: str = "fast",
    progress_every: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Walk-forward strategy benchmark.

    edge_mode:
        fast  - optimized prefix-cache implementation suitable for full step=1 audits.
        exact - original dataframe path. Use only for small sampled parity checks.
    """
    if len(df) <= min_train:
        raise ValueError(f"Need more than {min_train} draws to benchmark. Found {len(df)}.")
    if step < 1:
        raise ValueError("step must be >= 1")
    if edge_mode not in {"fast", "exact"}:
        raise ValueError("edge_mode must be 'fast' or 'exact'")

    cache = StrategyBenchmarkCache.from_dataframe(df)
    rows = []
    test_indices = list(range(min_train, len(df), step))

    strategy_names = [
        "Random",
        "Hot Numbers",
        "Cold Numbers",
        "Overdue",
        "Recent Repeat",
        "Pair Frequency",
        "Triplet Frequency",
        "Balanced Hot",
    ]
    if include_edge_ai:
        strategy_names.insert(0, "EDGE AI")

    for test_no, idx in enumerate(test_indices, start=1):
        actual = cache.actual(idx)
        draw_date = cache.draw_dates[idx]
        draw_number = int(cache.draw_numbers[idx])

        if progress_every and (test_no == 1 or test_no % progress_every == 0 or test_no == len(test_indices)):
            print(f"Benchmark progress: {test_no}/{len(test_indices)} tests", flush=True)

        for strategy_name in strategy_names:
            if strategy_name == "EDGE AI":
                if edge_mode == "exact":
                    predicted = edge_ai_strategy(df.iloc[:idx].copy())
                else:
                    predicted = cache.edge_ai_fast_direct(idx)
            elif strategy_name == "Random":
                predicted = cache.random(idx)
            elif strategy_name == "Hot Numbers":
                predicted = cache.hot(idx)
            elif strategy_name == "Cold Numbers":
                predicted = cache.cold(idx)
            elif strategy_name == "Overdue":
                predicted = cache.overdue(idx)
            elif strategy_name == "Recent Repeat":
                predicted = cache.recent_repeat(idx)
            elif strategy_name == "Pair Frequency":
                predicted = cache.pair_frequency(idx)
            elif strategy_name == "Triplet Frequency":
                predicted = cache.triplet_frequency(idx)
            elif strategy_name == "Balanced Hot":
                predicted = cache.balanced_hot(idx)
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")

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
        "method": "Walk-forward validation. Every prediction uses only prior draws.",
        "min_train": int(min_train),
        "step": int(step),
        "tests_per_strategy": int(len(test_indices)),
        "include_edge_ai": bool(include_edge_ai),
        "edge_mode": edge_mode,
        "optimisation": "v2.9 prefix-cache benchmark engine. Avoids rebuilding counters from scratch per draw.",
        "warning": "Lottery outcomes are random. Treat this as model validation, not guaranteed prediction power.",
    }
    if len(summary):
        metadata["leader"] = summary.iloc[0].to_dict()

    return results, summary, metadata
