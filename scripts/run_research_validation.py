
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.common import NUMBER_COLS, load_history
from models.number_model import score_numbers

DATA_PATH = PROJECT_ROOT / "data" / "daily_lotto_history.csv"
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)


def expected_hits(n: int) -> float:
    return n * (5 / 36)


def repeat_rate_study(df: pd.DataFrame) -> pd.DataFrame:
    draw_sets = [set(map(int, row)) for row in df[NUMBER_COLS].values.tolist()]
    rows = []
    for number in range(1, 37):
        positions = [i for i, s in enumerate(draw_sets) if number in s]
        row = {"number": number, "appearances": len(positions), "baseline_rate": round((5 / 36) * 100, 3)}
        for lag in [1, 2, 3, 5, 10]:
            eligible = [i for i in positions if i + lag < len(draw_sets)]
            hits = sum(1 for i in eligible if number in draw_sets[i + lag])
            rate = hits / len(eligible) if eligible else 0
            row[f"repeat_after_{lag}_tests"] = len(eligible)
            row[f"repeat_after_{lag}_hits"] = hits
            row[f"repeat_after_{lag}_rate"] = round(rate * 100, 3)
        row["repeat_after_1_vs_baseline"] = round(row["repeat_after_1_rate"] - row["baseline_rate"], 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("repeat_after_1_rate", ascending=False)


def rank_with_penalty(train: pd.DataFrame, p1: float, p2: float, p3: float) -> list[int]:
    ns = score_numbers(train).copy()
    latest = set(map(int, train.iloc[-1][NUMBER_COLS].tolist())) if len(train) >= 1 else set()
    prev2 = set(map(int, train.iloc[-2][NUMBER_COLS].tolist())) if len(train) >= 2 else set()
    prev3 = set(map(int, train.iloc[-3][NUMBER_COLS].tolist())) if len(train) >= 3 else set()

    def penalty(n):
        n = int(n)
        if n in latest:
            return p1
        if n in prev2:
            return p2
        if n in prev3:
            return p3
        return 0

    score_col = "final_score" if "final_score" in ns.columns else "score"
    ns["research_score"] = ns[score_col] - ns["number"].apply(penalty)
    return ns.sort_values(["research_score", "number"], ascending=[False, True])["number"].astype(int).tolist()


def cooling_backtest(df: pd.DataFrame, min_train: int = 300, step: int = 20) -> pd.DataFrame:
    strategies = [
        ("no_penalty", 0, 0, 0),
        ("light_penalty", 10, 5, 2),
        ("current_style_penalty", 22, 12, 6),
        ("strong_penalty", 35, 20, 10),
    ]
    rows = []
    for name, p1, p2, p3 in strategies:
        t3, t5, t10, t15 = [], [], [], []
        for i in range(min_train, len(df), step):
            pred = rank_with_penalty(df.iloc[:i].copy(), p1, p2, p3)
            actual = set(map(int, df.iloc[i][NUMBER_COLS].tolist()))
            t3.append(len(set(pred[:3]) & actual))
            t5.append(len(set(pred[:5]) & actual))
            t10.append(len(set(pred[:10]) & actual))
            t15.append(len(set(pred[:15]) & actual))
        rows.append({
            "strategy": name,
            "penalty_latest": p1,
            "penalty_2_draws": p2,
            "penalty_3_draws": p3,
            "tests": len(t10),
            "top3_avg_hits": round(pd.Series(t3).mean(), 4),
            "top5_avg_hits": round(pd.Series(t5).mean(), 4),
            "top10_avg_hits": round(pd.Series(t10).mean(), 4),
            "top15_avg_hits": round(pd.Series(t15).mean(), 4),
            "top10_expected_random": round(expected_hits(10), 4),
            "top10_lift_vs_random": round(pd.Series(t10).mean() - expected_hits(10), 4),
        })
    return pd.DataFrame(rows).sort_values("top10_avg_hits", ascending=False)


def main():
    df = load_history(DATA_PATH)
    repeat_df = repeat_rate_study(df)
    cooling_df = cooling_backtest(df)

    repeat_df.to_csv(RESEARCH_DIR / "number_repeat_rate_study.csv", index=False)
    cooling_df.to_csv(RESEARCH_DIR / "recency_cooling_backtest.csv", index=False)

    summary = {
        "purpose": "Research validation only. Does not directly change live model.",
        "best_cooling_strategy": cooling_df.iloc[0].to_dict(),
        "strongest_next_draw_repeat_numbers": repeat_df.head(5).to_dict(orient="records"),
        "weakest_next_draw_repeat_numbers": repeat_df.tail(5).to_dict(orient="records"),
        "warning": "Keep model rules only if they improve out-of-sample results."
    }
    (RESEARCH_DIR / "research_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
