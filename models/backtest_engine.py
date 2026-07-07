from __future__ import annotations

import pandas as pd

from models.common import add_features


def build_backtest_summary(df: pd.DataFrame) -> dict:
    feat = add_features(df)

    return {
        "draws_loaded": int(len(df)),
        "actual_over_92_5_rate": round(float(feat["over_92_5"].mean()) * 100, 2),
        "latest_sum": int(feat.iloc[-1]["sum"]),
        "latest_over_under": "Over 92.5" if bool(feat.iloc[-1]["over_92_5"]) else "Under 92.5",
        "avg_sum": round(float(feat["sum"].mean()), 2),
        "median_sum": round(float(feat["sum"].median()), 2),
        "most_common_structure": str(feat["structure"].mode().iloc[0]) if "structure" in feat.columns and len(feat) else None,
        "note": "Fast summary mode. Full walk-forward back-testing should run as a separate optional audit, not during every model update.",
    }
