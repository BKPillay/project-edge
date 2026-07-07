from __future__ import annotations

import pandas as pd

from models.common import add_features


def predict_over_under(df: pd.DataFrame, threshold: float = 92.5) -> dict:
    feat = add_features(df)

    all_rate = float(feat["over_92_5"].mean())
    r20 = float(feat.tail(min(20, len(feat)))["over_92_5"].mean())
    r50 = float(feat.tail(min(50, len(feat)))["over_92_5"].mean())
    r100 = float(feat.tail(min(100, len(feat)))["over_92_5"].mean())
    r250 = float(feat.tail(min(250, len(feat)))["over_92_5"].mean())
    avg10 = float(feat.tail(min(10, len(feat)))["sum"].mean())

    prob_over = (
        0.30 * 0.50
        + 0.20 * all_rate
        + 0.15 * r20
        + 0.15 * r50
        + 0.10 * r100
        + 0.10 * r250
    )
    prob_over += 0.025 if avg10 > threshold else -0.025
    prob_over = max(0.05, min(0.95, prob_over))

    prediction = "Over 92.5" if prob_over >= 0.5 else "Under 92.5"
    confidence = prob_over if prediction.startswith("Over") else 1 - prob_over

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "prob_over": round(prob_over * 100, 2),
        "prob_under": round((1 - prob_over) * 100, 2),
        "recent_avg_sum_10": round(avg10, 2),
        "all_time_over_rate": round(all_rate * 100, 2),
        "recent_20_over_rate": round(r20 * 100, 2),
        "recent_50_over_rate": round(r50 * 100, 2),
        "recent_100_over_rate": round(r100 * 100, 2),
        "recent_250_over_rate": round(r250 * 100, 2),
    }
