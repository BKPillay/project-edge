
from __future__ import annotations
import pandas as pd
from models.common import add_features
def build_backtest_summary(df:pd.DataFrame)->dict:
    feat=add_features(df)
    return {"draws_tested":int(len(df)),"actual_over_92_5_rate":round(float(feat["over_92_5"].mean())*100,2),"latest_sum":int(feat.iloc[-1]["sum"]),"latest_over_under":"Over 92.5" if bool(feat.iloc[-1]["over_92_5"]) else "Under 92.5","note":"Heavy back-testing belongs in model scripts, not Streamlit page rendering."}
