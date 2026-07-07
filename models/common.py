
from __future__ import annotations
from pathlib import Path
from typing import Dict
import pandas as pd
NUMBER_COLS=["n1","n2","n3","n4","n5"]
NUMBERS=range(1,37)
PRIMES={2,3,5,7,11,13,17,19,23,29,31}
def load_history(path: str|Path)->pd.DataFrame:
    df=pd.read_csv(path); df["draw_date"]=pd.to_datetime(df["draw_date"])
    for c in NUMBER_COLS: df[c]=pd.to_numeric(df[c],errors="raise").astype(int)
    df=df.sort_values("draw_date").drop_duplicates("draw_date",keep="last").reset_index(drop=True)
    validate_history(df); return df
def validate_history(df: pd.DataFrame)->None:
    for idx,row in df.iterrows():
        vals=[int(row[c]) for c in NUMBER_COLS]
        if len(set(vals))!=5 or any(v<1 or v>36 for v in vals):
            raise ValueError(f"Invalid numbers on row {idx+2}: {vals}")
def minmax(values: Dict)->Dict:
    if not values: return {}
    mn,mx=min(values.values()),max(values.values())
    if mx==mn: return {k:.5 for k in values}
    return {k:(v-mn)/(mx-mn) for k,v in values.items()}
def draw_sets(df:pd.DataFrame):
    return [set(map(int,row)) for row in df[NUMBER_COLS].values.tolist()]
def add_features(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy(); nums=out[NUMBER_COLS]
    out["sum"]=nums.sum(axis=1); out["over_92_5"]=out["sum"]>92.5
    out["odd_count"]=nums.apply(lambda r: sum(int(x)%2 for x in r),axis=1)
    out["even_count"]=5-out["odd_count"]
    out["low_count"]=nums.apply(lambda r: sum(int(x)<=18 for x in r),axis=1)
    out["high_count"]=5-out["low_count"]
    reps=[0]
    for i in range(1,len(out)):
        reps.append(len(set(out.loc[i-1,NUMBER_COLS].tolist()) & set(out.loc[i,NUMBER_COLS].tolist())))
    out["repeat_from_previous"]=reps
    for w in [5,10,20,50,100]:
        out[f"rolling_sum_{w}"]=out["sum"].rolling(w,min_periods=1).mean()
        out[f"rolling_over_rate_{w}"]=out["over_92_5"].rolling(w,min_periods=1).mean()
    return out
