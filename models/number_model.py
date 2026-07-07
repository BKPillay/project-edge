
from __future__ import annotations
import pandas as pd
from models.common import NUMBERS, draw_sets, minmax
def count_numbers(draws):
    counts={n:0 for n in NUMBERS}
    for draw in draws:
        for n in draw: counts[n]+=1
    return counts
def gap_counts(draws):
    gaps={}
    for n in NUMBERS:
        gap=0
        for draw in reversed(draws):
            if n in draw: break
            gap+=1
        gaps[n]=min(gap,60)
    return gaps
def score_numbers(df:pd.DataFrame)->pd.DataFrame:
    draws=draw_sets(df); long_counts=count_numbers(draws); r20=count_numbers(draws[-20:]); r50=count_numbers(draws[-50:]); r100=count_numbers(draws[-100:]); gaps=gap_counts(draws)
    long_s=minmax(long_counts); r20_s=minmax(r20); r50_s=minmax(r50); r100_s=minmax(r100); gap_s=minmax(gaps)
    rows=[]
    for n in NUMBERS:
        score=.35*long_s[n]+.20*r20_s[n]+.18*r50_s[n]+.17*r100_s[n]+.10*gap_s[n]
        rows.append({"rank":None,"number":n,"score":round(score*100,2),"long_count":long_counts[n],"last_20":r20[n],"last_50":r50[n],"last_100":r100[n],"gap":gaps[n]})
    out=pd.DataFrame(rows).sort_values(["score","number"],ascending=[False,True]).reset_index(drop=True); out["rank"]=out.index+1; return out
