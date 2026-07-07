
from __future__ import annotations
import itertools, numpy as np, pandas as pd
from models.common import PRIMES
def score_combinations(numbers_df:pd.DataFrame, top_n:int=50, pool_size:int=13)->pd.DataFrame:
    pool=numbers_df.head(pool_size).copy(); score_map=dict(zip(pool["number"].astype(int),pool["score"].astype(float)))
    rows=[]
    for combo in itertools.combinations(sorted(score_map.keys()),5):
        total=sum(combo); odd=sum(n%2 for n in combo); low=sum(n<=18 for n in combo); prime=sum(n in PRIMES for n in combo)
        bonus=(5 if 65<=total<=120 else 0)+(4 if odd in (2,3) else 0)+(4 if low in (2,3) else 0)
        score=round(float(np.mean([score_map[n] for n in combo]))+bonus,2)
        rows.append({"rank":None,"combination":"-".join(map(str,combo)),"score":score,"sum":total,"odd":odd,"even":5-odd,"low":low,"high":5-low,"prime":prime})
    out=pd.DataFrame(rows).sort_values(["score","combination"],ascending=[False,True]).head(top_n).reset_index(drop=True); out["rank"]=out.index+1; return out
