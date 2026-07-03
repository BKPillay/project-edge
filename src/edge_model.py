
from __future__ import annotations
import itertools, math
from pathlib import Path
import numpy as np
import pandas as pd

NUMBERS = range(1, 37)
COMBINATIONS = list(itertools.combinations(NUMBERS, 5))
NUMBER_COLS = ["n1", "n2", "n3", "n4", "n5"]
PRIMES = {2,3,5,7,11,13,17,19,23,29,31}
FIBONACCI = {1,2,3,5,8,13,21,34}

def load_history(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    for col in NUMBER_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    df = df.sort_values("draw_date").drop_duplicates(subset=["draw_date"], keep="last").reset_index(drop=True)
    validate_history(df)
    return df

def validate_history(df):
    for idx, row in df.iterrows():
        vals = [int(row[c]) for c in NUMBER_COLS]
        if len(set(vals)) != 5 or any(v < 1 or v > 36 for v in vals):
            raise ValueError(f"Invalid numbers on row {idx + 2}: {vals}")

def count_consecutive_pairs(numbers):
    numbers = sorted(numbers)
    return sum(1 for a, b in zip(numbers, numbers[1:]) if b == a + 1)

def add_draw_features(df):
    out = df.copy().reset_index(drop=True)
    nums = out[NUMBER_COLS]
    out["sum"] = nums.sum(axis=1)
    out["over_92_5"] = out["sum"] > 92.5
    out["odd_count"] = nums.apply(lambda r: sum(x % 2 for x in r), axis=1)
    out["even_count"] = 5 - out["odd_count"]
    out["low_count"] = nums.apply(lambda r: sum(x <= 18 for x in r), axis=1)
    out["high_count"] = 5 - out["low_count"]
    out["range"] = nums.max(axis=1) - nums.min(axis=1)
    out["consecutive_pairs"] = nums.apply(lambda r: count_consecutive_pairs(r.tolist()), axis=1)
    out["prime_count"] = nums.apply(lambda r: sum(x in PRIMES for x in r), axis=1)
    out["fibonacci_count"] = nums.apply(lambda r: sum(x in FIBONACCI for x in r), axis=1)
    out["multiple_3_count"] = nums.apply(lambda r: sum(x % 3 == 0 for x in r), axis=1)
    out["multiple_5_count"] = nums.apply(lambda r: sum(x % 5 == 0 for x in r), axis=1)
    out["structure"] = out["odd_count"].astype(str)+"O/"+out["even_count"].astype(str)+"E | "+out["low_count"].astype(str)+"L/"+out["high_count"].astype(str)+"H"
    repeats = [0]
    for i in range(1, len(out)):
        repeats.append(len(set(out.loc[i-1, NUMBER_COLS]) & set(out.loc[i, NUMBER_COLS])))
    out["repeat_from_previous"] = repeats
    for w in [5,10,20,50,100]:
        out[f"rolling_sum_{w}"] = out["sum"].rolling(w, min_periods=1).mean()
        out[f"rolling_over_rate_{w}"] = out["over_92_5"].rolling(w, min_periods=1).mean()
    return out

def minmax(values):
    mn, mx = min(values.values()), max(values.values())
    if math.isclose(mx, mn): return {k: 0.5 for k in values}
    return {k: (v - mn) / (mx - mn) for k, v in values.items()}

def draw_sets(df): return [set(map(int, row)) for row in df[NUMBER_COLS].values.tolist()]

def count_numbers(draws):
    counts = {n: 0 for n in NUMBERS}
    for draw in draws:
        for n in draw: counts[n] += 1
    return counts

def gap_counts(draws):
    gaps = {}
    for n in NUMBERS:
        gap = 0
        for draw in reversed(draws):
            if n in draw: break
            gap += 1
        gaps[n] = min(gap, 60)
    return gaps

def number_scores(df):
    draws = draw_sets(df)
    long_counts = count_numbers(draws)
    r20_counts = count_numbers(draws[-20:])
    r50_counts = count_numbers(draws[-50:])
    r100_counts = count_numbers(draws[-100:])
    gaps = gap_counts(draws)
    long_s, r20_s, r50_s, r100_s, gap_s = map(minmax, [long_counts, r20_counts, r50_counts, r100_counts, gaps])
    rows=[]
    for n in NUMBERS:
        score = 0.35*long_s[n]+0.20*r20_s[n]+0.18*r50_s[n]+0.17*r100_s[n]+0.10*gap_s[n]
        rows.append({"number":n,"score":round(score*100,3),"long_count":long_counts[n],"recent_20_count":r20_counts[n],"recent_50_count":r50_counts[n],"recent_100_count":r100_counts[n],"gap":gaps[n]})
    return pd.DataFrame(rows).sort_values(["score","number"], ascending=[False,True]).reset_index(drop=True)

def pair_counts(df):
    counts = {p: 0 for p in itertools.combinations(NUMBERS, 2)}
    for row in df[NUMBER_COLS].values.tolist():
        for p in itertools.combinations(sorted(map(int, row)), 2): counts[p]+=1
    return counts

def pair_stats(df):
    rows=[{"pair":f"{a}-{b}","count":c} for (a,b),c in pair_counts(df).items()]
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)

def triplet_stats(df, top_n=100):
    counts={}
    for row in df[NUMBER_COLS].values.tolist():
        for t in itertools.combinations(sorted(map(int,row)),3): counts[t]=counts.get(t,0)+1
    rows=[{"triplet":"-".join(map(str,k)),"count":v} for k,v in counts.items()]
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(top_n).reset_index(drop=True)

def structure_score(combo):
    s=sum(combo); odd=sum(n%2 for n in combo); low=sum(n<=18 for n in combo); cons=count_consecutive_pairs(list(combo)); prime=sum(n in PRIMES for n in combo)
    score=0
    score+=0.30 if 65<=s<=120 else 0.10
    score+=0.25 if odd in (2,3) else 0.08
    score+=0.25 if low in (2,3) else 0.08
    score+=0.12 if cons<=1 else 0.04
    score+=0.08 if prime in (1,2,3) else 0.03
    return score

def combination_scores(df, top_n=10):
    ns=number_scores(df); nscore=dict(zip(ns["number"], ns["score"])); pc=pair_counts(df); pair_norm=minmax(pc)
    rows=[]
    for combo in COMBINATIONS:
        avg_num=np.mean([nscore[n] for n in combo])/100
        avg_pair=np.mean([pair_norm[tuple(sorted(p))] for p in itertools.combinations(combo,2)])
        edge=(0.68*avg_num+0.17*avg_pair+0.15*structure_score(combo))*100
        rows.append({"combination":"-".join(map(str,combo)),"sum":sum(combo),"odd_count":sum(n%2 for n in combo),"low_count":sum(n<=18 for n in combo),"prime_count":sum(n in PRIMES for n in combo),"score":round(edge,3)})
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

def over_under_prediction(df, threshold=92.5):
    feat=add_draw_features(df)
    all_rate=float((feat["sum"]>threshold).mean()); r20=float((feat.tail(min(20,len(feat)))["sum"]>threshold).mean()); r50=float((feat.tail(min(50,len(feat)))["sum"]>threshold).mean()); r100=float((feat.tail(min(100,len(feat)))["sum"]>threshold).mean()); avg10=float(feat.tail(min(10,len(feat)))["sum"].mean())
    prob_over=0.40*0.50+0.20*all_rate+0.15*r20+0.15*r50+0.10*r100
    prob_over += 0.025 if avg10 > threshold else -0.025
    prob_over=max(0.05,min(0.95,prob_over)); pred="Over 92.5" if prob_over>=0.5 else "Under 92.5"; conf=prob_over if pred.startswith("Over") else 1-prob_over
    return {"prediction":pred,"confidence":round(conf*100,2),"prob_over":round(prob_over*100,2),"prob_under":round((1-prob_over)*100,2),"recent_avg_sum_10":round(avg10,2),"all_time_over_rate":round(all_rate*100,2),"recent_20_over_rate":round(r20*100,2),"recent_50_over_rate":round(r50*100,2),"recent_100_over_rate":round(r100*100,2)}

def walk_forward_over_under_backtest(df, min_train=200, threshold=92.5):
    rows=[]
    for i in range(min_train, len(df)):
        pred=over_under_prediction(df.iloc[:i], threshold); actual_sum=int(df.iloc[i][NUMBER_COLS].sum()); actual="Over 92.5" if actual_sum>threshold else "Under 92.5"
        rows.append({"draw_date":df.iloc[i]["draw_date"],"prediction":pred["prediction"],"confidence":pred["confidence"],"actual_sum":actual_sum,"actual":actual,"correct":pred["prediction"]==actual})
    return pd.DataFrame(rows)

def simple_top3_backtest(df, min_train=200):
    rows=[]
    for i in range(min_train, len(df)):
        top3=number_scores(df.iloc[:i]).head(3)["number"].astype(int).tolist(); actual=set(map(int,df.iloc[i][NUMBER_COLS].tolist()))
        rows.append({"draw_date":df.iloc[i]["draw_date"],"top3":"-".join(map(str,top3)),"matches":len(actual & set(top3))})
    return pd.DataFrame(rows)

def performance_summary(df):
    ou=walk_forward_over_under_backtest(df); t3=simple_top3_backtest(df)
    return {"over_under_accuracy":round(float(ou["correct"].mean())*100,2),"over_under_tests":int(len(ou)),"top3_avg_matches":round(float(t3["matches"].mean()),3),"top3_2plus_hit_rate":round(float((t3["matches"]>=2).mean())*100,2),"top3_tests":int(len(t3))}
