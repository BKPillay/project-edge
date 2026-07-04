from __future__ import annotations
import itertools, math
from pathlib import Path
import numpy as np
import pandas as pd

NUMBERS = range(1, 37)
NUMBER_COLS = ["n1", "n2", "n3", "n4", "n5"]
COMBINATIONS = list(itertools.combinations(NUMBERS, 5))
PRIMES = {2,3,5,7,11,13,17,19,23,29,31}
FIBONACCI = {1,2,3,5,8,13,21,34}

def load_history(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    for c in NUMBER_COLS:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(int)
    df = df.sort_values("draw_date").drop_duplicates("draw_date", keep="last").reset_index(drop=True)
    for idx, r in df.iterrows():
        vals = [int(r[c]) for c in NUMBER_COLS]
        if len(set(vals)) != 5 or any(v < 1 or v > 36 for v in vals):
            raise ValueError(f"Invalid Daily Lotto row {idx+2}: {vals}")
    return df

def count_consecutive_pairs(numbers) -> int:
    nums = sorted(map(int, numbers))
    return sum(1 for a,b in zip(nums, nums[1:]) if b == a + 1)

def add_draw_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    nums = out[NUMBER_COLS]
    out["sum"] = nums.sum(axis=1)
    out["over_92_5"] = out["sum"] > 92.5
    out["odd_count"] = nums.apply(lambda r: sum(int(x) % 2 for x in r), axis=1)
    out["even_count"] = 5 - out["odd_count"]
    out["low_count"] = nums.apply(lambda r: sum(int(x) <= 18 for x in r), axis=1)
    out["high_count"] = 5 - out["low_count"]
    out["range"] = nums.max(axis=1) - nums.min(axis=1)
    out["consecutive_pairs"] = nums.apply(lambda r: count_consecutive_pairs(r.tolist()), axis=1)
    out["prime_count"] = nums.apply(lambda r: sum(int(x) in PRIMES for x in r), axis=1)
    out["fibonacci_count"] = nums.apply(lambda r: sum(int(x) in FIBONACCI for x in r), axis=1)
    out["structure"] = out["odd_count"].astype(str)+" Odd / "+out["even_count"].astype(str)+" Even · "+out["low_count"].astype(str)+" Low / "+out["high_count"].astype(str)+" High"
    repeats = [0]
    for i in range(1, len(out)):
        repeats.append(len(set(out.loc[i-1, NUMBER_COLS]) & set(out.loc[i, NUMBER_COLS])))
    out["repeat_from_previous"] = repeats
    for w in [5,10,20,50,100]:
        out[f"rolling_sum_{w}"] = out["sum"].rolling(w, min_periods=1).mean()
        out[f"rolling_over_rate_{w}"] = out["over_92_5"].rolling(w, min_periods=1).mean()
    return out

def minmax(d):
    if not d: return {}
    mn, mx = min(d.values()), max(d.values())
    if math.isclose(mn, mx): return {k:.5 for k in d}
    return {k:(v-mn)/(mx-mn) for k,v in d.items()}

def draw_sets(df):
    return [set(map(int, row)) for row in df[NUMBER_COLS].values.tolist()]

def count_numbers(draws):
    c = {n:0 for n in NUMBERS}
    for draw in draws:
        for n in draw: c[n] += 1
    return c

def gap_counts(draws):
    gaps = {}
    for n in NUMBERS:
        gap = 0
        for draw in reversed(draws):
            if n in draw: break
            gap += 1
        gaps[n] = min(gap, 60)
    return gaps

def number_reason(long_count, r20, r50, gap):
    if r20 >= 4: return "Strong recent momentum"
    if r50 >= 9: return "Strong 50-draw form"
    if long_count >= 390: return "Strong long-term frequency"
    if gap >= 15: return "Overdue factor"
    return "Balanced profile"

def number_scores(df: pd.DataFrame) -> pd.DataFrame:
    draws = draw_sets(df)
    longc, r20, r50, r100, gaps = count_numbers(draws), count_numbers(draws[-20:]), count_numbers(draws[-50:]), count_numbers(draws[-100:]), gap_counts(draws)
    sl, s20, s50, s100, sg = map(minmax, [longc, r20, r50, r100, gaps])
    rows = []
    for n in NUMBERS:
        score = .35*sl[n] + .20*s20[n] + .18*s50[n] + .17*s100[n] + .10*sg[n]
        rows.append({"Rank":None,"Number":n,"Score":round(score*100,2),"Long Count":longc[n],"Last 20":r20[n],"Last 50":r50[n],"Last 100":r100[n],"Gap":gaps[n],"Reason":number_reason(longc[n], r20[n], r50[n], gaps[n])})
    out = pd.DataFrame(rows).sort_values(["Score","Number"], ascending=[False, True]).reset_index(drop=True)
    out["Rank"] = out.index + 1
    return out

def pair_counts(df):
    c = {p:0 for p in itertools.combinations(NUMBERS, 2)}
    for row in df[NUMBER_COLS].values.tolist():
        for p in itertools.combinations(sorted(map(int, row)), 2): c[p] += 1
    return c

def pair_last_seen(df, pair):
    for gap, (_, r) in enumerate(df.iloc[::-1].iterrows()):
        vals = set(r[NUMBER_COLS].tolist())
        if pair[0] in vals and pair[1] in vals: return gap
    return len(df)

def pair_predictions(df, top_n=10):
    pc = pair_counts(df); norm = minmax(pc); vals = list(pc.values())
    p90 = np.percentile(vals, 90)
    rows = []
    for pair, count in pc.items():
        gap = pair_last_seen(df, pair)
        score = .80*norm[pair] + .20*(min(gap, 60)/60)
        rows.append({"Rank":None,"Pair":f"{pair[0]}-{pair[1]}","Score":round(score*100,2),"Appeared":count,"Last Seen":"latest draw" if gap==0 else f"{gap} draws ago","Reason":"Frequent pair" if count >= p90 else "Pair + gap balance"})
    out = pd.DataFrame(rows).sort_values(["Score","Pair"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    out["Rank"] = out.index + 1
    return out

def triplet_counts(df):
    c = {}
    for row in df[NUMBER_COLS].values.tolist():
        for t in itertools.combinations(sorted(map(int, row)), 3): c[t] = c.get(t,0)+1
    return c

def triplet_last_seen(df, triplet):
    for gap, (_, r) in enumerate(df.iloc[::-1].iterrows()):
        vals = set(r[NUMBER_COLS].tolist())
        if all(n in vals for n in triplet): return gap
    return len(df)

def triplet_predictions(df, top_n=10):
    tc = triplet_counts(df)
    if not tc: return pd.DataFrame(columns=["Rank","Triplet","Score","Appeared","Last Seen","Reason"])
    norm = minmax(tc); p95 = np.percentile(list(tc.values()), 95)
    rows = []
    for t, count in tc.items():
        gap = triplet_last_seen(df, t)
        score = .85*norm[t] + .15*(min(gap, 120)/120)
        rows.append({"Rank":None,"Triplet":"-".join(map(str,t)),"Score":round(score*100,2),"Appeared":count,"Last Seen":"latest draw" if gap==0 else f"{gap} draws ago","Reason":"Recurring triplet" if count >= p95 else "Triplet + gap balance"})
    out = pd.DataFrame(rows).sort_values(["Score","Triplet"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    out["Rank"] = out.index + 1
    return out

def structure_score(combo):
    s=sum(combo); odd=sum(n%2 for n in combo); low=sum(n<=18 for n in combo); consec=count_consecutive_pairs(combo); prime=sum(n in PRIMES for n in combo)
    return (0.30 if 65<=s<=120 else .10) + (.25 if odd in (2,3) else .08) + (.25 if low in (2,3) else .08) + (.12 if consec<=1 else .04) + (.08 if prime in (1,2,3) else .03)

def combination_scores(df, top_n=10):
    ns = number_scores(df); nscore = dict(zip(ns["Number"], ns["Score"])); pc = pair_counts(df); pn = minmax(pc)
    rows=[]
    for combo in COMBINATIONS:
        avg_num = np.mean([nscore[n] for n in combo]) / 100
        avg_pair = np.mean([pn[tuple(sorted(p))] for p in itertools.combinations(combo, 2)])
        score = (0.68*avg_num + 0.17*avg_pair + 0.15*structure_score(combo))*100
        odd=sum(n%2 for n in combo); low=sum(n<=18 for n in combo)
        rows.append({"Rank":None,"Combination":"-".join(map(str,combo)),"Score":round(score,2),"Sum":sum(combo),"Odd":odd,"Even":5-odd,"Low":low,"High":5-low,"Prime":sum(n in PRIMES for n in combo)})
    out = pd.DataFrame(rows).sort_values(["Score","Combination"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    out["Rank"] = out.index + 1
    return out

def over_under_prediction(df, threshold=92.5):
    feat = add_draw_features(df)
    all_rate = float((feat["sum"] > threshold).mean())
    r20 = float((feat.tail(min(20,len(feat)))["sum"] > threshold).mean())
    r50 = float((feat.tail(min(50,len(feat)))["sum"] > threshold).mean())
    r100 = float((feat.tail(min(100,len(feat)))["sum"] > threshold).mean())
    avg10 = float(feat.tail(min(10,len(feat)))["sum"].mean())
    prob_over = .40*.50 + .20*all_rate + .15*r20 + .15*r50 + .10*r100
    prob_over += .025 if avg10 > threshold else -.025
    prob_over = max(.05, min(.95, prob_over))
    pred = "Over 92.5" if prob_over >= .5 else "Under 92.5"
    conf = prob_over if pred.startswith("Over") else 1-prob_over
    return {"prediction":pred,"confidence":round(conf*100,2),"prob_over":round(prob_over*100,2),"prob_under":round((1-prob_over)*100,2),"all_time_over_rate":round(all_rate*100,2),"recent_20_over_rate":round(r20*100,2),"recent_50_over_rate":round(r50*100,2),"recent_100_over_rate":round(r100*100,2),"recent_avg_sum_10":round(avg10,2)}

def walk_forward_over_under_backtest(df, min_train=200):
    rows=[]
    for i in range(min_train, len(df)):
        pred=over_under_prediction(df.iloc[:i]); actual_sum=int(df.iloc[i][NUMBER_COLS].sum()); actual="Over 92.5" if actual_sum>92.5 else "Under 92.5"
        rows.append({"draw_date":df.iloc[i]["draw_date"],"prediction":pred["prediction"],"confidence":pred["confidence"],"actual_sum":actual_sum,"actual":actual,"correct":pred["prediction"]==actual})
    return pd.DataFrame(rows)

def simple_top3_backtest(df, min_train=200):
    rows=[]
    for i in range(min_train, len(df)):
        top3=number_scores(df.iloc[:i]).head(3)["Number"].astype(int).tolist(); actual=set(map(int, df.iloc[i][NUMBER_COLS].tolist()))
        rows.append({"draw_date":df.iloc[i]["draw_date"],"top3":"-".join(map(str, top3)),"matches":len(actual & set(top3))})
    return pd.DataFrame(rows)

def performance_summary(df):
    ou=walk_forward_over_under_backtest(df); t3=simple_top3_backtest(df)
    return {"over_under_accuracy":round(float(ou["correct"].mean())*100,2),"over_under_tests":int(len(ou)),"top3_avg_matches":round(float(t3["matches"].mean()),3),"top3_2plus_hit_rate":round(float((t3["matches"]>=2).mean())*100,2),"top3_tests":int(len(t3))}

# EDGE AI v8: latest draw review + sampled fast backtests

def walk_forward_over_under_backtest_fast(df, min_train=200, step=1):
    rows=[]
    for i in range(min_train, len(df), step):
        pred=over_under_prediction(df.iloc[:i]); actual_sum=int(df.iloc[i][NUMBER_COLS].sum()); actual='Over 92.5' if actual_sum>92.5 else 'Under 92.5'
        rows.append({'draw_date':df.iloc[i]['draw_date'],'prediction':pred['prediction'],'confidence':pred['confidence'],'actual_sum':actual_sum,'actual':actual,'correct':pred['prediction']==actual})
    return pd.DataFrame(rows)

def simple_top3_backtest_fast(df, min_train=200, step=5):
    rows=[]
    for i in range(min_train, len(df), step):
        top3=number_scores(df.iloc[:i]).head(3)['Number'].astype(int).tolist(); actual=set(map(int, df.iloc[i][NUMBER_COLS].tolist()))
        rows.append({'draw_date':df.iloc[i]['draw_date'],'top3':'-'.join(map(str, top3)),'matches':len(actual & set(top3))})
    return pd.DataFrame(rows)

def latest_draw_review(df):
    if len(df) < 2:
        return {'available': False, 'reason': 'Need at least two draws.'}
    pre = df.iloc[:-1].copy()
    latest = df.iloc[-1].copy()
    latest_numbers = sorted([int(latest[c]) for c in NUMBER_COLS])
    actual_set = set(latest_numbers)

    ns = number_scores(pre)
    ns_map = ns.set_index('Number').to_dict(orient='index')
    pairs = pair_predictions(pre, top_n=666)
    pair_rank = {row['Pair']: row for _, row in pairs.iterrows()}
    trips = triplet_predictions(pre, top_n=10000)
    trip_rank = {row['Triplet']: row for _, row in trips.iterrows()}
    combos = combination_scores(pre, top_n=10)
    top3 = set(ns.head(3)['Number'].astype(int).tolist())
    ou = over_under_prediction(pre)
    actual_sum = sum(latest_numbers)
    actual_ou = 'Over 92.5' if actual_sum > 92.5 else 'Under 92.5'

    ball_rows=[]
    for n in latest_numbers:
        info=ns_map[n]; rank=int(info['Rank']); score=float(info['Score'])
        if rank <= 3: insight='Elite pre-draw pick'
        elif rank <= 10: insight='Strong pre-draw pick'
        elif rank <= 20: insight='Mid-table model pick'
        else: insight='Model underweighted this ball'
        ball_rows.append({'Ball':n,'Pre-draw Rank':rank,'Model Score':score,'Long Count':int(info['Long Count']),'Last 20':int(info['Last 20']),'Last 50':int(info['Last 50']),'Gap Before Draw':int(info['Gap']),'Insight':insight})

    pair_rows=[]
    for a,b in itertools.combinations(latest_numbers,2):
        key=f'{a}-{b}'
        if key in pair_rank:
            row=pair_rank[key]
            pair_rows.append({'Pair':key,'Pre-draw Pair Rank':int(row['Rank']),'Score':float(row['Score']),'Appeared Before':int(row['Appeared']),'Insight':'High-ranked pair hit' if int(row['Rank']) <= 20 else 'Lower-ranked pair hit'})

    trip_rows=[]
    for t in itertools.combinations(latest_numbers,3):
        key='-'.join(map(str,t))
        if key in trip_rank:
            row=trip_rank[key]
            trip_rows.append({'Triplet':key,'Pre-draw Triplet Rank':int(row['Rank']),'Score':float(row['Score']),'Appeared Before':int(row['Appeared']),'Insight':'Recurring triplet hit' if int(row['Rank']) <= 50 else 'Rare/lower-ranked triplet hit'})

    best_matches=0; best_combo=None
    for _, row in combos.iterrows():
        cset=set(map(int, row['Combination'].split('-')))
        matches=len(cset & actual_set)
        if matches > best_matches:
            best_matches=matches; best_combo=row['Combination']

    return {'available':True,'draw_date':str(latest['draw_date'].date()),'latest_numbers':latest_numbers,'actual_sum':actual_sum,'actual_ou':actual_ou,'ou_prediction':ou['prediction'],'ou_correct':ou['prediction']==actual_ou,'ou_confidence':ou['confidence'],'top3_hits':len(actual_set & top3),'best_top10_combo':best_combo,'best_top10_combo_matches':best_matches,'ball_review':pd.DataFrame(ball_rows),'pair_review':pd.DataFrame(pair_rows).sort_values('Pre-draw Pair Rank').reset_index(drop=True) if pair_rows else pd.DataFrame(),'triplet_review':pd.DataFrame(trip_rows).sort_values('Pre-draw Triplet Rank').reset_index(drop=True) if trip_rows else pd.DataFrame()}

# EDGE AI v8.1 efficient prediction overrides

def _pair_count_last_seen(df):
    counts = {pair: 0 for pair in itertools.combinations(NUMBERS, 2)}
    last_seen = {pair: len(df) for pair in itertools.combinations(NUMBERS, 2)}
    for idx, row in df.reset_index(drop=True).iterrows():
        draw = sorted(map(int, row[NUMBER_COLS].tolist()))
        for pair in itertools.combinations(draw, 2):
            counts[pair] += 1
            last_seen[pair] = len(df) - 1 - idx
    return counts, last_seen

def pair_predictions(df, top_n=10):
    pc, last_seen = _pair_count_last_seen(df)
    norm = minmax(pc); p90 = np.percentile(list(pc.values()), 90)
    rows=[]
    for pair, count in pc.items():
        gap=last_seen[pair]; score=.80*norm[pair]+.20*(min(gap,60)/60)
        rows.append({'Rank':None,'Pair':f'{pair[0]}-{pair[1]}','Score':round(score*100,2),'Appeared':count,'Last Seen':'latest draw' if gap==0 else f'{gap} draws ago','Reason':'Frequent pair' if count>=p90 else 'Pair + gap balance'})
    out=pd.DataFrame(rows).sort_values(['Score','Pair'], ascending=[False,True]).head(top_n).reset_index(drop=True)
    out['Rank']=out.index+1
    return out

def _triplet_count_last_seen(df):
    counts = {}
    last_seen = {}
    for idx, row in df.reset_index(drop=True).iterrows():
        draw = sorted(map(int, row[NUMBER_COLS].tolist()))
        for triplet in itertools.combinations(draw, 3):
            counts[triplet] = counts.get(triplet, 0) + 1
            last_seen[triplet] = len(df) - 1 - idx
    return counts, last_seen

def triplet_predictions(df, top_n=10):
    tc, last_seen = _triplet_count_last_seen(df)
    if not tc:
        return pd.DataFrame(columns=['Rank','Triplet','Score','Appeared','Last Seen','Reason'])
    norm=minmax(tc); p95=np.percentile(list(tc.values()),95)
    rows=[]
    for t, count in tc.items():
        gap=last_seen[t]; score=.85*norm[t]+.15*(min(gap,120)/120)
        rows.append({'Rank':None,'Triplet':'-'.join(map(str,t)),'Score':round(score*100,2),'Appeared':count,'Last Seen':'latest draw' if gap==0 else f'{gap} draws ago','Reason':'Recurring triplet' if count>=p95 else 'Triplet + gap balance'})
    out=pd.DataFrame(rows).sort_values(['Score','Triplet'], ascending=[False,True]).head(top_n).reset_index(drop=True)
    out['Rank']=out.index+1
    return out

# EDGE AI v8.2 latest review fast override: no full combination scoring

def latest_draw_review(df):
    if len(df) < 2:
        return {'available': False, 'reason': 'Need at least two draws.'}
    pre = df.iloc[:-1].copy()
    latest = df.iloc[-1].copy()
    latest_numbers = sorted([int(latest[c]) for c in NUMBER_COLS])
    actual_set = set(latest_numbers)

    ns = number_scores(pre)
    ns_map = ns.set_index('Number').to_dict(orient='index')
    top3 = set(ns.head(3)['Number'].astype(int).tolist())
    top10_numbers = set(ns.head(10)['Number'].astype(int).tolist())
    ou = over_under_prediction(pre)
    actual_sum = sum(latest_numbers)
    actual_ou = 'Over 92.5' if actual_sum > 92.5 else 'Under 92.5'

    ball_rows=[]
    for n in latest_numbers:
        info=ns_map[n]; rank=int(info['Rank']); score=float(info['Score'])
        if rank <= 3: insight='Elite pre-draw pick'
        elif rank <= 10: insight='Strong pre-draw pick'
        elif rank <= 20: insight='Mid-table model pick'
        else: insight='Model underweighted this ball'
        ball_rows.append({'Ball':n,'Pre-draw Rank':rank,'Model Score':score,'Long Count':int(info['Long Count']),'Last 20':int(info['Last 20']),'Last 50':int(info['Last 50']),'Gap Before Draw':int(info['Gap']),'Insight':insight})

    # Only evaluate the 10 actual pairs and 10 actual triplets directly, not all ranked triplets.
    pc, pls = _pair_count_last_seen(pre)
    pair_rows=[]
    pnorm=minmax(pc)
    all_pair_scores=[]
    for pair,count in pc.items():
        gap=pls[pair]
        all_pair_scores.append((.80*pnorm[pair]+.20*(min(gap,60)/60), pair))
    all_pair_scores=sorted(all_pair_scores, reverse=True)
    pair_rank={pair:i+1 for i,(_,pair) in enumerate(all_pair_scores)}
    for a,b in itertools.combinations(latest_numbers,2):
        pair=(a,b); gap=pls.get(pair,len(pre)); score=.80*pnorm.get(pair,0)+.20*(min(gap,60)/60)
        pair_rows.append({'Pair':f'{a}-{b}','Pre-draw Pair Rank':pair_rank.get(pair),'Score':round(score*100,2),'Appeared Before':pc.get(pair,0),'Insight':'High-ranked pair hit' if pair_rank.get(pair,9999)<=20 else 'Lower-ranked pair hit'})

    tc,tls=_triplet_count_last_seen(pre)
    tnorm=minmax(tc)
    all_trip_scores=[]
    for trip,count in tc.items():
        gap=tls[trip]
        all_trip_scores.append((.85*tnorm[trip]+.15*(min(gap,120)/120), trip))
    all_trip_scores=sorted(all_trip_scores, reverse=True)
    trip_rank={trip:i+1 for i,(_,trip) in enumerate(all_trip_scores)}
    trip_rows=[]
    for t in itertools.combinations(latest_numbers,3):
        gap=tls.get(t,len(pre)); score=.85*tnorm.get(t,0)+.15*(min(gap,120)/120)
        trip_rows.append({'Triplet':'-'.join(map(str,t)),'Pre-draw Triplet Rank':trip_rank.get(t),'Score':round(score*100,2),'Appeared Before':tc.get(t,0),'Insight':'Recurring/high-ranked triplet hit' if trip_rank.get(t,999999)<=50 else 'Rare/lower-ranked triplet hit'})

    return {'available':True,'draw_date':str(latest['draw_date'].date()),'latest_numbers':latest_numbers,'actual_sum':actual_sum,'actual_ou':actual_ou,'ou_prediction':ou['prediction'],'ou_correct':ou['prediction']==actual_ou,'ou_confidence':ou['confidence'],'top3_hits':len(actual_set & top3),'top10_number_hits':len(actual_set & top10_numbers),'best_top10_combo':'N/A in fast review','best_top10_combo_matches':'N/A','ball_review':pd.DataFrame(ball_rows),'pair_review':pd.DataFrame(pair_rows).sort_values('Pre-draw Pair Rank').reset_index(drop=True),'triplet_review':pd.DataFrame(trip_rows).sort_values('Pre-draw Triplet Rank').reset_index(drop=True)}

# EDGE AI v8.3 fast backtest overrides

def _fast_over_under_from_sums(sums, threshold=92.5):
    all_rate=float((sums > threshold).mean())
    r20=float((sums.tail(min(20,len(sums))) > threshold).mean())
    r50=float((sums.tail(min(50,len(sums))) > threshold).mean())
    r100=float((sums.tail(min(100,len(sums))) > threshold).mean())
    avg10=float(sums.tail(min(10,len(sums))).mean())
    prob_over=.40*.50+.20*all_rate+.15*r20+.15*r50+.10*r100
    prob_over += .025 if avg10 > threshold else -.025
    prob_over=max(.05,min(.95,prob_over))
    pred='Over 92.5' if prob_over >= .5 else 'Under 92.5'
    conf=prob_over if pred.startswith('Over') else 1-prob_over
    return pred, round(conf*100,2)

def walk_forward_over_under_backtest_fast(df, min_train=200, step=10):
    sums=df[NUMBER_COLS].sum(axis=1)
    rows=[]
    for i in range(min_train, len(df), step):
        pred, conf = _fast_over_under_from_sums(sums.iloc[:i])
        actual_sum=int(sums.iloc[i]); actual='Over 92.5' if actual_sum>92.5 else 'Under 92.5'
        rows.append({'draw_date':df.iloc[i]['draw_date'],'prediction':pred,'confidence':conf,'actual_sum':actual_sum,'actual':actual,'correct':pred==actual})
    return pd.DataFrame(rows)
