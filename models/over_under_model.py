
from __future__ import annotations
import pandas as pd
from models.common import add_features
def predict_over_under(df:pd.DataFrame, threshold:float=92.5)->dict:
    feat=add_features(df); all_rate=float(feat["over_92_5"].mean()); r20=float(feat.tail(min(20,len(feat)))["over_92_5"].mean()); r50=float(feat.tail(min(50,len(feat)))["over_92_5"].mean()); r100=float(feat.tail(min(100,len(feat)))["over_92_5"].mean()); avg10=float(feat.tail(min(10,len(feat)))["sum"].mean())
    prob_over=.40*.50+.20*all_rate+.15*r20+.15*r50+.10*r100
    prob_over += .025 if avg10>threshold else -.025; prob_over=max(.05,min(.95,prob_over))
    prediction="Over 92.5" if prob_over>=.5 else "Under 92.5"; confidence=prob_over if prediction.startswith("Over") else 1-prob_over
    return {"prediction":prediction,"confidence":round(confidence*100,2),"prob_over":round(prob_over*100,2),"prob_under":round((1-prob_over)*100,2),"recent_avg_sum_10":round(avg10,2),"all_time_over_rate":round(all_rate*100,2),"recent_20_over_rate":round(r20*100,2),"recent_50_over_rate":round(r50*100,2),"recent_100_over_rate":round(r100*100,2)}
