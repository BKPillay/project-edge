
from __future__ import annotations
import pandas as pd
from models.common import NUMBER_COLS
from models.number_model import score_numbers
from models.over_under_model import predict_over_under
def build_latest_draw_review(df:pd.DataFrame)->pd.DataFrame:
    if len(df)<2: return pd.DataFrame()
    pre=df.iloc[:-1].copy(); latest=df.iloc[-1]; pre_numbers=score_numbers(pre); pre_ou=predict_over_under(pre); nmap=pre_numbers.set_index("number").to_dict(orient="index")
    nums=sorted([int(latest[c]) for c in NUMBER_COLS]); actual_sum=sum(nums); actual_ou="Over 92.5" if actual_sum>92.5 else "Under 92.5"
    rows=[]
    for n in nums:
        info=nmap[n]; rank=int(info["rank"])
        insight="Elite pre-draw pick" if rank<=3 else "Strong pre-draw pick" if rank<=10 else "Mid-ranked pick" if rank<=20 else "Model underweighted this ball"
        rows.append({"draw_date":str(latest["draw_date"].date()),"ball":n,"pre_draw_rank":rank,"model_score":info["score"],"last_20":info["last_20"],"last_50":info["last_50"],"gap":info["gap"],"actual_sum":actual_sum,"actual_ou":actual_ou,"ou_prediction_before_draw":pre_ou["prediction"],"ou_correct":pre_ou["prediction"]==actual_ou,"insight":insight})
    return pd.DataFrame(rows)
