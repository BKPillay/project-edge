
from __future__ import annotations
import json, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT_ROOT))
from models.backtest_engine import build_backtest_summary
from models.combination_model import score_combinations
from models.common import NUMBER_COLS, add_features, load_history
from models.latest_draw_engine import build_latest_draw_review
from models.number_model import score_numbers
from models.over_under_model import predict_over_under
from models.pair_model import score_pairs
from models.triplet_model import score_triplets
DATA_PATH=PROJECT_ROOT/"data"/"daily_lotto_history.csv"; OUTPUT_DIR=PROJECT_ROOT/"outputs"; OUTPUT_DIR.mkdir(exist_ok=True)
def write_json(path,payload): path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
def main():
    df=load_history(DATA_PATH); features=add_features(df); numbers=score_numbers(df); pairs=score_pairs(df,50); triplets=score_triplets(df,50); combos=score_combinations(numbers,50); ou=predict_over_under(df); latest_review=build_latest_draw_review(df); backtest=build_backtest_summary(df)
    features.to_csv(OUTPUT_DIR/"features.csv",index=False); numbers.to_csv(OUTPUT_DIR/"number_predictions.csv",index=False); pairs.to_csv(OUTPUT_DIR/"pair_predictions.csv",index=False); triplets.to_csv(OUTPUT_DIR/"triplet_predictions.csv",index=False); combos.to_csv(OUTPUT_DIR/"combination_predictions.csv",index=False); latest_review.to_csv(OUTPUT_DIR/"latest_draw_ball_review.csv",index=False)
    write_json(OUTPUT_DIR/"over_under.json",ou); write_json(OUTPUT_DIR/"backtest_summary.json",backtest)
    summary={"draws_loaded":int(len(df)),"latest_draw_date":str(df["draw_date"].max().date()),"latest_numbers":[int(df.iloc[-1][c]) for c in NUMBER_COLS],"generated_files":["features.csv","number_predictions.csv","pair_predictions.csv","triplet_predictions.csv","combination_predictions.csv","over_under.json","latest_draw_ball_review.csv","backtest_summary.json"]}
    write_json(OUTPUT_DIR/"run_summary.json",summary); print("EDGE AI outputs regenerated."); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
