from pathlib import Path

NUMBER_MODEL = Path("models/number_model.py")
APP_PATH = Path("app.py")

text = NUMBER_MODEL.read_text(encoding="utf-8")

old_sort = '    out = pd.DataFrame(rows).sort_values(["final_score", "number"], ascending=[False, True]).reset_index(drop=True)\n    out["rank"] = out.index + 1\n    return out\n'
new_sort = '    out = pd.DataFrame(rows).sort_values(["final_score", "number"], ascending=[False, True]).reset_index(drop=True)\n    out["rank"] = out.index + 1\n\n    # EDGE AI v2.4: post-hit cooling layer.\n    # final_score remains the raw model score.\n    # selection_score is the next-draw recommendation score.\n    recent_draws = []\n    if len(df) >= 1:\n        recent_draws.append(set(map(int, df.iloc[-1][NUMBER_COLS].tolist())))\n    if len(df) >= 2:\n        recent_draws.append(set(map(int, df.iloc[-2][NUMBER_COLS].tolist())))\n    if len(df) >= 3:\n        recent_draws.append(set(map(int, df.iloc[-3][NUMBER_COLS].tolist())))\n\n    def recency_penalty(number):\n        number = int(number)\n        if len(recent_draws) >= 1 and number in recent_draws[0]:\n            return 25.0\n        if len(recent_draws) >= 2 and number in recent_draws[1]:\n            return 15.0\n        if len(recent_draws) >= 3 and number in recent_draws[2]:\n            return 8.0\n        return 0.0\n\n    out["recency_penalty"] = out["number"].apply(recency_penalty)\n    out["selection_score"] = (out["final_score"] * (1 - out["recency_penalty"] / 100)).round(2)\n\n    selection_order = out.sort_values(["selection_score", "number"], ascending=[False, True]).reset_index(drop=True)\n    selection_order["selection_rank"] = selection_order.index + 1\n    out = out.merge(selection_order[["number", "selection_rank"]], on="number", how="left")\n\n    # Keep raw rank visible, but order the output by today\'s selection rank.\n    out = out.sort_values(["selection_rank", "number"], ascending=[True, True]).reset_index(drop=True)\n    return out\n'

if "selection_score" not in text:
    if old_sort not in text:
        raise RuntimeError("Could not patch number_model.py. Expected final sort block not found.")
    text = text.replace(old_sort, new_sort)
    NUMBER_MODEL.write_text(text, encoding="utf-8")
else:
    print("number_model.py already appears to have selection_score. Skipping model patch.")

app = APP_PATH.read_text(encoding="utf-8")

insert_marker = """features = read_csv("features.csv")
numbers = read_csv("number_predictions.csv")
pairs = read_csv("pair_predictions.csv")
"""

replacement = """features = read_csv("features.csv")
numbers = read_csv("number_predictions.csv")
if "selection_rank" in numbers.columns:
    numbers = numbers.sort_values(["selection_rank", "number"]).reset_index(drop=True)
pairs = read_csv("pair_predictions.csv")
"""

if insert_marker in app and "numbers = numbers.sort_values" not in app:
    app = app.replace(insert_marker, replacement)

app = app.replace(
    "round(numbers.head(3)['final_score'].mean(),2)",
    "round(numbers.head(3).get('selection_score', numbers.head(3)['final_score']).mean(),2)"
)

pred_marker = """elif page == "Predictions":
    st.markdown("<div class='edge-title'>Predictions</div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Full-history rankings. Sliders only slice saved outputs.</div>", unsafe_allow_html=True)
"""

pred_replacement = """elif page == "Predictions":
    st.markdown("<div class='edge-title'>Predictions</div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Full-history rankings. Sliders only slice saved outputs.</div>", unsafe_allow_html=True)
    if "selection_score" in numbers.columns:
        st.info("v2.4 uses selection_rank for recommendations. final_score remains the raw model score; selection_score applies a cooling penalty to numbers drawn in the last 3 draws.")
"""

if pred_marker in app and "v2.4 uses selection_rank" not in app:
    app = app.replace(pred_marker, pred_replacement)

APP_PATH.write_text(app, encoding="utf-8")

print("EDGE AI v2.4 recency cooling patch applied successfully.")
print("Next run: python scripts\\run_model_update.py")
