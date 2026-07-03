
from pathlib import Path
import pandas as pd
import streamlit as st
from src.edge_model import NUMBER_COLS, add_draw_features, combination_scores, load_history, number_scores, over_under_prediction, pair_stats, performance_summary, simple_top3_backtest, triplet_stats, walk_forward_over_under_backtest

st.set_page_config(page_title="EDGE AI", page_icon="🎯", layout="wide")
DATA_PATH = Path("data/daily_lotto_history.csv")
st.title("🎯 EDGE AI — SA Daily Lotto")
st.caption("Ranking engine, analytics lab, and performance tracker. Not guaranteed prediction.")
if not DATA_PATH.exists():
    st.error("Missing data/daily_lotto_history.csv"); st.stop()
df = load_history(DATA_PATH); features = add_draw_features(df)
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Dashboard", "Predictions", "Analytics", "Back-testing", "History / Updates"], label_visibility="collapsed")
    st.divider(); st.header("➕ Add new draw")
    draw_date = st.date_input("Draw date")
    cols = st.columns(5)
    nums = [cols[i].number_input(f"N{i+1}", min_value=1, max_value=36, value=i+1, step=1) for i in range(5)]
    overwrite_existing = st.checkbox("Replace same date", value=True)
    if st.button("Save draw", use_container_width=True):
        new_nums = sorted([int(n) for n in nums])
        if len(set(new_nums)) != 5: st.error("The 5 numbers must be unique.")
        else:
            draw_date_ts = pd.to_datetime(draw_date); existing_date = df["draw_date"].dt.date.eq(draw_date_ts.date()).any()
            if existing_date and not overwrite_existing: st.error("Date exists. Tick replace.")
            else:
                new_row = pd.DataFrame([{"draw_date": draw_date_ts, "n1": new_nums[0], "n2": new_nums[1], "n3": new_nums[2], "n4": new_nums[3], "n5": new_nums[4]}])
                updated = pd.concat([df, new_row], ignore_index=True)
                if overwrite_existing: updated = updated.drop_duplicates(subset=["draw_date"], keep="last")
                updated = updated.sort_values("draw_date").reset_index(drop=True); updated.to_csv(DATA_PATH, index=False)
                st.success(f"Saved: {draw_date_ts.date()} — {'-'.join(map(str, new_nums))}"); st.info("Refresh the app to recalculate.")
    st.download_button("Download history CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="daily_lotto_history.csv", mime="text/csv", use_container_width=True)

def metric_row():
    latest = features.iloc[-1]
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Draws loaded", len(df)); c2.metric("Latest draw", str(latest["draw_date"].date())); c3.metric("Latest sum", int(latest["sum"])); c4.metric("Latest O/U", "Over" if latest["over_92_5"] else "Under")

if page == "Dashboard":
    st.subheader("Dashboard"); metric_row(); st.divider(); left,right=st.columns([2,1])
    with left:
        st.subheader("Top 10 ranked combinations"); st.dataframe(combination_scores(df, top_n=10), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Top 3 numbers"); st.dataframe(number_scores(df).head(3)[["number","score","long_count","recent_20_count","recent_50_count","gap"]], use_container_width=True, hide_index=True)
        st.subheader("Over / Under 92.5"); ou=over_under_prediction(df); st.metric(ou["prediction"], f'{ou["confidence"]}% confidence'); st.caption(f"Over {ou['prob_over']}% | Under {ou['prob_under']}%")
    st.divider(); st.subheader("Performance snapshot")
    with st.spinner("Running walk-forward back-tests..."): perf=performance_summary(df)
    p1,p2,p3,p4=st.columns(4); p1.metric("O/U accuracy", f"{perf['over_under_accuracy']}%"); p2.metric("O/U tests", perf["over_under_tests"]); p3.metric("Top 3 avg matches", perf["top3_avg_matches"]); p4.metric("Top 3 2+ hit rate", f"{perf['top3_2plus_hit_rate']}%")
elif page == "Predictions":
    st.subheader("Predictions"); metric_row(); top_n=st.slider("How many combinations?", 5, 50, 10)
    st.dataframe(combination_scores(df, top_n=top_n), use_container_width=True, hide_index=True); st.subheader("Individual number ranking"); st.dataframe(number_scores(df), use_container_width=True, hide_index=True); st.subheader("Over / Under 92.5"); st.json(over_under_prediction(df))
elif page == "Analytics":
    st.subheader("Analytics"); metric_row(); tab1,tab2,tab3,tab4=st.tabs(["Numbers","Pairs","Triplets","Structures"])
    with tab1:
        ns=number_scores(df); st.dataframe(ns, use_container_width=True, hide_index=True); st.bar_chart(ns.set_index("number")["score"])
    with tab2: st.dataframe(pair_stats(df).head(100), use_container_width=True, hide_index=True)
    with tab3: st.dataframe(triplet_stats(df, top_n=100), use_container_width=True, hide_index=True)
    with tab4:
        st.write("Most common structures"); st.dataframe(features["structure"].value_counts().reset_index(), use_container_width=True, hide_index=True); st.write("Sum trend"); st.line_chart(features.set_index("draw_date")["sum"])
elif page == "Back-testing":
    st.subheader("Back-testing"); st.warning("This is where we separate signal from nonsense.")
    with st.spinner("Running Over/Under walk-forward back-test..."): ou_bt=walk_forward_over_under_backtest(df)
    acc=round(float(ou_bt["correct"].mean())*100,2); st.metric("Over/Under 92.5 walk-forward accuracy", f"{acc}%"); st.dataframe(ou_bt.sort_values("draw_date", ascending=False).head(100), use_container_width=True, hide_index=True)
    st.divider()
    with st.spinner("Running Top 3 walk-forward back-test..."): t3_bt=simple_top3_backtest(df)
    c1,c2=st.columns(2); c1.metric("Top 3 average matches", round(float(t3_bt["matches"].mean()),3)); c2.metric("Top 3 2+ match rate", f"{round(float((t3_bt['matches']>=2).mean())*100,2)}%")
    st.dataframe(t3_bt.sort_values("draw_date", ascending=False).head(100), use_container_width=True, hide_index=True)
elif page == "History / Updates":
    st.subheader("History / Updates"); metric_row(); st.info("On free Streamlit hosting, manual changes may not persist forever. Download the CSV after adding draws and commit it back to GitHub.")
    uploaded=st.file_uploader("Bulk replace history CSV", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_df=pd.read_csv(uploaded); required={"draw_date","n1","n2","n3","n4","n5"}
            if not required.issubset(set(uploaded_df.columns)): st.error("CSV must contain draw_date,n1,n2,n3,n4,n5")
            else:
                uploaded_df["draw_date"]=pd.to_datetime(uploaded_df["draw_date"])
                for c in NUMBER_COLS: uploaded_df[c]=pd.to_numeric(uploaded_df[c], errors="raise").astype(int)
                uploaded_df=uploaded_df.sort_values("draw_date").drop_duplicates(subset=["draw_date"], keep="last")
                if st.button("Replace app history with uploaded CSV"):
                    uploaded_df.to_csv(DATA_PATH, index=False); st.success("CSV replaced. Refresh app.")
        except Exception as e: st.error(f"Could not process CSV: {e}")
    st.subheader("Historical draws"); st.dataframe(features.sort_values("draw_date", ascending=False), use_container_width=True, hide_index=True)
