from pathlib import Path
import pandas as pd
import streamlit as st
from src.edge_model import (
    NUMBER_COLS, add_draw_features, combination_scores, load_history,
    number_scores, over_under_prediction, pair_predictions, performance_summary,
    simple_top3_backtest, triplet_predictions, walk_forward_over_under_backtest,
)

st.set_page_config(page_title="EDGE AI", page_icon="🎯", layout="wide")
DATA_PATH = Path("data/daily_lotto_history.csv")

st.markdown("""
<style>
:root{--bg:#07111f;--panel:rgba(11,24,43,.88);--border:rgba(120,145,180,.22);--text:#eef4ff;--muted:#9fb0c7;--green:#39d975;--blue:#3f8cff;--purple:#9b5cff;}
.stApp{background:radial-gradient(circle at top left,rgba(95,68,255,.16),transparent 30%),radial-gradient(circle at top right,rgba(30,144,255,.12),transparent 28%),linear-gradient(180deg,#07111f 0%,#050b14 100%);color:var(--text)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#081422 0%,#050b14 100%);border-right:1px solid var(--border)}
.block-container{padding-top:1.4rem;max-width:1500px} h1,h2,h3{color:var(--text);letter-spacing:-.02em}
.edge-title{font-size:2.15rem;font-weight:900;margin-bottom:.1rem}.edge-ai{color:var(--purple)}.edge-subtitle{color:var(--muted);margin-bottom:1.1rem}
.edge-card{background:linear-gradient(180deg,rgba(14,31,56,.94),rgba(7,17,31,.94));border:1px solid var(--border);border-radius:14px;padding:1.05rem 1.2rem;box-shadow:0 12px 30px rgba(0,0,0,.22);min-height:132px}.edge-label{color:var(--muted);font-size:.88rem;margin-bottom:.25rem}.edge-big-green{color:var(--green);font-size:2.25rem;font-weight:900;line-height:1.05}.edge-big-blue{color:var(--blue);font-size:1.55rem;font-weight:800}.edge-pill{display:inline-block;border:1px solid rgba(155,92,255,.85);border-radius:999px;padding:.35rem .58rem;margin:.15rem .18rem .15rem 0;min-width:40px;text-align:center;background:rgba(155,92,255,.09);font-weight:800}.edge-pill.blue{border-color:rgba(63,140,255,.85);background:rgba(63,140,255,.09)}.edge-progress{height:8px;border-radius:999px;background:rgba(255,255,255,.12);overflow:hidden;margin:.5rem 0}.edge-progress>div{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--green),var(--purple))}.edge-grid{display:grid;grid-template-columns:repeat(5,minmax(54px,1fr));gap:.75rem;margin-top:.6rem}.edge-number-bubble{border:1px solid var(--green);border-radius:999px;height:54px;width:54px;display:flex;align-items:center;justify-content:center;font-weight:900;margin:auto;background:rgba(57,217,117,.08)}.edge-score{text-align:center;color:var(--muted);font-size:.82rem;margin-top:.25rem}div[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:12px;overflow:hidden}.stTabs [data-baseweb="tab-list"]{gap:14px}.stTabs [data-baseweb="tab"]{background:transparent;border-radius:10px 10px 0 0;color:var(--muted)}.stTabs [aria-selected="true"]{color:white;border-bottom:3px solid var(--purple)}
</style>
""", unsafe_allow_html=True)

def card(title, body, footer=""):
    st.markdown(f"""<div class='edge-card'><div class='edge-label'>{title}</div>{body}<div style='color:var(--muted);font-size:.82rem;margin-top:.65rem;'>{footer}</div></div>""", unsafe_allow_html=True)

def metric_cards(df, features):
    latest = features.iloc[-1]
    c1,c2,c3,c4 = st.columns(4)
    with c1: card("Draws Loaded", f"<div class='edge-big-blue'>{len(df):,}</div>", "Historical records")
    with c2: card("Latest Draw", f"<div class='edge-big-blue'>{latest['draw_date'].date()}</div>", "Most recent CSV record")
    with c3: card("Latest Sum", f"<div class='edge-big-blue'>{int(latest['sum'])}</div>", "Latest draw total")
    with c4: card("Latest Structure", f"<div class='edge-big-blue'>{latest['odd_count']}O/{latest['even_count']}E</div>", f"{latest['low_count']} Low / {latest['high_count']} High")

def number_grid(numbers_df):
    html = "<div class='edge-grid'>"
    for _, row in numbers_df.iterrows():
        html += f"<div><div class='edge-number-bubble'>{int(row['Number']):02d}</div><div class='edge-score'>{row['Score']}</div></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

if not DATA_PATH.exists():
    st.error("Missing data/daily_lotto_history.csv")
    st.stop()

df = load_history(DATA_PATH)
features = add_draw_features(df)

with st.sidebar:
    st.markdown("<div class='edge-title'>EDGE <span class='edge-ai'>AI</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Daily Lotto Intelligence</div>", unsafe_allow_html=True)
    page = st.radio("Navigation", ["Dashboard", "Predictions", "Analytics", "Back-testing", "History / Updates"], label_visibility="collapsed")
    st.divider(); st.subheader("➕ Add Draw")
    draw_date = st.date_input("Draw date")
    cols = st.columns(5)
    nums = [cols[i].number_input(f"N{i+1}", min_value=1, max_value=36, value=i+1, step=1) for i in range(5)]
    overwrite_existing = st.checkbox("Replace same date", value=True)
    if st.button("Save draw", use_container_width=True):
        new_nums = sorted([int(n) for n in nums])
        if len(set(new_nums)) != 5:
            st.error("The 5 numbers must be unique.")
        else:
            dts = pd.to_datetime(draw_date)
            if df["draw_date"].dt.date.eq(dts.date()).any() and not overwrite_existing:
                st.error("Date exists. Tick replace.")
            else:
                new_row = pd.DataFrame([{"draw_date":dts,"n1":new_nums[0],"n2":new_nums[1],"n3":new_nums[2],"n4":new_nums[3],"n5":new_nums[4]}])
                updated = pd.concat([df,new_row], ignore_index=True)
                if overwrite_existing: updated = updated.drop_duplicates("draw_date", keep="last")
                updated = updated.sort_values("draw_date").reset_index(drop=True)
                updated.to_csv(DATA_PATH, index=False)
                st.success(f"Saved: {dts.date()} — {'-'.join(map(str,new_nums))}")
                st.info("Refresh the app to recalculate.")
    st.download_button("Download history CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="daily_lotto_history.csv", mime="text/csv", use_container_width=True)

st.markdown(f"<div class='edge-title'>{page}</div>", unsafe_allow_html=True)

if page == "Dashboard":
    st.markdown("<div class='edge-subtitle'>Model overview and current performance snapshot.</div>", unsafe_allow_html=True)
    metric_cards(df, features)
    st.divider()
    col1,col2 = st.columns([2,1])
    with col1:
        st.subheader("Top 10 Combinations")
        st.dataframe(combination_scores(df, 10), use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Top 3 Numbers")
        number_grid(number_scores(df).head(3))
        st.subheader("Over / Under")
        ou = over_under_prediction(df)
        st.metric(ou["prediction"], f'{ou["confidence"]}% confidence')

elif page == "Predictions":
    st.markdown("<div class='edge-subtitle'>Individual, pair, triplet, combination, and over/under outputs.</div>", unsafe_allow_html=True)
    ns_full = number_scores(df); top3 = ns_full.head(3); combo_top = combination_scores(df,1).iloc[0]; ou = over_under_prediction(df)
    c1,c2,c3,c4 = st.columns([1.15,1,1.15,1.05])
    with c1:
        card("Over / Under 92.5", f"<div class='edge-big-green'>{ou['prediction']}</div><div>Confidence: <b>{ou['confidence']}%</b></div><div class='edge-progress'><div style='width:{ou['confidence']}%;'></div></div><div style='color:var(--muted);font-size:.85rem;'>Over {ou['prob_over']}% · Under {ou['prob_under']}%</div>", f"10-draw avg sum: {ou['recent_avg_sum_10']}")
    with c2:
        pills = ''.join([f"<span class='edge-pill'>{int(n):02d}</span>" for n in top3['Number'].tolist()])
        card("Top 3 Numbers", f"<div>{pills}</div><div style='margin-top:.5rem;'>Avg score: <b>{round(top3['Score'].mean(),2)}</b></div>", "Individual model")
    with c3:
        pills = ''.join([f"<span class='edge-pill blue'>{int(n):02d}</span>" for n in combo_top['Combination'].split('-')])
        card("Best 5-Number Combo", f"<div>{pills}</div><div style='margin-top:.5rem;'>Score: <b>{combo_top['Score']}</b></div>", f"Sum {combo_top['Sum']} · {combo_top['Odd']} Odd / {combo_top['Even']} Even")
    with c4:
        card("Draw Structure", f"<div class='edge-big-blue'>{combo_top['Low']}L/{combo_top['High']}H</div><div>{combo_top['Odd']} Odd / {combo_top['Even']} Even</div>", "Based on best combo")
    st.divider()
    tab1,tab2,tab3,tab4 = st.tabs(["Individual", "Pairs", "Triplets", "Combinations"])
    with tab1:
        show_n = st.slider("How many individual numbers to show?", 3, 36, 10, key="ind_slider")
        left,right = st.columns([1.25,1])
        with left:
            st.subheader(f"Top {show_n} Individual Numbers")
            st.dataframe(ns_full.head(show_n), use_container_width=True, hide_index=True)
        with right:
            st.subheader("Individual Numbers Grid")
            number_grid(ns_full.head(min(show_n,20)))
    with tab2:
        show_pairs = st.slider("How many pairs to show?", 5, 50, 10, key="pair_slider")
        st.subheader(f"Top {show_pairs} Pair Predictions")
        st.dataframe(pair_predictions(df, show_pairs), use_container_width=True, hide_index=True)
    with tab3:
        show_triplets = st.slider("How many triplets to show?", 5, 50, 10, key="triplet_slider")
        st.subheader(f"Top {show_triplets} Triplet Predictions")
        st.dataframe(triplet_predictions(df, show_triplets), use_container_width=True, hide_index=True)
    with tab4:
        show_combos = st.slider("How many combinations to show?", 5, 50, 10, key="combo_slider")
        st.subheader(f"Top {show_combos} Combination Predictions")
        st.dataframe(combination_scores(df, show_combos), use_container_width=True, hide_index=True)

elif page == "Analytics":
    st.markdown("<div class='edge-subtitle'>Historical patterns and model inputs.</div>", unsafe_allow_html=True)
    metric_cards(df, features)
    tab1,tab2,tab3,tab4 = st.tabs(["Numbers", "Pairs", "Triplets", "Structures"])
    with tab1:
        ns=number_scores(df); st.dataframe(ns, use_container_width=True, hide_index=True); st.bar_chart(ns.set_index("Number")["Score"])
    with tab2: st.dataframe(pair_predictions(df,100), use_container_width=True, hide_index=True)
    with tab3: st.dataframe(triplet_predictions(df,100), use_container_width=True, hide_index=True)
    with tab4:
        st.dataframe(features["structure"].value_counts().reset_index(), use_container_width=True, hide_index=True)
        st.line_chart(features.set_index("draw_date")["sum"])

elif page == "Back-testing":
    st.markdown("<div class='edge-subtitle'>Walk-forward checks. This is where weak ideas get exposed.</div>", unsafe_allow_html=True)
    with st.spinner("Running Over/Under back-test..."):
        ou_bt = walk_forward_over_under_backtest(df)
    acc = round(float(ou_bt["correct"].mean())*100,2) if len(ou_bt) else 0
    st.metric("Over/Under 92.5 accuracy", f"{acc}%")
    st.dataframe(ou_bt.sort_values("draw_date", ascending=False).head(100), use_container_width=True, hide_index=True)
    st.divider()
    with st.spinner("Running Top 3 back-test..."):
        t3_bt = simple_top3_backtest(df)
    c1,c2 = st.columns(2)
    c1.metric("Top 3 avg matches", round(float(t3_bt["matches"].mean()),3))
    c2.metric("Top 3 2+ hit rate", f"{round(float((t3_bt['matches']>=2).mean())*100,2)}%")
    st.dataframe(t3_bt.sort_values("draw_date", ascending=False).head(100), use_container_width=True, hide_index=True)

elif page == "History / Updates":
    st.markdown("<div class='edge-subtitle'>Manage the historical dataset.</div>", unsafe_allow_html=True)
    metric_cards(df, features)
    st.info("On free Streamlit hosting, manual changes may not persist forever. Download the updated CSV and commit it to GitHub.")
    uploaded = st.file_uploader("Bulk replace history CSV", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            required = {"draw_date","n1","n2","n3","n4","n5"}
            if not required.issubset(set(uploaded_df.columns)):
                st.error("CSV must contain draw_date,n1,n2,n3,n4,n5")
            else:
                uploaded_df["draw_date"] = pd.to_datetime(uploaded_df["draw_date"])
                for c in NUMBER_COLS: uploaded_df[c] = pd.to_numeric(uploaded_df[c], errors="raise").astype(int)
                uploaded_df = uploaded_df.sort_values("draw_date").drop_duplicates("draw_date", keep="last")
                if st.button("Replace app history with uploaded CSV"):
                    uploaded_df.to_csv(DATA_PATH, index=False); st.success("CSV replaced. Refresh app.")
        except Exception as e: st.error(f"Could not process CSV: {e}")
    st.subheader("Historical draws")
    st.dataframe(features.sort_values("draw_date", ascending=False), use_container_width=True, hide_index=True)
