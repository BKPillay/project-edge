from pathlib import Path
import json

import pandas as pd
import streamlit as st

st.set_page_config(page_title="EDGE AI v2", page_icon="🎯", layout="wide")

DATA_PATH = Path("data/daily_lotto_history.csv")
OUTPUT_DIR = Path("outputs")

st.markdown("""
<style>
header[data-testid="stHeader"] { background: rgba(0,0,0,0) !important; height: 0rem !important; }
div[data-testid="stToolbar"], div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] { visibility: hidden !important; height: 0rem !important; display: none !important; }
#MainMenu, footer { visibility: hidden !important; }
.stApp { background: linear-gradient(180deg, #07111f 0%, #050b14 100%) !important; color: #eef4ff !important; }
[data-testid="stSidebar"] { background: #07111f !important; border-right: 1px solid rgba(120,145,180,.22) !important; }
[data-testid="stSidebar"] * { color: #eef4ff !important; }
.block-container { padding-top: 1.2rem !important; max-width: 1450px !important; }
.edge-title { font-size: 2.1rem; font-weight: 900; margin-bottom: .1rem; color: #eef4ff; }
.edge-ai { color: #9b5cff; }
.edge-subtitle { color: #9fb0c7; margin-bottom: 1rem; }
.edge-card { background: rgba(14,31,56,.92); border: 1px solid rgba(120,145,180,.22); border-radius: 14px; padding: 1rem 1.15rem; min-height: 105px; }
.edge-label { color:#9fb0c7; font-size:.86rem; margin-bottom:.25rem; }
.edge-big { color:#39d975; font-size:1.8rem; font-weight:900; }
.edge-pill { display:inline-block; border:1px solid rgba(155,92,255,.85); border-radius:999px; padding:.32rem .55rem; margin:.12rem .15rem .12rem 0; min-width:38px; text-align:center; background:rgba(155,92,255,.09); font-weight:800; color:#eef4ff; }
.edge-pill.blue { border-color:rgba(63,140,255,.85); background:rgba(63,140,255,.09); }
</style>
""", unsafe_allow_html=True)


def read_csv(name):
    path = OUTPUT_DIR / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json(name):
    path = OUTPUT_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def card(title, value, footer=""):
    st.markdown(
        f"""
        <div class="edge-card">
            <div class="edge-label">{title}</div>
            <div class="edge-big">{value}</div>
            <div style="color:#9fb0c7;font-size:.82rem;margin-top:.5rem;">{footer}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pills(numbers, colour=""):
    cls = "edge-pill blue" if colour == "blue" else "edge-pill"
    return "".join([f"<span class='{cls}'>{int(n):02d}</span>" for n in numbers])


def outputs_ready():
    required = [
        "number_predictions.csv",
        "pair_predictions.csv",
        "triplet_predictions.csv",
        "combination_predictions.csv",
        "over_under.json",
        "features.csv",
        "latest_draw_ball_review.csv",
    ]
    return all((OUTPUT_DIR / f).exists() for f in required)


with st.sidebar:
    st.markdown("<div class='edge-title'>EDGE <span class='edge-ai'>AI</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>v2 Full-History Platform</div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Dashboard", "Predictions", "Latest Draw Review", "Analytics", "Back-testing", "History / Updates"],
        label_visibility="collapsed",
    )

    st.divider()
    if DATA_PATH.exists():
        st.download_button(
            "Download history CSV",
            data=DATA_PATH.read_bytes(),
            file_name="daily_lotto_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

if not outputs_ready():
    st.error("Model outputs are missing. Run `python scripts\\run_model_update.py`, commit the outputs folder, and push.")
    st.stop()

features = read_csv("features.csv")
numbers = read_csv("number_predictions.csv")
pairs = read_csv("pair_predictions.csv")
triplets = read_csv("triplet_predictions.csv")
combos = read_csv("combination_predictions.csv")
latest_review = read_csv("latest_draw_ball_review.csv")
latest_pair_review = read_csv("latest_draw_pair_review.csv")
latest_triplet_review = read_csv("latest_draw_triplet_review.csv")
ou = read_json("over_under.json")
summary = read_json("run_summary.json")
backtest = read_json("backtest_summary.json")
structural = read_json("structural_summary.json")

if page == "Dashboard":
    st.markdown("<div class='edge-title'>Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Output-driven dashboard using the full history.</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Draws Loaded", f"{summary.get('draws_loaded', len(features)):,}")
    with c2:
        card("Latest Draw", summary.get("latest_draw_date", "n/a"), f"Draw #{summary.get('latest_draw_number', 'n/a')}")
    with c3:
        card("Latest Sum", str(backtest.get("latest_sum", "n/a")))
    with c4:
        card("Over / Under", ou.get("prediction", "n/a"), f'{ou.get("confidence", "n/a")}% confidence')

    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Top 3 Numbers")
        st.markdown(pills(numbers.head(3)["number"].tolist()), unsafe_allow_html=True)
    with col2:
        st.subheader("Top 10 Numbers")
        st.dataframe(numbers.head(10), use_container_width=True, hide_index=True)

    st.subheader("Top 10 Combinations")
    st.dataframe(combos.head(10), use_container_width=True, hide_index=True)


elif page == "Predictions":
    st.markdown("<div class='edge-title'>Predictions</div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Full-history rankings. Sliders only slice saved outputs.</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Over / Under 92.5", ou.get("prediction", "n/a"), f'Confidence: {ou.get("confidence", "n/a")}%')
    with c2:
        st.markdown("<div class='edge-card'><div class='edge-label'>Top 3 Numbers</div>" + pills(numbers.head(3)["number"].tolist()) + f"<div style='color:#9fb0c7;font-size:.82rem;margin-top:.5rem;'>Avg score: {round(numbers.head(3)['final_score'].mean(),2)}</div></div>", unsafe_allow_html=True)
    with c3:
        combo1 = combos.iloc[0]
        st.markdown("<div class='edge-card'><div class='edge-label'>Best Combo</div>" + pills(str(combo1["combination"]).split("-"), "blue") + f"<div style='color:#9fb0c7;font-size:.82rem;margin-top:.5rem;'>Score: {combo1['score']} · Sum: {combo1['sum']}</div></div>", unsafe_allow_html=True)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_n = st.slider("Individual numbers", 3, min(36, len(numbers)), 10)
    with col2:
        show_pairs = st.slider("Pairs", 5, min(100, len(pairs)), 10)
    with col3:
        show_triplets = st.slider("Triplets", 5, min(100, len(triplets)), 10)
    with col4:
        show_combos = st.slider("Combinations", 5, min(100, len(combos)), 10)

    tab1, tab2, tab3, tab4 = st.tabs(["Individual", "Pairs", "Triplets", "Combinations"])
    with tab1:
        st.dataframe(numbers.head(show_n), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(pairs.head(show_pairs), use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(triplets.head(show_triplets), use_container_width=True, hide_index=True)
    with tab4:
        st.dataframe(combos.head(show_combos), use_container_width=True, hide_index=True)


elif page == "Latest Draw Review":
    st.markdown("<div class='edge-title'>Latest Draw Review</div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>How the latest draw looked to the model before the draw happened.</div>", unsafe_allow_html=True)

    latest_numbers = summary.get("latest_numbers", [])
    c1, c2, c3 = st.columns(3)
    with c1:
        card("Latest Draw", summary.get("latest_draw_date", "n/a"), f"Draw #{summary.get('latest_draw_number', 'n/a')}")
    with c2:
        st.markdown("<div class='edge-card'><div class='edge-label'>Latest Numbers</div>" + pills(latest_numbers) + f"<div style='color:#9fb0c7;font-size:.82rem;margin-top:.5rem;'>Sum: {backtest.get('latest_sum', 'n/a')}</div></div>", unsafe_allow_html=True)
    with c3:
        card("Latest O/U", backtest.get("latest_over_under", "n/a"))

    st.subheader("Ball Review")
    st.dataframe(latest_review, use_container_width=True, hide_index=True)

    st.subheader("Pairs from Latest Draw")
    st.dataframe(latest_pair_review, use_container_width=True, hide_index=True)

    st.subheader("Triplets from Latest Draw")
    st.dataframe(latest_triplet_review, use_container_width=True, hide_index=True)


elif page == "Analytics":
    st.markdown("<div class='edge-title'>Analytics</div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Numbers", "Draw Features", "Structural Summary"])
    with tab1:
        st.dataframe(numbers, use_container_width=True, hide_index=True)
        st.bar_chart(numbers.set_index("number")["final_score"])
    with tab2:
        st.dataframe(features.tail(100), use_container_width=True, hide_index=True)
        st.line_chart(features.set_index("draw_date")["sum"])
    with tab3:
        st.json(structural)


elif page == "Back-testing":
    st.markdown("<div class='edge-title'>Back-testing</div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Sampled walk-forward validation generated offline.</div>", unsafe_allow_html=True)
    st.json(backtest)


elif page == "History / Updates":
    st.markdown("<div class='edge-title'>History / Updates</div>", unsafe_allow_html=True)
    st.info("To update permanently: edit data/daily_lotto_history.csv, run the model update script, commit data + outputs, and push.")
    show_all = st.checkbox("Show full feature history", value=False)
    st.dataframe(features if show_all else features.tail(30), use_container_width=True, hide_index=True)
