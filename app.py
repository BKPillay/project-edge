from pathlib import Path
import json
import sys
import subprocess

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

# EDGE_AI_V2_3_DARK_THEME_FIX
st.markdown("""
<style>
header[data-testid="stHeader"] {
    background: rgba(0,0,0,0) !important;
    height: 0rem !important;
}
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"] {
    visibility: hidden !important;
    height: 0rem !important;
    display: none !important;
}
#MainMenu, footer {
    visibility: hidden !important;
}
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.block-container {
    background: linear-gradient(180deg, #07111f 0%, #050b14 100%) !important;
    color: #eef4ff !important;
}
[data-testid="stSidebar"] {
    background: #07111f !important;
    border-right: 1px solid rgba(120,145,180,.22) !important;
}
[data-testid="stSidebar"] * {
    color: #eef4ff !important;
}
[data-testid="stForm"] {
    background: rgba(14,31,56,.72) !important;
    border: 1px solid rgba(120,145,180,.22) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}
[data-testid="stForm"] * {
    color: #eef4ff !important;
}
.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stSelectbox div,
.stMultiSelect div,
div[data-baseweb="input"],
div[data-baseweb="select"],
div[data-baseweb="datepicker"] input {
    background-color: #0b182b !important;
    color: #eef4ff !important;
    border-color: rgba(120,145,180,.35) !important;
}
div[data-baseweb="popover"],
div[data-baseweb="menu"],
div[data-baseweb="calendar"] {
    background-color: #0b182b !important;
    color: #eef4ff !important;
    border: 1px solid rgba(120,145,180,.35) !important;
}
div[data-baseweb="calendar"] * {
    background-color: #0b182b !important;
    color: #eef4ff !important;
}
.stButton > button,
.stDownloadButton > button,
button[kind="primary"],
button[kind="secondary"] {
    background: rgba(14,31,56,.92) !important;
    color: #eef4ff !important;
    border: 1px solid rgba(120,145,180,.35) !important;
    border-radius: 10px !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
button[kind="primary"]:hover,
button[kind="secondary"]:hover {
    border-color: #9b5cff !important;
    color: #ffffff !important;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #9fb0c7 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #eef4ff !important;
    border-bottom-color: #9b5cff !important;
}
[data-testid="stDataFrame"] {
    background: #0b182b !important;
    border: 1px solid rgba(120,145,180,.22) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] * {
    color: #eef4ff !important;
}
[data-testid="stDataFrame"] div {
    background-color: transparent !important;
}
[data-testid="stDataFrame"] [role="grid"] {
    background-color: #0b182b !important;
}
[data-testid="stDataFrame"] [role="columnheader"] {
    background-color: #132947 !important;
    color: #c7d5ea !important;
    border-color: rgba(120,145,180,.22) !important;
}
[data-testid="stDataFrame"] [role="gridcell"] {
    background-color: #0b182b !important;
    color: #eef4ff !important;
    border-color: rgba(120,145,180,.12) !important;
}
[data-testid="stDataFrame"] [role="row"]:nth-child(even) [role="gridcell"] {
    background-color: #0e2038 !important;
}
table {
    background: #0b182b !important;
    color: #eef4ff !important;
}
thead tr th {
    background: #132947 !important;
    color: #c7d5ea !important;
}
tbody tr td {
    background: #0b182b !important;
    color: #eef4ff !important;
}
[data-testid="stAlert"] {
    background: rgba(14,31,56,.92) !important;
    color: #eef4ff !important;
    border: 1px solid rgba(120,145,180,.22) !important;
}
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


# EDGE_AI_V2_3_SAVE_DRAW_HELPERS
def load_history_for_save():
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=["draw_number", "draw_date", "n1", "n2", "n3", "n4", "n5"])
    df = pd.read_csv(DATA_PATH)
    if "draw_number" not in df.columns:
        df.insert(0, "draw_number", range(1, len(df) + 1))
    df["draw_number"] = pd.to_numeric(df["draw_number"], errors="coerce").astype("Int64")
    df["draw_date"] = pd.to_datetime(df["draw_date"], errors="coerce")
    for col in ["n1", "n2", "n3", "n4", "n5"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def save_new_draw_to_history(draw_date, numbers_to_save):
    df_hist = load_history_for_save()
    draw_date_ts = pd.to_datetime(draw_date)
    nums = sorted([int(n) for n in numbers_to_save])

    if len(set(nums)) != 5:
        return False, "Enter 5 unique numbers from 1 to 36."

    if any(n < 1 or n > 36 for n in nums):
        return False, "All numbers must be between 1 and 36."

    if len(df_hist):
        existing_date = df_hist["draw_date"].dt.date.eq(draw_date_ts.date()).any()
        if existing_date:
            return False, f"Draw date {draw_date_ts.date()} already exists in history."

        combo_cols = ["n1", "n2", "n3", "n4", "n5"]
        sorted_existing = df_hist[combo_cols].apply(lambda r: "-".join(map(str, sorted([int(x) for x in r]))), axis=1)
        combo_key = "-".join(map(str, nums))
        if combo_key in set(sorted_existing):
            return False, f"This exact combination already exists in history: {combo_key}"

        next_draw_number = int(df_hist["draw_number"].max()) + 1
    else:
        next_draw_number = 1

    new_row = pd.DataFrame([{
        "draw_number": next_draw_number,
        "draw_date": draw_date_ts.strftime("%Y-%m-%d"),
        "n1": nums[0],
        "n2": nums[1],
        "n3": nums[2],
        "n4": nums[3],
        "n5": nums[4],
    }])

    out = pd.concat([df_hist, new_row], ignore_index=True)
    out["draw_date"] = pd.to_datetime(out["draw_date"]).dt.strftime("%Y-%m-%d")
    out = out.sort_values(["draw_number", "draw_date"]).reset_index(drop=True)
    out.to_csv(DATA_PATH, index=False)

    return True, f"Saved draw #{next_draw_number}: {draw_date_ts.date()} — {'-'.join(map(str, nums))}"


def run_model_update_after_save():
    script_path = Path("scripts/run_model_update.py")
    if not script_path.exists():
        return False, "Saved the draw, but scripts/run_model_update.py was not found. Run the model update manually."

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception as exc:
        return False, f"Saved the draw, but model update failed to start: {exc}"

    if result.returncode != 0:
        return False, "Saved the draw, but model update failed. Check terminal logs."

    return True, "Model outputs regenerated successfully."


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
    st.markdown("<div class='edge-subtitle'>Preview, save, and then review the latest draw without changing the project structure.</div>", unsafe_allow_html=True)

    st.subheader("Preview / Save New Draw")

    with st.form("preview_save_new_draw_form"):
        preview_date = st.date_input("Draw date")
        cols = st.columns(5)
        preview_nums = [
            cols[0].number_input("N1", min_value=1, max_value=36, value=1, step=1),
            cols[1].number_input("N2", min_value=1, max_value=36, value=2, step=1),
            cols[2].number_input("N3", min_value=1, max_value=36, value=3, step=1),
            cols[3].number_input("N4", min_value=1, max_value=36, value=4, step=1),
            cols[4].number_input("N5", min_value=1, max_value=36, value=5, step=1),
        ]

        action_col1, action_col2 = st.columns(2)
        preview_clicked = action_col1.form_submit_button("Preview Ball Review")
        save_clicked = action_col2.form_submit_button("Save Draw + Update Model")

    preview_nums = sorted([int(n) for n in preview_nums])

    def render_preview(nums, draw_date):
        score_map = numbers.set_index("number").to_dict(orient="index")
        preview_rows = []

        for n in nums:
            info = score_map.get(n, {})
            rank = info.get("rank")
            final_score = info.get("final_score")
            freq = info.get("frequency_score")
            momentum = info.get("momentum_score")
            gap_score = info.get("gap_score")
            current_gap = info.get("current_gap")
            bucket = info.get("bucket")

            if rank is None:
                insight = "Not found in model output"
            elif rank <= 3:
                insight = "Elite current model pick"
            elif rank <= 10:
                insight = "Strong current model pick"
            elif rank <= 20:
                insight = "Mid-ranked current model pick"
            else:
                insight = "Model currently underweights this ball"

            preview_rows.append({
                "ball": n,
                "current_rank": rank,
                "final_score": final_score,
                "frequency_score": freq,
                "momentum_score": momentum,
                "gap_score": gap_score,
                "current_gap": current_gap,
                "bucket": bucket,
                "insight": insight,
            })

        preview_sum = sum(nums)
        preview_ou = "Over 92.5" if preview_sum > 92.5 else "Under 92.5"
        top3_hits = len(set(nums) & set(numbers.head(3)["number"].astype(int).tolist()))
        top10_hits = len(set(nums) & set(numbers.head(10)["number"].astype(int).tolist()))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            card("Preview Date", str(draw_date))
        with c2:
            st.markdown("<div class='edge-card'><div class='edge-label'>Entered Numbers</div>" + pills(nums) + f"<div style='color:#9fb0c7;font-size:.82rem;margin-top:.5rem;'>Sum: {preview_sum}</div></div>", unsafe_allow_html=True)
        with c3:
            card("O/U Result", preview_ou, f"Current model says: {ou.get('prediction', 'n/a')}")
        with c4:
            card("Model Capture", f"{top3_hits}/3", f"Top 10 hits: {top10_hits}/5")

        st.subheader("Preview Ball Review")
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    if preview_clicked or save_clicked:
        if len(set(preview_nums)) != 5:
            st.error("Enter 5 unique numbers from 1 to 36.")
        else:
            render_preview(preview_nums, preview_date)

    if save_clicked:
        if len(set(preview_nums)) != 5:
            st.stop()

        ok, message = save_new_draw_to_history(preview_date, preview_nums)
        if not ok:
            st.error(message)
        else:
            st.success(message)

            with st.spinner("Updating model outputs..."):
                update_ok, update_message = run_model_update_after_save()

            if update_ok:
                st.success(update_message)
                st.info("Commit and push the updated data/outputs/app.py files so Streamlit Cloud persists the changes.")
            else:
                st.warning(update_message)

    st.divider()

    st.subheader("Latest Saved Draw Review")

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
    if len(latest_pair_review):
        st.dataframe(latest_pair_review, use_container_width=True, hide_index=True)
    else:
        st.info("No latest draw pairs appeared in the saved top pair review output.")

    st.subheader("Triplets from Latest Draw")
    if len(latest_triplet_review):
        st.dataframe(latest_triplet_review, use_container_width=True, hide_index=True)
    else:
        st.info("No latest draw triplets appeared in the saved top triplet review output.")


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
