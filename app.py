
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import streamlit as st

from src.edge_model import (
    NUMBER_COLS,
    add_draw_features,
    load_history,
    number_scores,
    over_under_prediction,
    pair_predictions,
    triplet_predictions,
)

st.set_page_config(page_title="EDGE AI", page_icon="🎯", layout="wide")

DATA_PATH = Path("data/daily_lotto_history.csv")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #07111f 0%, #050b14 100%);
    color: #eef4ff;
}
[data-testid="stSidebar"] {
    background: #07111f;
    border-right: 1px solid rgba(120,145,180,.22);
}
.block-container { padding-top: 1.2rem; max-width: 1450px; }
.edge-title { font-size: 2.1rem; font-weight: 900; margin-bottom: .1rem; }
.edge-ai { color: #9b5cff; }
.edge-subtitle { color: #9fb0c7; margin-bottom: 1rem; }
.edge-card {
    background: rgba(14,31,56,.92);
    border: 1px solid rgba(120,145,180,.22);
    border-radius: 14px;
    padding: 1rem 1.15rem;
    min-height: 105px;
}
.edge-label { color:#9fb0c7; font-size:.86rem; margin-bottom:.25rem; }
.edge-big { color:#39d975; font-size:1.8rem; font-weight:900; }
.edge-pill {
    display:inline-block;
    border:1px solid rgba(155,92,255,.85);
    border-radius:999px;
    padding:.32rem .55rem;
    margin:.12rem .15rem .12rem 0;
    min-width:38px;
    text-align:center;
    background:rgba(155,92,255,.09);
    font-weight:800;
}
.edge-pill.blue {
    border-color:rgba(63,140,255,.85);
    background:rgba(63,140,255,.09);
}
</style>
""", unsafe_allow_html=True)


def card(title: str, value: str, footer: str = ""):
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


@st.cache_data(show_spinner=False)
def load_base(path_str: str):
    df = load_history(path_str)
    features = add_draw_features(df)
    return df, features


@st.cache_data(show_spinner=False)
def load_light_predictions(path_str: str):
    df = load_history(path_str)
    ns = number_scores(df)
    ou = over_under_prediction(df)
    return ns, ou


def fast_combos(numbers_df: pd.DataFrame, top_n: int = 10, pool_size: int = 13) -> pd.DataFrame:
    # Fast enough for Streamlit: C(13,5)=1287 combinations.
    pool = numbers_df.head(pool_size).copy()
    score_map = dict(zip(pool["Number"].astype(int), pool["Score"].astype(float)))
    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
    rows = []

    for combo in itertools.combinations(sorted(score_map.keys()), 5):
        total = sum(combo)
        odd = sum(n % 2 for n in combo)
        low = sum(n <= 18 for n in combo)
        prime = sum(n in primes for n in combo)

        bonus = 0
        bonus += 5 if 65 <= total <= 120 else 0
        bonus += 4 if odd in (2, 3) else 0
        bonus += 4 if low in (2, 3) else 0

        score = round(float(np.mean([score_map[n] for n in combo])) + bonus, 2)
        rows.append({
            "Rank": None,
            "Combination": "-".join(map(str, combo)),
            "Score": score,
            "Sum": total,
            "Odd": odd,
            "Even": 5 - odd,
            "Low": low,
            "High": 5 - low,
            "Prime": prime,
        })

    out = pd.DataFrame(rows).sort_values(["Score", "Combination"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    out["Rank"] = out.index + 1
    return out


def latest_review_light(df: pd.DataFrame, numbers_df: pd.DataFrame, ou: dict):
    if len(df) < 2:
        return None

    latest = df.iloc[-1]
    nums = sorted([int(latest[c]) for c in NUMBER_COLS])
    actual_sum = sum(nums)
    actual_ou = "Over 92.5" if actual_sum > 92.5 else "Under 92.5"

    score_map = numbers_df.set_index("Number").to_dict(orient="index")
    rows = []
    for n in nums:
        info = score_map.get(n)
        if info is None:
            continue
        rank = int(info["Rank"])
        if rank <= 3:
            insight = "Elite pre-draw pick"
        elif rank <= 10:
            insight = "Strong pre-draw pick"
        elif rank <= 20:
            insight = "Mid-ranked pick"
        else:
            insight = "Model underweighted this ball"
        rows.append({
            "Ball": n,
            "Current Rank": rank,
            "Model Score": info["Score"],
            "Last 20": info.get("Last 20"),
            "Last 50": info.get("Last 50"),
            "Gap": info.get("Gap"),
            "Insight": insight,
        })

    return {
        "date": str(latest["draw_date"].date()),
        "numbers": nums,
        "actual_sum": actual_sum,
        "actual_ou": actual_ou,
        "ou_prediction": ou["prediction"],
        "ou_correct": ou["prediction"] == actual_ou,
        "ball_review": pd.DataFrame(rows),
    }


if not DATA_PATH.exists():
    st.error("Missing data/daily_lotto_history.csv")
    st.stop()

df, features = load_base(str(DATA_PATH))
numbers_df, ou = load_light_predictions(str(DATA_PATH))

with st.sidebar:
    st.markdown("<div class='edge-title'>EDGE <span class='edge-ai'>AI</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Stable Mode</div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Dashboard", "Predictions", "Latest Draw Review", "Analytics", "History / Updates"],
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("➕ Add Draw")
    draw_date = st.date_input("Draw date")
    cols = st.columns(5)
    nums = [cols[i].number_input(f"N{i+1}", min_value=1, max_value=36, value=i+1, step=1) for i in range(5)]
    overwrite_existing = st.checkbox("Replace same date", value=True)

    if st.button("Save draw", use_container_width=True):
        new_nums = sorted([int(n) for n in nums])
        if len(set(new_nums)) != 5:
            st.error("The 5 numbers must be unique.")
        else:
            draw_date_ts = pd.to_datetime(draw_date)
            existing_date = df["draw_date"].dt.date.eq(draw_date_ts.date()).any()
            if existing_date and not overwrite_existing:
                st.error("Date exists. Tick replace.")
            else:
                new_row = pd.DataFrame([{
                    "draw_date": draw_date_ts,
                    "n1": new_nums[0],
                    "n2": new_nums[1],
                    "n3": new_nums[2],
                    "n4": new_nums[3],
                    "n5": new_nums[4],
                }])
                updated = pd.concat([df, new_row], ignore_index=True)
                if overwrite_existing:
                    updated = updated.drop_duplicates(subset=["draw_date"], keep="last")
                updated = updated.sort_values("draw_date").reset_index(drop=True)
                updated.to_csv(DATA_PATH, index=False)
                st.cache_data.clear()
                st.success("Saved. Refresh the app.")

    st.download_button(
        "Download history CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="daily_lotto_history.csv",
        mime="text/csv",
        use_container_width=True,
    )


if page == "Dashboard":
    st.markdown("<div class='edge-title'>Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Stable fast summary.</div>", unsafe_allow_html=True)

    latest = features.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Draws Loaded", f"{len(df):,}")
    with c2:
        card("Latest Draw", str(latest["draw_date"].date()))
    with c3:
        card("Latest Sum", str(int(latest["sum"])))
    with c4:
        card("Over / Under", ou["prediction"], f'{ou["confidence"]}% confidence')

    st.divider()
    st.subheader("Top 3 Numbers")
    top3 = numbers_df.head(3)
    st.markdown(pills(top3["Number"].tolist()), unsafe_allow_html=True)

    st.subheader("Top 10 Fast Combinations")
    st.dataframe(fast_combos(numbers_df, 10), use_container_width=True, hide_index=True)


elif page == "Predictions":
    st.markdown("<div class='edge-title'>Predictions</div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Fast render first. Pair/triplet tables load only when requested.</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Over / Under 92.5", ou["prediction"], f'Confidence: {ou["confidence"]}% · Over {ou["prob_over"]}% / Under {ou["prob_under"]}%')
    with c2:
        top3 = numbers_df.head(3)
        st.markdown("<div class='edge-card'><div class='edge-label'>Top 3 Numbers</div>" + pills(top3["Number"].tolist()) + f"<div style='color:#9fb0c7;font-size:.82rem;margin-top:.5rem;'>Avg score: {round(top3['Score'].mean(),2)}</div></div>", unsafe_allow_html=True)
    with c3:
        combo1 = fast_combos(numbers_df, 1).iloc[0]
        st.markdown("<div class='edge-card'><div class='edge-label'>Best Fast Combo</div>" + pills(combo1["Combination"].split("-"), "blue") + f"<div style='color:#9fb0c7;font-size:.82rem;margin-top:.5rem;'>Score: {combo1['Score']} · Sum: {combo1['Sum']}</div></div>", unsafe_allow_html=True)

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Individual", "Pairs", "Triplets", "Combinations"])

    with tab1:
        show_n = st.slider("How many individual numbers?", 3, 36, 10)
        st.dataframe(numbers_df.head(show_n), use_container_width=True, hide_index=True)

    with tab2:
        show_pairs = st.slider("How many pairs?", 5, 50, 10)
        st.info("Pairs are calculated only after pressing the button.")
        if st.button("Load pair predictions"):
            with st.spinner("Calculating pairs..."):
                st.dataframe(pair_predictions(df, show_pairs), use_container_width=True, hide_index=True)

    with tab3:
        show_triplets = st.slider("How many triplets?", 5, 50, 10)
        st.info("Triplets are calculated only after pressing the button.")
        if st.button("Load triplet predictions"):
            with st.spinner("Calculating triplets..."):
                st.dataframe(triplet_predictions(df, show_triplets), use_container_width=True, hide_index=True)

    with tab4:
        show_combos = st.slider("How many combinations?", 5, 50, 10)
        st.caption("Fast candidate-based ranking. Not the full 376,992 combination audit.")
        st.dataframe(fast_combos(numbers_df, show_combos), use_container_width=True, hide_index=True)


elif page == "Latest Draw Review":
    st.markdown("<div class='edge-title'>Latest Draw Review</div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Lightweight review of the latest draw.</div>", unsafe_allow_html=True)

    review = latest_review_light(df, numbers_df, ou)
    if review is None:
        st.warning("Need at least two draws.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            card("Draw Date", review["date"])
        with c2:
            st.markdown("<div class='edge-card'><div class='edge-label'>Winning Numbers</div>" + pills(review["numbers"]) + f"<div style='color:#9fb0c7;font-size:.82rem;margin-top:.5rem;'>Sum: {review['actual_sum']}</div></div>", unsafe_allow_html=True)
        with c3:
            card("O/U Result", "Correct" if review["ou_correct"] else "Missed", f'Predicted: {review["ou_prediction"]}')

        st.subheader("Ball Review")
        st.dataframe(review["ball_review"], use_container_width=True, hide_index=True)


elif page == "Analytics":
    st.markdown("<div class='edge-title'>Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Lightweight analytics.</div>", unsafe_allow_html=True)

    st.subheader("Number Scores")
    st.dataframe(numbers_df, use_container_width=True, hide_index=True)
    st.bar_chart(numbers_df.set_index("Number")["Score"])

    st.subheader("Sum Trend")
    st.line_chart(features.set_index("draw_date")["sum"])


elif page == "History / Updates":
    st.markdown("<div class='edge-title'>History / Updates</div>", unsafe_allow_html=True)
    st.info("Only the latest 30 rows show by default for speed.")
    show_all = st.checkbox("Show full history", value=False)
    if show_all:
        st.dataframe(features.sort_values("draw_date", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.dataframe(features.sort_values("draw_date", ascending=False).head(30), use_container_width=True, hide_index=True)
