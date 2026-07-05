
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
    background:
      radial-gradient(circle at top left, rgba(95,68,255,.14), transparent 28%),
      radial-gradient(circle at top right, rgba(30,144,255,.10), transparent 28%),
      linear-gradient(180deg, #07111f 0%, #050b14 100%);
    color: #eef4ff;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #081422 0%, #050b14 100%);
    border-right: 1px solid rgba(120,145,180,.22);
}
.block-container { padding-top: 1.25rem; max-width: 1500px; }
.edge-title { font-size: 2.1rem; font-weight: 900; margin-bottom: .1rem; }
.edge-ai { color: #9b5cff; }
.edge-subtitle { color: #9fb0c7; margin-bottom: 1rem; }
.edge-card {
    background: linear-gradient(180deg, rgba(14,31,56,.92), rgba(7,17,31,.92));
    border: 1px solid rgba(120,145,180,.22);
    border-radius: 14px;
    padding: 1.0rem 1.15rem;
    box-shadow: 0 12px 30px rgba(0,0,0,.20);
    min-height: 115px;
}
.edge-label { color:#9fb0c7; font-size:.86rem; margin-bottom:.25rem; }
.edge-big-green { color:#39d975; font-size:1.9rem; font-weight:900; line-height:1.05; }
.edge-big-blue { color:#3f8cff; font-size:1.45rem; font-weight:800; }
.edge-pill {
    display:inline-block; border:1px solid rgba(155,92,255,.85); border-radius:999px;
    padding:.32rem .55rem; margin:.12rem .15rem .12rem 0; min-width:38px;
    text-align:center; background:rgba(155,92,255,.09); font-weight:800;
}
.edge-pill.blue { border-color:rgba(63,140,255,.85); background:rgba(63,140,255,.09); }
.edge-progress { height:8px; border-radius:999px; background:rgba(255,255,255,.12); overflow:hidden; margin:.5rem 0; }
.edge-progress > div { height:100%; border-radius:999px; background:linear-gradient(90deg,#39d975,#9b5cff); }
.edge-grid { display:grid; grid-template-columns:repeat(5,minmax(50px,1fr)); gap:.6rem; margin-top:.5rem; }
.edge-number-bubble {
    border:1px solid #39d975; border-radius:999px; height:50px; width:50px;
    display:flex; align-items:center; justify-content:center; font-weight:900; margin:auto;
    background:rgba(57,217,117,.08);
}
.edge-score { text-align:center; color:#9fb0c7; font-size:.8rem; margin-top:.2rem; }
div[data-testid="stDataFrame"] { border:1px solid rgba(120,145,180,.22); border-radius:12px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)


def card(title: str, body: str, footer: str = ""):
    st.markdown(
        f"""
        <div class="edge-card">
          <div class="edge-label">{title}</div>
          {body}
          <div style="color:#9fb0c7;font-size:.82rem;margin-top:.6rem;">{footer}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def number_grid(numbers_df: pd.DataFrame):
    html = "<div class='edge-grid'>"
    for _, row in numbers_df.iterrows():
        html += f"""
        <div>
          <div class='edge-number-bubble'>{int(row['Number']):02d}</div>
          <div class='edge-score'>{row['Score']}</div>
        </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_base_data(path_str: str):
    df = load_history(path_str)
    features = add_draw_features(df)
    return df, features


@st.cache_data(show_spinner=False)
def compute_core_outputs(path_str: str):
    df = load_history(path_str)
    numbers = number_scores(df)
    ou = over_under_prediction(df)
    # Compute maximum table sizes once. Sliders only slice these results.
    pairs = pair_predictions(df, 50)
    triplets = triplet_predictions(df, 50)
    combos = fast_combos_from_numbers(numbers, 50)
    return numbers, pairs, triplets, combos, ou


def fast_combos_from_numbers(numbers_df: pd.DataFrame, top_n: int = 50, pool_size: int = 15) -> pd.DataFrame:
    pool = numbers_df.head(pool_size).copy()
    score_map = dict(zip(pool["Number"].astype(int), pool["Score"].astype(float)))
    rows = []
    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}

    for combo in itertools.combinations(sorted(score_map.keys()), 5):
        total = sum(combo)
        odd = sum(n % 2 for n in combo)
        low = sum(n <= 18 for n in combo)
        prime = sum(n in primes for n in combo)

        structure_bonus = 0
        structure_bonus += 5 if 65 <= total <= 120 else 0
        structure_bonus += 4 if odd in (2, 3) else 0
        structure_bonus += 4 if low in (2, 3) else 0

        avg_score = float(np.mean([score_map[n] for n in combo]))
        final_score = round(avg_score + structure_bonus, 2)

        rows.append({
            "Rank": None,
            "Combination": "-".join(map(str, combo)),
            "Score": final_score,
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


def latest_draw_review_light(df: pd.DataFrame, numbers_df: pd.DataFrame, pairs_df: pd.DataFrame, triplets_df: pd.DataFrame, ou: dict):
    if len(df) < 2:
        return None

    pre_numbers = numbers_df.copy()
    latest = df.iloc[-1]
    latest_numbers = sorted([int(latest[c]) for c in NUMBER_COLS])
    actual_sum = sum(latest_numbers)
    actual_ou = "Over 92.5" if actual_sum > 92.5 else "Under 92.5"

    nmap = pre_numbers.set_index("Number").to_dict(orient="index")

    ball_rows = []
    for n in latest_numbers:
        info = nmap.get(n)
        if info is None:
            continue
        rank = int(info["Rank"])
        if rank <= 3:
            insight = "Elite pre-draw pick"
        elif rank <= 10:
            insight = "Strong pre-draw pick"
        elif rank <= 20:
            insight = "Mid-table model pick"
        else:
            insight = "Model underweighted this ball"

        ball_rows.append({
            "Ball": n,
            "Pre-draw Rank": rank,
            "Model Score": float(info["Score"]),
            "Last 20": int(info["Last 20"]),
            "Last 50": int(info["Last 50"]),
            "Gap": int(info["Gap"]),
            "Insight": insight,
        })

    pair_set = set()
    for a, b in itertools.combinations(latest_numbers, 2):
        pair_set.add(f"{a}-{b}")

    triplet_set = set()
    for t in itertools.combinations(latest_numbers, 3):
        triplet_set.add("-".join(map(str, t)))

    pairs_hit = pairs_df[pairs_df["Pair"].isin(pair_set)].copy() if "Pair" in pairs_df.columns else pd.DataFrame()
    triplets_hit = triplets_df[triplets_df["Triplet"].isin(triplet_set)].copy() if "Triplet" in triplets_df.columns else pd.DataFrame()

    return {
        "draw_date": str(latest["draw_date"].date()),
        "numbers": latest_numbers,
        "actual_sum": actual_sum,
        "actual_ou": actual_ou,
        "ou_prediction": ou["prediction"],
        "ou_correct": ou["prediction"] == actual_ou,
        "top3_hits": len(set(latest_numbers) & set(numbers_df.head(3)["Number"].astype(int).tolist())),
        "ball_review": pd.DataFrame(ball_rows),
        "pairs_hit": pairs_hit,
        "triplets_hit": triplets_hit,
    }


if not DATA_PATH.exists():
    st.error("Missing data/daily_lotto_history.csv")
    st.stop()

df, features = load_base_data(str(DATA_PATH))
numbers_df, pairs_df, triplets_df, combos_df, ou = compute_core_outputs(str(DATA_PATH))

with st.sidebar:
    st.markdown("<div class='edge-title'>EDGE <span class='edge-ai'>AI</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Daily Lotto Intelligence</div>", unsafe_allow_html=True)

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
    st.markdown("<div class='edge-subtitle'>Fast summary from precomputed model outputs.</div>", unsafe_allow_html=True)

    latest = features.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Draws Loaded", f"<div class='edge-big-blue'>{len(df):,}</div>")
    with c2:
        card("Latest Draw", f"<div class='edge-big-blue'>{latest['draw_date'].date()}</div>")
    with c3:
        card("Latest Sum", f"<div class='edge-big-blue'>{int(latest['sum'])}</div>")
    with c4:
        card("Latest Structure", f"<div class='edge-big-blue'>{latest['odd_count']}O/{latest['even_count']}E</div>", f"{latest['low_count']} Low / {latest['high_count']} High")

    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Top 3 Numbers")
        number_grid(numbers_df.head(3))
    with col2:
        st.subheader("Over / Under")
        st.metric(ou["prediction"], f'{ou["confidence"]}% confidence')

    st.subheader("Top 10 Combinations")
    st.dataframe(combos_df.head(10), use_container_width=True, hide_index=True)


elif page == "Predictions":
    st.markdown("<div class='edge-title'>Predictions</div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Outputs load once. Sliders only slice precomputed results.</div>", unsafe_allow_html=True)

    control_cols = st.columns(4)
    with control_cols[0]:
        show_n = st.slider("Individual numbers", 3, 36, 10)
    with control_cols[1]:
        show_pairs = st.slider("Pairs", 5, 50, 10)
    with control_cols[2]:
        show_triplets = st.slider("Triplets", 5, 50, 10)
    with control_cols[3]:
        show_combos = st.slider("Combinations", 5, 50, 10)

    top3 = numbers_df.head(3)
    combo_top = combos_df.iloc[0]
    confidence_width = min(100, max(0, ou["confidence"]))

    st.divider()

    c1, c2, c3, c4 = st.columns([1.15, 1, 1.15, 1.05])
    with c1:
        card(
            "Over / Under 92.5",
            f"""
            <div class='edge-big-green'>{ou['prediction']}</div>
            <div>Confidence: <b>{ou['confidence']}%</b></div>
            <div class='edge-progress'><div style='width:{confidence_width}%;'></div></div>
            <div style='color:#9fb0c7;font-size:.85rem;'>Over {ou['prob_over']}% · Under {ou['prob_under']}%</div>
            """,
            f"10-draw avg sum: {ou.get('recent_avg_sum_10', 'n/a')}"
        )
    with c2:
        pills = "".join([f"<span class='edge-pill'>{int(n):02d}</span>" for n in top3["Number"].tolist()])
        card("Top 3 Numbers", f"<div>{pills}</div><div style='margin-top:.5rem;'>Avg score: <b>{round(top3['Score'].mean(),2)}</b></div>", "Individual model")
    with c3:
        nums = combo_top["Combination"].split("-")
        pills = "".join([f"<span class='edge-pill blue'>{int(n):02d}</span>" for n in nums])
        card("Best 5-Number Combo", f"<div>{pills}</div><div style='margin-top:.5rem;'>Score: <b>{combo_top['Score']}</b></div>", f"Sum {combo_top['Sum']}")
    with c4:
        card("Draw Structure", f"<div class='edge-big-blue'>{combo_top['Low']}L/{combo_top['High']}H</div><div>{combo_top['Odd']} Odd / {combo_top['Even']} Even</div>", "Based on best combo")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Individual", "Pairs", "Triplets", "Combinations"])

    with tab1:
        left, right = st.columns([1.3, 1])
        with left:
            st.subheader(f"Top {show_n} Individual Numbers")
            st.dataframe(numbers_df.head(show_n), use_container_width=True, hide_index=True)
        with right:
            st.subheader("Number Grid")
            number_grid(numbers_df.head(min(show_n, 20)))

    with tab2:
        st.subheader(f"Top {show_pairs} Pair Predictions")
        st.dataframe(pairs_df.head(show_pairs), use_container_width=True, hide_index=True)

    with tab3:
        st.subheader(f"Top {show_triplets} Triplet Predictions")
        st.dataframe(triplets_df.head(show_triplets), use_container_width=True, hide_index=True)

    with tab4:
        st.subheader(f"Top {show_combos} Combination Predictions")
        st.caption("Fast candidate-based ranking from the top individual numbers.")
        st.dataframe(combos_df.head(show_combos), use_container_width=True, hide_index=True)


elif page == "Latest Draw Review":
    st.markdown("<div class='edge-title'>Latest Draw Review</div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Lightweight review using already-computed rankings.</div>", unsafe_allow_html=True)

    review = latest_draw_review_light(df, numbers_df, pairs_df, triplets_df, ou)

    if review is None:
        st.warning("Need at least two draws.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            card("Draw Date", f"<div class='edge-big-blue'>{review['draw_date']}</div>")
        with c2:
            pills = "".join([f"<span class='edge-pill'>{n:02d}</span>" for n in review["numbers"]])
            card("Winning Numbers", f"<div>{pills}</div>", f"Sum: {review['actual_sum']}")
        with c3:
            status = "✅ Correct" if review["ou_correct"] else "❌ Missed"
            card("Over / Under", f"<div class='edge-big-green'>{status}</div>", f"Predicted {review['ou_prediction']}")
        with c4:
            card("Top 3 Hit Count", f"<div class='edge-big-blue'>{review['top3_hits']} / 3</div>")

        st.divider()
        st.subheader("Ball-by-ball Review")
        st.dataframe(review["ball_review"], use_container_width=True, hide_index=True)

        st.subheader("Pairs from Latest Draw that were in Top 50")
        if len(review["pairs_hit"]):
            st.dataframe(review["pairs_hit"], use_container_width=True, hide_index=True)
        else:
            st.info("None of the latest draw pairs appeared in the current top 50 pair rankings.")

        st.subheader("Triplets from Latest Draw that were in Top 50")
        if len(review["triplets_hit"]):
            st.dataframe(review["triplets_hit"], use_container_width=True, hide_index=True)
        else:
            st.info("None of the latest draw triplets appeared in the current top 50 triplet rankings.")


elif page == "Analytics":
    st.markdown("<div class='edge-title'>Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Lightweight analytics only. Heavy back-tests have been removed from live page loading.</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Numbers", "Pairs", "Triplets", "Sums"])
    with tab1:
        st.dataframe(numbers_df, use_container_width=True, hide_index=True)
        st.bar_chart(numbers_df.set_index("Number")["Score"])
    with tab2:
        st.dataframe(pairs_df, use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(triplets_df, use_container_width=True, hide_index=True)
    with tab4:
        st.line_chart(features.set_index("draw_date")["sum"])


elif page == "History / Updates":
    st.markdown("<div class='edge-title'>History / Updates</div>", unsafe_allow_html=True)
    st.info("On free Streamlit, manual changes may not persist forever. Download the CSV and commit it to GitHub.")

    show_history = st.checkbox("Show full history table", value=False)
    if show_history:
        st.dataframe(features.sort_values("draw_date", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.dataframe(features.sort_values("draw_date", ascending=False).head(20), use_container_width=True, hide_index=True)
