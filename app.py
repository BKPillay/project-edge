
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
      radial-gradient(circle at top left, rgba(95,68,255,.16), transparent 30%),
      radial-gradient(circle at top right, rgba(30,144,255,.12), transparent 28%),
      linear-gradient(180deg, #07111f 0%, #050b14 100%);
    color: #eef4ff;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #081422 0%, #050b14 100%);
    border-right: 1px solid rgba(120,145,180,.22);
}
.block-container { padding-top: 1.4rem; max-width: 1500px; }
.edge-title { font-size: 2.2rem; font-weight: 900; margin-bottom: .1rem; }
.edge-ai { color: #9b5cff; }
.edge-subtitle { color: #9fb0c7; margin-bottom: 1.2rem; }
.edge-card {
    background: linear-gradient(180deg, rgba(14,31,56,.92), rgba(7,17,31,.92));
    border: 1px solid rgba(120,145,180,.22);
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 12px 30px rgba(0,0,0,.22);
    min-height: 120px;
}
.edge-label { color:#9fb0c7; font-size:.88rem; margin-bottom:.25rem; }
.edge-big-green { color:#39d975; font-size:2.0rem; font-weight:900; line-height:1.05; }
.edge-big-blue { color:#3f8cff; font-size:1.55rem; font-weight:800; }
.edge-pill {
    display:inline-block; border:1px solid rgba(155,92,255,.85); border-radius:999px;
    padding:.35rem .58rem; margin:.15rem .18rem .15rem 0; min-width:40px;
    text-align:center; background:rgba(155,92,255,.09); font-weight:800;
}
.edge-pill.blue { border-color:rgba(63,140,255,.85); background:rgba(63,140,255,.09); }
.edge-progress { height:8px; border-radius:999px; background:rgba(255,255,255,.12); overflow:hidden; margin:.5rem 0; }
.edge-progress > div { height:100%; border-radius:999px; background:linear-gradient(90deg,#39d975,#9b5cff); }
.edge-grid { display:grid; grid-template-columns:repeat(5,minmax(54px,1fr)); gap:.75rem; margin-top:.6rem; }
.edge-number-bubble {
    border:1px solid #39d975; border-radius:999px; height:54px; width:54px;
    display:flex; align-items:center; justify-content:center; font-weight:900; margin:auto;
    background:rgba(57,217,117,.08);
}
.edge-score { text-align:center; color:#9fb0c7; font-size:.82rem; margin-top:.25rem; }
div[data-testid="stDataFrame"] { border:1px solid rgba(120,145,180,.22); border-radius:12px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)


def card(title: str, body: str, footer: str = ""):
    st.markdown(
        f"""
        <div class="edge-card">
          <div class="edge-label">{title}</div>
          {body}
          <div style="color:#9fb0c7;font-size:.82rem;margin-top:.65rem;">{footer}</div>
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
def load_all():
    df = load_history(DATA_PATH)
    features = add_draw_features(df)
    numbers = number_scores(df)
    ou = over_under_prediction(df)
    return df, features, numbers, ou


def fast_combos(numbers_df: pd.DataFrame, top_n: int = 10, pool_size: int = 15) -> pd.DataFrame:
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


if not DATA_PATH.exists():
    st.error("Missing data/daily_lotto_history.csv")
    st.stop()

df, features, numbers_df, ou = load_all()

with st.sidebar:
    st.markdown("<div class='edge-title'>EDGE <span class='edge-ai'>AI</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Daily Lotto Intelligence</div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Dashboard", "Predictions", "Analytics", "History / Updates"],
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
    st.markdown("<div class='edge-subtitle'>Fast summary from the model.</div>", unsafe_allow_html=True)

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
    st.subheader("Top 3 Numbers")
    number_grid(numbers_df.head(3))

    st.subheader("Top 10 Combinations")
    st.dataframe(fast_combos(numbers_df, 10), use_container_width=True, hide_index=True)


elif page == "Predictions":
    st.markdown("<div class='edge-title'>Predictions</div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Individual, pair, triplet, combination, and Over/Under outputs.</div>", unsafe_allow_html=True)

    control_cols = st.columns(4)
    with control_cols[0]:
        show_n = st.slider("Individual numbers", 3, 36, 10)
    with control_cols[1]:
        show_pairs = st.slider("Pairs", 5, 50, 10)
    with control_cols[2]:
        show_triplets = st.slider("Triplets", 5, 50, 10)
    with control_cols[3]:
        show_combos = st.slider("Combinations", 5, 50, 10)

    with st.spinner("Loading prediction outputs..."):
        pairs_df = pair_predictions(df, show_pairs)
        triplets_df = triplet_predictions(df, show_triplets)
        combos_df = fast_combos(numbers_df, show_combos)

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
        st.dataframe(pairs_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader(f"Top {show_triplets} Triplet Predictions")
        st.dataframe(triplets_df, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader(f"Top {show_combos} Combination Predictions")
        st.caption("Fast candidate-based ranking from the top individual numbers.")
        st.dataframe(combos_df, use_container_width=True, hide_index=True)


elif page == "Analytics":
    st.markdown("<div class='edge-title'>Analytics</div>", unsafe_allow_html=True)
    st.dataframe(numbers_df, use_container_width=True, hide_index=True)
    st.bar_chart(numbers_df.set_index("Number")["Score"])


elif page == "History / Updates":
    st.markdown("<div class='edge-title'>History / Updates</div>", unsafe_allow_html=True)
    st.info("On free Streamlit, manual changes may not persist forever. Download the CSV and commit it to GitHub.")
    st.dataframe(features.sort_values("draw_date", ascending=False), use_container_width=True, hide_index=True)
