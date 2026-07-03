from pathlib import Path
import pandas as pd
import streamlit as st
from src.edge_model import add_draw_features, combination_scores, load_history, number_scores, over_under_prediction

st.set_page_config(page_title="Project EDGE", page_icon="🎯", layout="wide")
DATA_PATH = Path("data/daily_lotto_history.csv")

st.title("🎯 Project EDGE — SA Daily Lotto")
st.caption("Phone-friendly edge-seeking model suite. Ranking, not guaranteed prediction.")

if not DATA_PATH.exists():
    st.error("Missing data/daily_lotto_history.csv")
    st.stop()

df = load_history(DATA_PATH)
features = add_draw_features(df)

with st.sidebar:
    st.header("➕ Add new draw")

    draw_date = st.date_input("Draw date")

    st.write("Enter 5 unique numbers from 1 to 36.")
    cols = st.columns(5)
    nums = [
        cols[0].number_input("N1", min_value=1, max_value=36, value=1, step=1),
        cols[1].number_input("N2", min_value=1, max_value=36, value=2, step=1),
        cols[2].number_input("N3", min_value=1, max_value=36, value=3, step=1),
        cols[3].number_input("N4", min_value=1, max_value=36, value=4, step=1),
        cols[4].number_input("N5", min_value=1, max_value=36, value=5, step=1),
    ]

    overwrite_existing = st.checkbox("Replace existing draw if date already exists", value=True)

    if st.button("Save draw", use_container_width=True):
        new_nums = sorted([int(n) for n in nums])

        if len(set(new_nums)) != 5:
            st.error("The 5 numbers must be unique.")
        elif any(n < 1 or n > 36 for n in new_nums):
            st.error("All numbers must be between 1 and 36.")
        else:
            draw_date_ts = pd.to_datetime(draw_date)

            existing_date = df["draw_date"].dt.date.eq(draw_date_ts.date()).any()

            if existing_date and not overwrite_existing:
                st.error("That draw date already exists. Tick the replace option to overwrite it.")
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

                st.success(f"Saved draw: {draw_date_ts.date()} — {'-'.join(map(str, new_nums))}")
                st.info("Refresh the app to recalculate the model with the new record.")

    st.divider()
    st.header("⬇️ Export data")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download history CSV",
        data=csv_data,
        file_name="daily_lotto_history.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()
    st.header("📤 Bulk replace CSV")
    uploaded = st.file_uploader("Upload full history CSV", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            required = {"draw_date", "n1", "n2", "n3", "n4", "n5"}
            if not required.issubset(set(uploaded_df.columns)):
                st.error("CSV must contain: draw_date,n1,n2,n3,n4,n5")
            else:
                uploaded_df["draw_date"] = pd.to_datetime(uploaded_df["draw_date"])
                for c in ["n1", "n2", "n3", "n4", "n5"]:
                    uploaded_df[c] = pd.to_numeric(uploaded_df[c], errors="raise").astype(int)

                invalid_rows = []
                for i, row in uploaded_df.iterrows():
                    vals = [row["n1"], row["n2"], row["n3"], row["n4"], row["n5"]]
                    if len(set(vals)) != 5 or any(v < 1 or v > 36 for v in vals):
                        invalid_rows.append(i + 2)

                if invalid_rows:
                    st.error(f"Invalid numbers found in CSV rows: {invalid_rows[:10]}")
                elif st.button("Replace app history with uploaded CSV", use_container_width=True):
                    uploaded_df = uploaded_df.sort_values("draw_date").drop_duplicates(subset=["draw_date"], keep="last")
                    uploaded_df.to_csv(DATA_PATH, index=False)
                    st.success("CSV replaced. Refresh the app.")
        except Exception as e:
            st.error(f"Could not process CSV: {e}")

st.subheader("Daily summary")
latest = features.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Draws loaded", len(df))
c2.metric("Latest draw", str(latest["draw_date"].date()))
c3.metric("Latest sum", int(latest["sum"]))
c4.metric("Latest O/U", "Over" if latest["over_92_5"] else "Under")

st.divider()
left, right = st.columns([2,1])

with left:
    st.subheader("Model 1 — Top 10 ranked combinations")
    st.dataframe(combination_scores(df, top_n=10), use_container_width=True, hide_index=True)

with right:
    st.subheader("Model 2 — Top 3 numbers")
    top3 = number_scores(df).head(3)
    st.dataframe(top3[["number","score","long_count","recent_20_count","gap"]], use_container_width=True, hide_index=True)

    st.subheader("Model 3 — Over / Under 92.5")
    ou = over_under_prediction(df)
    st.metric(ou["prediction"], f'{ou["confidence"]}% confidence')
    st.write(f'Over: {ou["prob_over"]}%')
    st.write(f'Under: {ou["prob_under"]}%')
    st.write(f'Recent avg sum: {ou["recent_avg_sum"]}')

st.divider()
st.subheader("History")
st.dataframe(features.sort_values("draw_date", ascending=False), use_container_width=True, hide_index=True)

st.warning("Ruthless truth: judge this by long-run tracking. A few good hits mean nothing unless it beats random over time.")
