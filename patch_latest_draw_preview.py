
from pathlib import Path

APP_PATH = Path("app.py")
text = APP_PATH.read_text(encoding="utf-8")

start_marker = 'elif page == "Latest Draw Review":'
end_marker = 'elif page == "Analytics":'

start = text.find(start_marker)
end = text.find(end_marker)

if start == -1 or end == -1 or end <= start:
    raise RuntimeError("Could not locate Latest Draw Review block in app.py.")

new_block = r"""
elif page == "Latest Draw Review":
    st.markdown("<div class='edge-title'>Latest Draw Review</div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Preview a new draw instantly, then compare the latest saved draw against the pre-draw model snapshot.</div>", unsafe_allow_html=True)

    st.subheader("Preview New Draw")

    with st.form("preview_new_draw_form"):
        preview_date = st.date_input("Draw date")
        cols = st.columns(5)
        preview_nums = [
            cols[0].number_input("N1", min_value=1, max_value=36, value=1, step=1),
            cols[1].number_input("N2", min_value=1, max_value=36, value=2, step=1),
            cols[2].number_input("N3", min_value=1, max_value=36, value=3, step=1),
            cols[3].number_input("N4", min_value=1, max_value=36, value=4, step=1),
            cols[4].number_input("N5", min_value=1, max_value=36, value=5, step=1),
        ]
        preview_clicked = st.form_submit_button("Preview Ball Review")

    preview_nums = sorted([int(n) for n in preview_nums])

    if preview_clicked:
        if len(set(preview_nums)) != 5:
            st.error("Enter 5 unique numbers from 1 to 36.")
        else:
            score_map = numbers.set_index("number").to_dict(orient="index")
            preview_rows = []

            for n in preview_nums:
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

            preview_sum = sum(preview_nums)
            preview_ou = "Over 92.5" if preview_sum > 92.5 else "Under 92.5"
            top3_hits = len(set(preview_nums) & set(numbers.head(3)["number"].astype(int).tolist()))
            top10_hits = len(set(preview_nums) & set(numbers.head(10)["number"].astype(int).tolist()))

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                card("Preview Date", str(preview_date))
            with c2:
                st.markdown("<div class='edge-card'><div class='edge-label'>Entered Numbers</div>" + pills(preview_nums) + f"<div style='color:#9fb0c7;font-size:.82rem;margin-top:.5rem;'>Sum: {preview_sum}</div></div>", unsafe_allow_html=True)
            with c3:
                card("O/U Result", preview_ou, f"Current model says: {ou.get('prediction', 'n/a')}")
            with c4:
                card("Model Capture", f"{top3_hits}/3", f"Top 10 hits: {top10_hits}/5")

            st.subheader("Preview Ball Review")
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

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


"""

text = text[:start] + new_block + text[end:]
APP_PATH.write_text(text, encoding="utf-8")
print("Patched Latest Draw Review page with Preview New Draw section.")
