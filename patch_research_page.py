
from pathlib import Path

APP_PATH = Path("app.py")
text = APP_PATH.read_text(encoding="utf-8")

old_nav = '["Dashboard", "Predictions", "Latest Draw Review", "Analytics", "Back-testing", "History / Updates"]'
new_nav = '["Dashboard", "Predictions", "Latest Draw Review", "Analytics", "Back-testing", "Research / Validation", "History / Updates"]'
if old_nav in text:
    text = text.replace(old_nav, new_nav)

marker = 'structural = read_json("structural_summary.json")'
insert = '''
research_repeat = read_csv("research/number_repeat_rate_study.csv")
research_cooling = read_csv("research/recency_cooling_backtest.csv")
research_summary = read_json("research/research_summary.json")
'''
if marker in text and "research_cooling = read_csv" not in text:
    text = text.replace(marker, marker + "\n" + insert)

history_marker = 'elif page == "History / Updates":'
research_page = '''
elif page == "Research / Validation":
    st.markdown("<div class='edge-title'>Research / Validation</div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Evidence reports generated offline. This page does not run heavy analysis live.</div>", unsafe_allow_html=True)

    if not len(research_cooling) and not len(research_repeat):
        st.warning("No research outputs found yet. Run: python scripts\\run_research_validation.py")
    else:
        st.subheader("Recency Cooling Back-test")
        st.dataframe(research_cooling, use_container_width=True, hide_index=True)

        st.subheader("Number Repeat Rate Study")
        st.dataframe(research_repeat, use_container_width=True, hide_index=True)

        st.subheader("Research Summary")
        st.json(research_summary)


'''
if history_marker in text and 'elif page == "Research / Validation":' not in text:
    text = text.replace(history_marker, research_page + history_marker)

APP_PATH.write_text(text, encoding="utf-8")
print("Research page patched into app.py.")
