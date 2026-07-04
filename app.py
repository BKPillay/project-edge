from pathlib import Path
import pandas as pd
import streamlit as st
from src.edge_model import (
    NUMBER_COLS, add_draw_features, combination_scores, latest_draw_review, load_history,
    number_scores, over_under_prediction, pair_predictions, triplet_predictions,
    walk_forward_over_under_backtest_fast, simple_top3_backtest_fast
)

st.set_page_config(page_title='EDGE AI', page_icon='🎯', layout='wide')
DATA_PATH = Path('data/daily_lotto_history.csv')

st.markdown('''
<style>
.stApp {background: radial-gradient(circle at top left, rgba(95,68,255,.16), transparent 30%), radial-gradient(circle at top right, rgba(30,144,255,.12), transparent 28%), linear-gradient(180deg,#07111f 0%,#050b14 100%); color:#eef4ff;}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#081422 0%,#050b14 100%); border-right:1px solid rgba(120,145,180,.22);}
.block-container {padding-top:1.4rem; max-width:1500px;}
.edge-title {font-size:2.2rem; font-weight:900; margin-bottom:.1rem;}
.edge-ai {color:#9b5cff;}
.edge-subtitle {color:#9fb0c7; margin-bottom:1.2rem;}
.edge-card {background:linear-gradient(180deg,rgba(14,31,56,.92),rgba(7,17,31,.92)); border:1px solid rgba(120,145,180,.22); border-radius:14px; padding:1.1rem 1.25rem; box-shadow:0 12px 30px rgba(0,0,0,.22); min-height:120px;}
.edge-label {color:#9fb0c7; font-size:.88rem; margin-bottom:.25rem;}
.edge-big-green {color:#39d975; font-size:2rem; font-weight:900; line-height:1.05;}
.edge-big-blue {color:#3f8cff; font-size:1.55rem; font-weight:800;}
.edge-pill {display:inline-block; border:1px solid rgba(155,92,255,.85); border-radius:999px; padding:.35rem .58rem; margin:.15rem .18rem .15rem 0; min-width:40px; text-align:center; background:rgba(155,92,255,.09); font-weight:800;}
.edge-pill.blue {border-color:rgba(63,140,255,.85); background:rgba(63,140,255,.09);}
.edge-progress {height:8px; border-radius:999px; background:rgba(255,255,255,.12); overflow:hidden; margin:.5rem 0;}
.edge-progress > div {height:100%; border-radius:999px; background:linear-gradient(90deg,#39d975,#9b5cff);}
.edge-grid {display:grid; grid-template-columns:repeat(5,minmax(54px,1fr)); gap:.75rem; margin-top:.6rem;}
.edge-number-bubble {border:1px solid #39d975; border-radius:999px; height:54px; width:54px; display:flex; align-items:center; justify-content:center; font-weight:900; margin:auto; background:rgba(57,217,117,.08);}
.edge-score {text-align:center; color:#9fb0c7; font-size:.82rem; margin-top:.25rem;}
div[data-testid="stDataFrame"] {border:1px solid rgba(120,145,180,.22); border-radius:12px; overflow:hidden;}
</style>
''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def cached_load(path):
    df = load_history(path)
    return df, add_draw_features(df)

@st.cache_data(show_spinner=False)
def cached_predictions(path, combo_n, pair_n, triplet_n):
    df = load_history(path)
    return number_scores(df), combination_scores(df, combo_n), pair_predictions(df, pair_n), triplet_predictions(df, triplet_n), over_under_prediction(df)

@st.cache_data(show_spinner=False)
def cached_latest_review(path):
    return latest_draw_review(load_history(path))

@st.cache_data(show_spinner=False)
def cached_backtests(path):
    df = load_history(path)
    return walk_forward_over_under_backtest_fast(df, step=50), simple_top3_backtest_fast(df, step=50)

def card(title, body, footer=''):
    st.markdown(f"""<div class='edge-card'><div class='edge-label'>{title}</div>{body}<div style='color:#9fb0c7;font-size:.82rem;margin-top:.65rem;'>{footer}</div></div>""", unsafe_allow_html=True)

def number_grid(numbers_df):
    html = "<div class='edge-grid'>"
    for _, row in numbers_df.iterrows():
        html += f"<div><div class='edge-number-bubble'>{int(row['Number']):02d}</div><div class='edge-score'>{row['Score']}</div></div>"
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def metric_cards(df, features):
    latest = features.iloc[-1]
    c1,c2,c3,c4 = st.columns(4)
    with c1: card('Draws Loaded', f"<div class='edge-big-blue'>{len(df):,}</div>", 'Historical records')
    with c2: card('Latest Draw', f"<div class='edge-big-blue'>{latest['draw_date'].date()}</div>", 'Most recent CSV record')
    with c3: card('Latest Sum', f"<div class='edge-big-blue'>{int(latest['sum'])}</div>", 'Latest result sum')
    with c4: card('Latest Structure', f"<div class='edge-big-blue'>{latest['odd_count']}O/{latest['even_count']}E</div>", f"{latest['low_count']} Low / {latest['high_count']} High")

if not DATA_PATH.exists():
    st.error('Missing data/daily_lotto_history.csv')
    st.stop()

df, features = cached_load(str(DATA_PATH))

with st.sidebar:
    st.markdown("<div class='edge-title'>EDGE <span class='edge-ai'>AI</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='edge-subtitle'>Daily Lotto Intelligence</div>", unsafe_allow_html=True)
    page = st.radio('Navigation', ['Dashboard','Predictions','Latest Draw Review','Back-testing','Analytics','History / Updates'], label_visibility='collapsed')
    st.divider()
    st.subheader('➕ Add Draw')
    draw_date = st.date_input('Draw date')
    cols = st.columns(5)
    nums = [cols[i].number_input(f'N{i+1}', min_value=1, max_value=36, value=i+1, step=1) for i in range(5)]
    overwrite_existing = st.checkbox('Replace same date', value=True)
    if st.button('Save draw', use_container_width=True):
        new_nums = sorted([int(n) for n in nums])
        if len(set(new_nums)) != 5:
            st.error('The 5 numbers must be unique.')
        else:
            draw_date_ts = pd.to_datetime(draw_date)
            existing_date = df['draw_date'].dt.date.eq(draw_date_ts.date()).any()
            if existing_date and not overwrite_existing:
                st.error('Date exists. Tick replace.')
            else:
                new_row = pd.DataFrame([{'draw_date': draw_date_ts, 'n1': new_nums[0], 'n2': new_nums[1], 'n3': new_nums[2], 'n4': new_nums[3], 'n5': new_nums[4]}])
                updated = pd.concat([df, new_row], ignore_index=True)
                if overwrite_existing:
                    updated = updated.drop_duplicates(subset=['draw_date'], keep='last')
                updated = updated.sort_values('draw_date').reset_index(drop=True)
                updated.to_csv(DATA_PATH, index=False)
                st.cache_data.clear()
                st.success(f"Saved: {draw_date_ts.date()} — {'-'.join(map(str, new_nums))}")
                st.info('Refresh the app to recalculate all cached outputs.')
    st.download_button('Download history CSV', data=df.to_csv(index=False).encode('utf-8'), file_name='daily_lotto_history.csv', mime='text/csv', use_container_width=True)

st.markdown(f"<div class='edge-title'>{page}</div>", unsafe_allow_html=True)

if page == 'Dashboard':
    st.markdown("<div class='edge-subtitle'>Fast summary from cached model outputs.</div>", unsafe_allow_html=True)
    metric_cards(df, features)
    ns, combos, pairs, trips, ou = cached_predictions(str(DATA_PATH), 10, 10, 10)
    st.divider(); col1,col2 = st.columns([2,1])
    with col1:
        st.subheader('Top 10 Combinations'); st.dataframe(combos, use_container_width=True, hide_index=True)
    with col2:
        st.subheader('Top 3 Numbers'); number_grid(ns.head(3)); st.subheader('Over / Under'); st.metric(ou['prediction'], f"{ou['confidence']}% confidence")

elif page == 'Predictions':
    st.markdown("<div class='edge-subtitle'>Individual, pair, triplet, combination, and Over/Under outputs.</div>", unsafe_allow_html=True)
    show_n = st.slider('Individual numbers', 3, 36, 10)
    show_pairs = st.slider('Pairs', 5, 50, 10)
    show_triplets = st.slider('Triplets', 5, 50, 10)
    show_combos = st.slider('Combinations', 5, 50, 10)
    ns, combos, pairs, trips, ou = cached_predictions(str(DATA_PATH), show_combos, show_pairs, show_triplets)
    combo_top = combos.iloc[0]; top3 = ns.head(3); confidence_width = min(100, max(0, ou['confidence']))
    c1,c2,c3,c4 = st.columns([1.15,1,1.15,1.05])
    with c1: card('Over / Under 92.5', f"<div class='edge-big-green'>{ou['prediction']}</div><div>Confidence: <b>{ou['confidence']}%</b></div><div class='edge-progress'><div style='width:{confidence_width}%;'></div></div><div style='color:#9fb0c7;font-size:.85rem;'>Over {ou['prob_over']}% · Under {ou['prob_under']}%</div>", f"10-draw avg sum: {ou['recent_avg_sum_10']}")
    with c2: card('Top 3 Numbers', ''.join([f"<span class='edge-pill'>{int(n):02d}</span>" for n in top3['Number'].tolist()]), 'Individual model')
    with c3: card('Best 5-Number Combo', ''.join([f"<span class='edge-pill blue'>{int(n):02d}</span>" for n in combo_top['Combination'].split('-')]) + f"<div style='margin-top:.5rem;'>Score: <b>{combo_top['Score']}</b></div>", f"Sum {combo_top['Sum']}")
    with c4: card('Draw Structure', f"<div class='edge-big-blue'>{combo_top['Low']}L/{combo_top['High']}H</div><div>{combo_top['Odd']} Odd / {combo_top['Even']} Even</div>", 'Based on best combo')
    st.divider(); tab1,tab2,tab3,tab4 = st.tabs(['Individual','Pairs','Triplets','Combinations'])
    with tab1: st.dataframe(ns.head(show_n), use_container_width=True, hide_index=True)
    with tab2: st.dataframe(pairs, use_container_width=True, hide_index=True)
    with tab3: st.dataframe(trips, use_container_width=True, hide_index=True)
    with tab4: st.dataframe(combos, use_container_width=True, hide_index=True)

elif page == 'Latest Draw Review':
    st.markdown("<div class='edge-subtitle'>How the latest result looked to the model before the draw happened.</div>", unsafe_allow_html=True)
    review = cached_latest_review(str(DATA_PATH))
    if not review.get('available'):
        st.warning(review.get('reason','Not enough data.'))
    else:
        nums = review['latest_numbers']; c1,c2,c3,c4 = st.columns(4)
        with c1: card('Draw Date', f"<div class='edge-big-blue'>{review['draw_date']}</div>")
        with c2: card('Winning Numbers', ''.join([f"<span class='edge-pill'>{n:02d}</span>" for n in nums]), f"Sum: {review['actual_sum']}")
        with c3: card('Over / Under', f"<div class='edge-big-green'>{'✅ Correct' if review['ou_correct'] else '❌ Missed'}</div>", f"Predicted {review['ou_prediction']} at {review['ou_confidence']}%")
        with c4: card('Top 3 Hit Count', f"<div class='edge-big-blue'>{review['top3_hits']} / 3</div>", 'Individual model')
        st.divider(); st.subheader('Ball-by-ball Pre-draw Review'); st.dataframe(review['ball_review'], use_container_width=True, hide_index=True)
        st.subheader('Pairs from Latest Draw'); st.dataframe(review['pair_review'], use_container_width=True, hide_index=True) if len(review['pair_review']) else st.info('No pair review available.')
        st.subheader('Triplets from Latest Draw'); st.dataframe(review['triplet_review'], use_container_width=True, hide_index=True) if len(review['triplet_review']) else st.info('No triplet review available.')
        st.subheader('Combination Model'); st.write(f"Best top-10 pre-draw combo: **{review['best_top10_combo']}**"); st.write(f"Matched latest draw: **{review['best_top10_combo_matches']} / 5**")

elif page == 'Back-testing':
    st.markdown("<div class='edge-subtitle'>Cached walk-forward checks. No heavy recalculation on every click.</div>", unsafe_allow_html=True)
    st.warning('Back-tests use a 50-draw step sample to keep free Streamlit fast. Full audit can be added later as a cached offline file.')
    if st.button('Run / refresh cached back-tests', use_container_width=True): st.cache_data.clear(); st.success('Cache cleared. Reopen this page or refresh.')
    with st.spinner('Loading cached back-tests...'):
        ou_bt, t3_bt = cached_backtests(str(DATA_PATH))
    acc = round(float(ou_bt['correct'].mean()) * 100, 2) if len(ou_bt) else 0
    c1,c2,c3 = st.columns(3); c1.metric('O/U accuracy', f'{acc}%'); c2.metric('O/U tests', len(ou_bt)); c3.metric('Top 3 avg matches', round(float(t3_bt['matches'].mean()),3) if len(t3_bt) else 0)
    st.subheader('Over / Under Back-test'); st.dataframe(ou_bt.sort_values('draw_date', ascending=False).head(100), use_container_width=True, hide_index=True)
    st.subheader('Top 3 Back-test Sample'); st.dataframe(t3_bt.sort_values('draw_date', ascending=False).head(100), use_container_width=True, hide_index=True)

elif page == 'Analytics':
    st.markdown("<div class='edge-subtitle'>Historical patterns and model inputs.</div>", unsafe_allow_html=True)
    metric_cards(df, features)
    ns, combos, pairs, trips, ou = cached_predictions(str(DATA_PATH), 10, 100, 100)
    tab1,tab2,tab3,tab4 = st.tabs(['Numbers','Pairs','Triplets','Structures'])
    with tab1: st.dataframe(ns, use_container_width=True, hide_index=True); st.bar_chart(ns.set_index('Number')['Score'])
    with tab2: st.dataframe(pairs, use_container_width=True, hide_index=True)
    with tab3: st.dataframe(trips, use_container_width=True, hide_index=True)
    with tab4: st.dataframe(features['structure'].value_counts().reset_index(), use_container_width=True, hide_index=True); st.line_chart(features.set_index('draw_date')['sum'])

elif page == 'History / Updates':
    st.markdown("<div class='edge-subtitle'>Manage the historical dataset.</div>", unsafe_allow_html=True)
    metric_cards(df, features)
    st.info('On free Streamlit hosting, manual changes may not persist forever. Download the updated CSV and commit it to GitHub.')
    uploaded = st.file_uploader('Bulk replace history CSV', type=['csv'])
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded); required = {'draw_date','n1','n2','n3','n4','n5'}
            if not required.issubset(set(uploaded_df.columns)): st.error('CSV must contain draw_date,n1,n2,n3,n4,n5')
            else:
                uploaded_df['draw_date'] = pd.to_datetime(uploaded_df['draw_date'])
                for c in NUMBER_COLS: uploaded_df[c] = pd.to_numeric(uploaded_df[c], errors='raise').astype(int)
                uploaded_df = uploaded_df.sort_values('draw_date').drop_duplicates(subset=['draw_date'], keep='last')
                if st.button('Replace app history with uploaded CSV'):
                    uploaded_df.to_csv(DATA_PATH, index=False); st.cache_data.clear(); st.success('CSV replaced. Refresh app.')
        except Exception as e: st.error(f'Could not process CSV: {e}')
    st.subheader('Historical draws'); st.dataframe(features.sort_values('draw_date', ascending=False), use_container_width=True, hide_index=True)
