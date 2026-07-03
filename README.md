# Project EDGE — Daily Lotto MVP v3 Full History

This version includes the downloaded South African Daily Lotto history.

- Rows loaded: 2665
- First draw date: 2019-03-10
- Latest draw date: 2026-07-02
- Invalid rows excluded: 0

## Models

1. Model 1 — Top 10 ranked combinations
2. Model 2 — Top 3 ranked numbers
3. Model 3 — Over/Under 92.5 prediction

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Free hosting

Deploy to Streamlit Community Cloud from GitHub.

## Important Streamlit persistence warning

On free Streamlit hosting, manually added records may not permanently write back to GitHub.
After adding draws, download the updated CSV from the app and keep it safe.

## Data files

- `data/daily_lotto_history.csv` — normalized app history
- `data/source_africanlottery_daily_lotto_raw.csv` — raw source copy

## Brutal truth

This ranks historically well-formed selections. It does not make random lottery draws predictable.


## Streamlit Cloud deployment fix

This package includes:

```text
runtime.txt
```

with:

```text
python-3.12
```

and a simplified `requirements.txt`:

```text
streamlit
pandas
numpy
```

This avoids Streamlit Cloud attempting to build the app with Python 3.14 and unnecessary packages.
