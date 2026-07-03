# Project EDGE — Daily Lotto MVP

This is the first working version of your phone-friendly Daily Lotto model suite.

It gives:

1. Model 1 — Top 10 ranked combinations
2. Model 2 — Top 3 ranked numbers
3. Model 3 — Over/Under 92.5 prediction

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

Update:

```text
data/daily_lotto_history.csv
```

Required columns:

```text
draw_date,n1,n2,n3,n4,n5
```

The included CSV only contains the recent known rows from our conversation. You should load full historical data before trusting the model.

## Azure

Best MVP deployment:

- Azure App Service for Containers, or
- Azure Container Apps

Do not start with Databricks, AKS, or Fabric. That is overkill until the edge is proven.

## Important

This model ranks historically well-formed selections. It cannot guarantee random lottery results.


## New-record entry

The app now includes a sidebar form to:

- Add a new Daily Lotto draw
- Replace an existing draw for the same date
- Download the current history CSV
- Bulk upload/replace the full history CSV

After saving a new draw, refresh the app so all three models recalculate.
