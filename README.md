# EDGE AI v2 Full-History Platform

This version rebuilds EDGE AI around the full Daily Lotto history file.

## Key changes

- Uses the master CSV with `draw_number`
- Preserves full history
- Splits the model into independent engines
- Generates outputs offline
- Streamlit reads output files only
- Adds richer number scores:
  - frequency score
  - momentum score
  - gap score
  - final score
- Adds latest draw ball/pair/triplet review
- Adds fast back-test summary

## Run model update

From the project root:

```powershell
python scripts\run_model_update.py
```

Then commit:

```powershell
git add data outputs models scripts app.py .github README.md
git commit -m "EDGE AI v2 full history platform"
git push
```

## Note

Full walk-forward back-testing is deliberately not part of the standard update because it can be expensive. It should be added later as an optional audit script.

---

## EDGE AI v2.8.0 - Strategy Benchmark Suite

This release adds the Strategy Benchmark Suite without replacing the existing v2 model architecture.

### New files

- `models/strategy_benchmark.py`
- `scripts/run_strategy_benchmark.py`
- `outputs/strategy_benchmark_results.csv`
- `outputs/strategy_benchmark_summary.csv`
- `outputs/strategy_benchmark_summary.json`
- `PROJECT_STATE.md`
- `CHANGELOG.md`

### What the benchmark does

The benchmark compares EDGE AI against simple baseline strategies:

- Random
- Hot Numbers
- Cold Numbers
- Overdue
- Recent Repeat
- Pair Frequency
- Triplet Frequency
- Balanced Hot

The EDGE AI benchmark path uses the actual model chain:

```text
score_numbers -> score_pairs -> score_triplets -> score_combinations
```

This is important. It means the benchmark tests the same ensemble logic used by the app, not a fake simplified EDGE score.

### Run benchmark only

```powershell
python scripts\run_strategy_benchmark.py
```

For a heavier full audit:

```powershell
python scripts\run_strategy_benchmark.py --step 1
```

### Run complete model update

```powershell
python scripts\run_model_update.py
```

This now regenerates the normal prediction outputs and the sampled benchmark outputs.

### Streamlit

The app now includes a `Strategy Benchmark` page.


## v2.9 Benchmark Optimisation Notes

The Strategy Benchmark Suite now has an optimised prefix-cache engine.

Recommended benchmark commands:

```bash
python scripts/run_strategy_benchmark.py --min-train 100 --step 20
python scripts/run_strategy_benchmark.py --min-train 100 --step 5
```

Use full step 1 only for the heaviest audit:

```bash
python scripts/run_strategy_benchmark.py --min-train 100 --step 1
```

The Streamlit Strategy Benchmark page currently reads the output CSV files from `outputs/`. Run the script first, then refresh the Streamlit page.
