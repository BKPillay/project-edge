# EDGE AI Daily Lotto - Project State

## Current Version
v2.9.0

## Current Milestone
Strategy Benchmark Suite optimisation.

## What Changed
The Strategy Benchmark Suite was redesigned so benchmark strategies do not repeatedly rebuild historical counters from scratch. The new benchmark engine precomputes prefix caches for:

- number counts
- pair counts
- triplet counts
- pair-derived number scores
- triplet-derived number scores
- last-seen gaps
- repeat-after-1/2/3 statistics

This makes sampled walk-forward benchmarking practical and makes stronger audits possible.

## Benchmark Commands

Fast sampled benchmark:

```bash
python scripts/run_strategy_benchmark.py --min-train 100 --step 20
```

Stronger sampled benchmark:

```bash
python scripts/run_strategy_benchmark.py --min-train 100 --step 5
```

Full heavier walk-forward benchmark:

```bash
python scripts/run_strategy_benchmark.py --min-train 100 --step 1
```

Baseline-only benchmark:

```bash
python scripts/run_strategy_benchmark.py --min-train 100 --step 1 --no-edge-ai
```

Exact original EDGE path for small parity checks only:

```bash
python scripts/run_strategy_benchmark.py --min-train 100 --step 20 --edge-mode exact
```

## Current Finding From Local Test
A `--step 5` benchmark completed successfully in the packaging environment and produced 514 tests per strategy.

The sampled result showed EDGE AI was competitive but not dominant. This is exactly why the Feature Importance Engine is still required next.

## Next Recommended Milestone
Feature Importance Engine.

Purpose:
- Disable or isolate features one at a time.
- Rerun benchmark.
- Measure which components genuinely improve results.
- Remove or reduce features that are dead weight.

## Important Project Principle
Do not treat EDGE AI as successful unless it beats simple baselines consistently over meaningful walk-forward tests.
