# Changelog

## v2.9.0 - Optimised Strategy Benchmark Engine

### Added
- Prefix-cache benchmark engine for walk-forward strategy validation.
- `--edge-mode fast|exact` option for benchmark runs.
- `--progress-every` option so long benchmark runs show progress.
- Optimised baseline strategy execution using cumulative counts instead of rebuilding history each test.
- Optimised EDGE AI benchmark path using cached number, pair, triplet and repeat/gap statistics.

### Changed
- `scripts/run_strategy_benchmark.py` now defaults to `edge-mode=fast`.
- `models/strategy_benchmark.py` now supports practical sampled benchmarks such as `--step 5` and stronger full audits.

### Notes
- `--edge-mode exact` keeps the original dataframe-heavy path for small parity checks only.
- Full `--step 1` is still the heaviest audit because EDGE AI evaluates combinations for every historical test draw.
- Recommended practical command for now: `python scripts/run_strategy_benchmark.py --min-train 100 --step 5`.
