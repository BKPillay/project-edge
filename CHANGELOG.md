# Changelog

## EDGE_AI_v2.8.0

### Added

- Strategy Benchmark Suite.
- New model module: `models/strategy_benchmark.py`.
- New script: `scripts/run_strategy_benchmark.py`.
- New Streamlit page: `Strategy Benchmark`.
- New benchmark outputs:
  - `outputs/strategy_benchmark_results.csv`
  - `outputs/strategy_benchmark_summary.csv`
  - `outputs/strategy_benchmark_summary.json`
- `PROJECT_STATE.md` to preserve project context across conversations.

### Changed

- `scripts/run_model_update.py` now also generates benchmark outputs.
- `README.md` updated with benchmark instructions.

### Preserved

- Existing v2 model architecture.
- Existing full-history data structure.
- Existing research validation outputs.
- Existing pair/triplet cooling and combination ensemble logic.

### Notes

The benchmark is intentionally strict. EDGE AI must prove it beats simple baselines, otherwise the model should be simplified or reweighted.
