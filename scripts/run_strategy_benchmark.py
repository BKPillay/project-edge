from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.common import load_history
from models.strategy_benchmark import run_strategy_benchmark

DATA_PATH = PROJECT_ROOT / "data" / "daily_lotto_history.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Run EDGE AI Strategy Benchmark Suite")
    parser.add_argument("--min-train", type=int, default=300)
    parser.add_argument("--step", type=int, default=20, help="Use 1 for a full walk-forward audit")
    parser.add_argument("--edge-mode", choices=["fast", "exact"], default="fast", help="fast uses optimized prefix caches; exact uses the original heavy dataframe path")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N test draws; use 0 to disable")
    parser.add_argument("--no-edge-ai", action="store_true", help="Skip EDGE AI strategy for faster baseline-only checks")
    args = parser.parse_args()

    df = load_history(DATA_PATH)
    results, summary, metadata = run_strategy_benchmark(
        df,
        min_train=args.min_train,
        step=args.step,
        include_edge_ai=not args.no_edge_ai,
        edge_mode=args.edge_mode,
        progress_every=args.progress_every or None,
    )

    results.to_csv(OUTPUT_DIR / "strategy_benchmark_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "strategy_benchmark_summary.csv", index=False)
    (OUTPUT_DIR / "strategy_benchmark_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Strategy Benchmark Suite complete.")
    print(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
