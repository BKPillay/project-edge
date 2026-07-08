# EDGE AI Project State

## Current Release

Version: `EDGE_AI_v2.8.0`

Baseline source: user-uploaded `project-edge.zip`.

This release preserves the existing v2 platform and adds the Strategy Benchmark Suite.

## Existing Architecture Preserved

The uploaded project already contained:

- Full-history Daily Lotto dataset
- Streamlit app
- Offline output generation flow
- Number model
- Pair model
- Triplet model
- Combination ensemble model
- Structural model
- Over/Under model
- Latest draw review engine
- Research validation scripts
- Recency cooling research outputs
- Pair/triplet cooling logic
- Combination ensemble using cooled lower-level selection scores

## New in v2.8.0

Added Strategy Benchmark Suite:

- `models/strategy_benchmark.py`
- `scripts/run_strategy_benchmark.py`
- Streamlit `Strategy Benchmark` page
- Offline benchmark outputs

## Benchmark Purpose

The benchmark exists to answer one question:

> Is EDGE AI outperforming simple strategies that take almost no intelligence?

If EDGE AI cannot beat these baselines, the model is not earning its complexity.

## Baseline Strategies

- Random
- Hot Numbers
- Cold Numbers
- Overdue
- Recent Repeat
- Pair Frequency
- Triplet Frequency
- Balanced Hot

## EDGE AI Benchmark Path

EDGE AI is benchmarked through the real ensemble path:

```text
score_numbers -> score_pairs -> score_triplets -> score_combinations
```

This prevents a false benchmark where EDGE AI is represented by a simplified proxy.

## Validation Method

Sampled walk-forward validation:

```text
Train on historical draws before target draw
Predict target draw
Compare prediction to actual result
Repeat forward through history
```

Default benchmark settings:

- `min_train = 300`
- `step = 20`
- `include_edge_ai = True`

Use `--step 1` for a heavier full audit.

## Current Next Milestones

1. Feature Importance Engine
2. Weight Optimiser
3. Walk-Forward Validator with configurable model versions
4. Experiment Registry

## Project Continuity Rule

Future work should be delivered as versioned ZIP releases, not loose code snippets.

Each release should include:

- Complete source files
- Updated outputs where relevant
- README updates
- CHANGELOG updates
- PROJECT_STATE updates
- Version number

