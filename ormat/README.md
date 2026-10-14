# ormat: multi-warehouse replenishment with a Multi-Agent Transformer

A small project that uses a minimal MAT to study whether the order in which agents choose their actions matters in an operations-research problem.

## Problem

`envs/replenishment.py`: N = 4 warehouses share one supplier with limited capacity.

- Each step, every warehouse orders 0 to 12 units. If the total order exceeds the capacity C, orders are rationed proportionally.
- Lead time is 2 steps, demand ~ Poisson(μᵢ) with μ = (2, 4, 6, 8), and unmet demand is lost.
- Warehouses are heterogeneous: stockout penalties b = (2, 4, 8, 16), holding cost 1 for all.
- Episodes last 50 steps; the shared team reward is −(holding cost + stockout cost).
- `capacity_ratio` = capacity / mean total demand. The smaller it is, the more the warehouses must coordinate who orders more and who orders less.

## Methods

| Name | Description |
|---|---|
| base-stock (critical ratio) | Newsvendor order-up-to levels Sᵢ; each warehouse decides independently |
| (s,S) grid search | Searches a common offset around the critical-ratio levels and a reorder gap |
| priority-rationing base-stock | When total orders exceed capacity, serve warehouses in decreasing order of b; a hand-written, capacity-aware heuristic |
| MAT-fixed | Original MAT; warehouses act in index order |
| MAT-random | A new random order every step |
| MAT-learned | A scoring network + Plackett-Luce sampling learns the order (the core idea of PMAT); the order is trained with PPO as part of the action |
| MAT-none | No autoregression; all warehouses act simultaneously (MAPPO-style) |

## Running

```bash
pip install -r requirements.txt
python -m ormat.baselines --capacity-ratio 0.85
python -m ormat.train --order-mode learned --seed 0 --capacity-ratio 0.85   # about 5 minutes on one CPU thread
python -m ormat.plot --results ormat/results_cap0.85 --capacity-ratio 0.85
```

## Preliminary results

Mean cost per episode over 200 evaluation episodes after 300 PPO updates (480k environment steps), 3 seeds per method. Lower is better.

| Method | Capacity 1.05 × demand | Capacity 0.85 × demand |
|---|---|---|
| base-stock (critical ratio) | 1144 | 2516 |
| (s,S) grid search | 1116 | 1594 |
| priority-rationing base-stock | 1123 | **1272** |
| MAT-fixed | 1120 ± 10 | 1389 ± 16 |
| MAT-random | 1133 ± 9 | 1387 ± 13 |
| MAT-learned | 1133 ± 6 | **1362 ± 19** |
| MAT-none | 1129 ± 3 | 1376 ± 14 |

Learning curves: `results/curves.png` and `results_cap0.85/curves.png`.

Observations:

1. **With loose capacity, the order does not matter.** Every method ties with base-stock, so this setting cannot separate the methods.
2. **With tight capacity**, every MAT variant is far better than base-stock and (s,S), which ignore the capacity. The learned order is the best of the four MAT variants, but only by one to two standard deviations; more seeds are needed.
3. The learned order puts **warehouse 0, with the lowest stockout penalty, first** (mean position 0.39) and the high-penalty warehouses last. One explanation is that the most important warehouse decides after seeing everyone else's order.
4. The hand-written priority heuristic still beats every MAT variant by about 7%, so there is clear room for improvement.

## Next steps

- Treat the priority heuristic as the baseline to beat; try longer training, a larger model and value normalization.
- Run 5 to 10 seeds per method and test for significance.
- Change the rationing rule to first-come-first-served in decision order, so the order directly affects allocation.
- Scale to 8 or 16 warehouses and check whether the benefit of learning the order grows.
- Reuse the same framework for patient-to-hospital allocation.
