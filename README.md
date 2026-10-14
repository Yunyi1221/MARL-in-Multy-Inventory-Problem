# MARL in Multi-Inventory Problems

Multi-agent reinforcement learning (Multi-Agent Transformer, MAT) for multi-warehouse inventory problems, measured against operations-research baselines and exact dynamic-programming optima.

## Contents

| Directory | What it contains |
|---|---|
| [`ormat/`](ormat/README.md) | A multi-warehouse replenishment environment with a capacity-limited supplier, a single-file MAT with four action orders (fixed / random / learned / none), PPO training, base-stock and (s,S) baselines, and experiment results |
| [`ormat/hw3/`](ormat/hw3/README.md) | Five two-warehouse extensions of the fixed-cost inventory problem from ORIE 5570 HW3 (independent, joint fixed cost, shared supplier, shared storage, two-echelon serial), each solved exactly by value iteration |

## Quick start

```bash
pip install -r requirements.txt

# Replenishment environment: baselines, training, plots
python -m ormat.baselines --capacity-ratio 0.85
python -m ormat.train --order-mode learned --seed 0 --capacity-ratio 0.85
python -m ormat.plot --results ormat/results_cap0.85 --capacity-ratio 0.85

# HW3 multi-warehouse versions: exact solutions and policy-region plot
python -m ormat.hw3.run
python -m ormat.hw3.plot
```

## Main findings (preliminary)

- When supplier capacity is tight, MAT with a learned decision order has the lowest cost of the four MAT variants, and it learns to let the warehouse with the highest stockout penalty decide last. A hand-written priority-rationing heuristic is still better.
- In the coupled HW3 versions, applying the single-warehouse optimum (3,40) to each warehouse costs 4% to 13% more than the optimum. The optimal policies show three coordination patterns: replenishing together, taking turns, and yielding storage space.
