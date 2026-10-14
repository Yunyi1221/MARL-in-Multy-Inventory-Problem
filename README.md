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

## Algorithm comparison

### Candidate algorithms

| Algorithm | Family | Training / execution | How agents coordinate | Status in this repo |
|---|---|---|---|---|
| IPPO | Independent learners | Each agent has its own actor and critic on local observations | None explicit; every other warehouse is part of a non-stationary environment | Planned |
| MAPPO | CTDE policy gradient | Decentralized actors, centralized MLP critic on the global state | Through the shared value function only; all agents act simultaneously | Approximated by `MAT-none` (no autoregression) |
| HAPPO | CTDE, trust-region | Agents are updated one after another in a random order each iteration | Each agent's update is corrected by the importance weights of the agents updated before it, so joint improvement is monotonic | Planned |
| MAT | Sequence model | Transformer encoder (critic) + autoregressive decoder (actor) | At execution time, agent *k* sees the actions already chosen by agents 1..*k*−1 within the same step | Implemented in [`ormat/mat.py`](ormat/mat.py) |
| (s,S) / base-stock | OR policy | Closed-form or grid-searched levels per warehouse | None; capacity is ignored and orders are rationed proportionally | Implemented in [`ormat/baselines.py`](ormat/baselines.py) |
| Urgency-greedy (priority rationing) | OR heuristic | Base-stock orders, then capacity is allocated by an urgency ranking | Hand-written priority rule | Implemented (ranking by stockout penalty *b*) |
| Exact DP | Optimum | Value iteration on two-warehouse versions | Optimal | Implemented in [`ormat/hw3/`](ormat/hw3/README.md) |

### Agent order variants for MAT

MAT's decoder needs an order in which warehouses choose their actions. The variants compared are:

| Variant | Order source | Notes |
|---|---|---|
| Fixed | Warehouse index | Original MAT |
| Random | New permutation every step | Tests whether any specific order helps |
| OR urgency | Ranking computed from the state, e.g. days of cover, stockout probability, or marginal value V(s+1) − V(s) | **Main direction**: the order is given, not learned, because inventory costs are separable per warehouse and urgency is quantifiable |
| Learned | Scoring network + Plackett-Luce sampling (PMAT idea), trained with PPO | Implemented; a fuller comparison against OR urgency is deferred |

A dynamic order (random, urgency or learned) requires storing the per-step permutation in the rollout buffer and including an agent ID in each observation.

### Which problem structure suits MAT

**Multi-warehouse allocation with a shared, capacity-limited supplier (recommended starting point).** All warehouses order from the same supplier in the same period and compete for its capacity. This is exactly the coupling MAT's autoregressive decoder models: if an earlier warehouse orders more, a later one can order less in the same step. Costs are separable per warehouse and coupled only through linear capacity constraints, so an OR urgency ranking gives a meaningful decision order. The preliminary results above support this: with tight capacity every MAT variant beats capacity-unaware OR policies, while with loose capacity all methods tie.

**Serial multi-echelon chains (e.g. the beer game).** Here the coupling runs mainly through time: an upstream decision affects downstream stock only after one or more lead times. The hard parts are long-horizon credit assignment and the bullwhip effect, not same-step contention. Sequential decisions within one step therefore help little; MAPPO, possibly with a recurrent policy, is usually sufficient, and the natural order (downstream to upstream) is fixed rather than something to learn. Prior work on such chains mostly uses IPPO, MAPPO and HAPPO, and MAT has no clear structural advantage there.

These are expectations from the structure of the algorithms, not yet experimental results.

### Planned experiments

1. On the shared-supplier environment, compare IPPO, MAPPO, HAPPO and MAT (fixed / random / OR urgency / learned order) against (s,S), urgency-greedy and, for two warehouses, the exact DP optimum.
2. Measure how much sequential decision-making gains over simultaneous action as capacity tightens and as the number of warehouses grows.
3. Extend to a two-echelon network (supplier → regional warehouses → stores) so that both same-step contention and cross-period coupling appear in one problem.

## Main findings (preliminary)

- When supplier capacity is tight, MAT with a learned decision order has the lowest cost of the four MAT variants, and it learns to let the warehouse with the highest stockout penalty decide last. A hand-written priority-rationing heuristic is still better.
- In the coupled HW3 versions, applying the single-warehouse optimum (3,40) to each warehouse costs 4% to 13% more than the optimum. The optimal policies show three coordination patterns: replenishing together, taking turns, and yielding storage space.
