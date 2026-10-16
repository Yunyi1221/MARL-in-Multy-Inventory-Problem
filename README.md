# MARL in Multi-Inventory Problems

Multi-agent reinforcement learning (Multi-Agent Transformer, MAT) for multi-warehouse inventory problems, measured against operations-research baselines and exact dynamic-programming optima.

## Contents

| Directory | What it contains |
|---|---|
| [`ormat/`](ormat/README.md) | A multi-warehouse replenishment environment with a capacity-limited supplier, a single-file MAT with five action orders (fixed / random / urgency / learned / none), MLP-based IPPO / MAPPO / HAPPO, PPO training, base-stock and (s,S) baselines, and experiment results |
| [`ormat/hw3/`](ormat/hw3/README.md) | Five two-warehouse extensions of the fixed-cost inventory problem from ORIE 5570 HW3 (independent, joint fixed cost, shared supplier, shared storage, two-echelon serial), each solved exactly by value iteration |

## Quick start

```bash
pip install -r requirements.txt

# Replenishment environment: baselines, training, plots
python -m ormat.baselines --capacity-ratio 0.85
python -m ormat.train --order-mode learned --seed 0 --capacity-ratio 0.85
python -m ormat.ppo_marl --algo mappo --seed 0 --capacity-ratio 0.85
python -m ormat.ppo_marl --algo mappo --residual --seed 0 --capacity-ratio 0.85   # learn corrections to the heuristic
python -m ormat.plot --results ormat/results_cap0.85 --capacity-ratio 0.85

# HW3 multi-warehouse versions: exact solutions and policy-region plot
python -m ormat.hw3.run
python -m ormat.hw3.plot
```

## Algorithm comparison

### Candidate algorithms

| Algorithm | Family | Training / execution | How agents coordinate | Status in this repo |
|---|---|---|---|---|
| IPPO | Independent learners | Each agent acts and evaluates on its local observation (shared parameters + agent ID) | None explicit; every other warehouse is part of a non-stationary environment | Implemented in [`ormat/ppo_marl.py`](ormat/ppo_marl.py) |
| MAPPO | CTDE policy gradient | Decentralized actors, centralized MLP critic on the global state | Through the shared value function only; all agents act simultaneously | Implemented in [`ormat/ppo_marl.py`](ormat/ppo_marl.py); `MAT-none` is the transformer analogue |
| HAPPO | CTDE, trust-region | Agents are updated one after another in a random order each iteration | Each agent's update is corrected by the importance weights of the agents updated before it, so joint improvement is monotonic | Implemented in [`ormat/ppo_marl.py`](ormat/ppo_marl.py) |
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
| OR urgency | Ranking computed from the state, e.g. days of cover, stockout probability, or marginal value V(s+1) − V(s) | **Main direction**: the order is given, not learned, because inventory costs are separable per warehouse and urgency is quantifiable. Implemented as `--order-mode urgency` with score bᵢ · (μᵢ(L+1) − inventory positionᵢ) |
| Learned | Scoring network + Plackett-Luce sampling (PMAT idea), trained with PPO | Implemented; a fuller comparison against OR urgency is deferred |

A dynamic order (random, urgency or learned) requires storing the per-step permutation in the rollout buffer and including an agent ID in each observation.

### Which problem structure suits MAT

**Multi-warehouse allocation with a shared, capacity-limited supplier (recommended starting point).** All warehouses order from the same supplier in the same period and compete for its capacity. This is exactly the coupling MAT's autoregressive decoder models: if an earlier warehouse orders more, a later one can order less in the same step. Costs are separable per warehouse and coupled only through linear capacity constraints, so an OR urgency ranking gives a meaningful decision order. The preliminary results above support this: with tight capacity every MAT variant beats capacity-unaware OR policies, while with loose capacity all methods tie.

**Serial multi-echelon chains (e.g. the beer game).** Here the coupling runs mainly through time: an upstream decision affects downstream stock only after one or more lead times. The hard parts are long-horizon credit assignment and the bullwhip effect, not same-step contention. Sequential decisions within one step therefore help little; MAPPO, possibly with a recurrent policy, is usually sufficient, and the natural order (downstream to upstream) is fixed rather than something to learn. Prior work on such chains mostly uses IPPO, MAPPO and HAPPO, and MAT has no clear structural advantage there.

These are expectations from the structure of the algorithms, not yet experimental results.

### First results

Shared-supplier environment, 4 warehouses, mean cost per episode over 200 evaluation episodes after 480k environment steps, 3 seeds. Lower is better. Full table and discussion in [`ormat/README.md`](ormat/README.md#preliminary-results).

| Method | Capacity 1.05 × demand | Capacity 0.85 × demand |
|---|---|---|
| priority-rationing base-stock (OR) | 1123 | **1272** |
| (s,S) grid search (OR) | 1116 | 1594 |
| IPPO | 1136 ± 8 | 1422 ± 14 |
| MAPPO | 1124 ± 3 | 1395 ± 50 |
| HAPPO | 1125 ± 4 | 1398 ± 42 |
| MAT-fixed | 1120 ± 10 | 1389 ± 16 |
| MAT-urgency | 1128 ± 5 | 1392 ± 5 |
| MAT-learned | 1133 ± 6 | 1362 ± 19 |
| MAT-none | 1129 ± 3 | 1376 ± 14 |

- With tight capacity, all MARL methods beat capacity-unaware (s,S) but none beats the hand-written priority heuristic.
- IPPO is the weakest learned method; MAPPO, HAPPO and the MAT variants are within noise of each other with 3 seeds.

### Residual learning

Learning a correction δ ∈ {−3, …, +3} on top of the priority heuristic (`--residual`, see [`ormat/README.md`](ormat/README.md#residual-learning-on-top-of-the-priority-heuristic)) makes every algorithm beat the heuristic: at capacity 0.85 residual IPPO / MAPPO / MAT-fixed reach 1226 / 1229 / 1230 against the heuristic's 1272, and at 1.05 residual MAT-fixed reaches 1087 against 1123. Once the heuristic is the base, the choice of MARL algorithm matters little.
- The OR urgency order did not help MAT here, likely because proportional rationing makes the decision order irrelevant to who receives capacity.

### Planned experiments

1. Switch rationing to first-come-first-served in decision order, so the agent order directly decides allocation, and rerun the comparison with 5 to 10 seeds.
2. Compare against the exact DP optimum on the two-warehouse HW3 versions.
3. Measure how much sequential decision-making gains over simultaneous action as capacity tightens and as the number of warehouses grows.
4. Extend to a two-echelon network (supplier → regional warehouses → stores) so that both same-step contention and cross-period coupling appear in one problem.

## Main findings (preliminary)

- When supplier capacity is tight, MAT with a learned decision order has the lowest cost of the MAT variants, and it learns to let the warehouse with the highest stockout penalty decide last. A hand-written priority-rationing heuristic is still better.
- In the coupled HW3 versions, applying the single-warehouse optimum (3,40) to each warehouse costs 4% to 13% more than the optimum. The optimal policies show three coordination patterns: replenishing together, taking turns, and yielding storage space.
