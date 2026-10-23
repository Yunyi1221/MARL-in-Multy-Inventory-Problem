# ormat: multi-warehouse replenishment with a Multi-Agent Transformer

A small project that uses a minimal MAT (and IPPO / MAPPO / HAPPO for comparison) to study whether the order in which agents choose their actions matters in an operations-research problem.

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
| MAT-urgency | Order given each step by the OR urgency score bᵢ · (μᵢ(L+1) − inventory positionᵢ), most urgent first |
| MAT-learned | A scoring network + Plackett-Luce sampling learns the order (the core idea of PMAT); the order is trained with PPO as part of the action |
| MAT-none | No autoregression; all warehouses act simultaneously (MAPPO-style) |
| IPPO | `ppo_marl.py`: shared MLP actor and per-agent critic on the local observation + agent ID |
| MAPPO | `ppo_marl.py`: shared MLP actor + centralized MLP critic on all observations |
| HAPPO | `ppo_marl.py`: one MLP actor per warehouse + centralized critic; actors updated sequentially in a random order with the ratio-product correction |

## Running

```bash
pip install -r requirements.txt
python -m ormat.baselines --capacity-ratio 0.85
python -m ormat.train --order-mode learned --seed 0 --capacity-ratio 0.85   # about 5 minutes on one CPU thread
python -m ormat.ppo_marl --algo mappo --seed 0 --capacity-ratio 0.85        # ippo / mappo / happo, about 1 to 2 minutes
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
| MAT-urgency | 1128 ± 5 | 1392 ± 5 |
| IPPO | 1136 ± 8 | 1422 ± 14 |
| MAPPO | 1124 ± 3 | 1395 ± 50 |
| HAPPO | 1125 ± 4 | 1398 ± 42 |

Learning curves: `results/curves.png` and `results_cap0.85/curves.png`.

Observations:

1. **With loose capacity, the order does not matter.** Every method ties with base-stock, so this setting cannot separate the methods.
2. **With tight capacity**, every MAT variant is far better than base-stock and (s,S), which ignore the capacity. The learned order is the best of the MAT variants, but only by one to two standard deviations; more seeds are needed.
3. The learned order puts **warehouse 0, with the lowest stockout penalty, first** (mean position 0.39) and the high-penalty warehouses last. One explanation is that the most important warehouse decides after seeing everyone else's order.
4. The hand-written priority heuristic still beats every MAT variant by about 7%, so there is clear room for improvement.
5. **IPPO, MAPPO and HAPPO land in the same range as MAT.** IPPO is the worst learned method in both settings, which fits the expectation that some centralized information helps. MAPPO and HAPPO have the best single seeds at capacity 0.85 (1334 and 1352) but also the worst ones (1457 and 1454), so their means are no better than MAT's. With 3 seeds, none of the MARL methods is significantly different from the others.
6. **Feeding the OR urgency order into MAT did not help** (1392 ± 5, close to the fixed order). One likely reason: rationing is proportional, so the decision order cannot change who gets the capacity; it only changes what each warehouse can see. The order should matter more once rationing is first-come-first-served in decision order (see next steps).

## Residual learning on top of the priority heuristic

`ormat/residual.py` changes what the agents learn. Instead of choosing an order quantity from scratch, each warehouse picks a correction δ ∈ {−3, …, +3} that is added to the heuristic's base-stock order before the heuristic's priority rationing:

    order_i = clip(want_i + δ_i, 0, max_order),  want_i = S_i − inventory position_i
    filled  = priority rationing of the orders (highest stockout penalty first)

With δ = 0 everywhere the policy is exactly the tuned priority heuristic (checked: 1272.24 at capacity 0.85, identical to the heuristic). The policy's output layer gets a +2 logit bias on δ = 0, so training starts at the heuristic, and each observation also carries the heuristic's order. Every algorithm uses it through the `--residual` flag:

```bash
python -m ormat.train --order-mode fixed --residual --seed 0 --capacity-ratio 0.85
python -m ormat.ppo_marl --algo ippo --residual --seed 0 --capacity-ratio 0.85
python -m ormat.plot --results ormat/results_residual/cap0.85 --capacity-ratio 0.85 --ylim 1180 1340
```

Results (same budget and evaluation as above: 300 updates, 480k steps, 200 evaluation episodes, 3 seeds):

| Method | Capacity 1.05 × demand | Capacity 0.85 × demand |
|---|---|---|
| priority-rationing base-stock (the residual base) | 1123 | 1272 |
| best from-scratch MARL (table above) | 1120 ± 10 | 1362 ± 19 |
| residual IPPO | 1093 ± 2 | **1226 ± 4** |
| residual MAPPO | 1098 ± 2 | 1229 ± 6 |
| residual HAPPO | 1108 ± 4 | 1237 ± 7 |
| residual MAT-fixed | **1087 ± 3** | 1230 ± 3 |
| residual MAT-urgency | 1099 ± 6 | 1243 ± 10 |
| residual MAT-learned | 1103 ± 12 | 1233 ± 4 |

Curves: `results_residual/cap0.85/curves.png` and `results_residual/cap1.05/curves.png`.

Observations:

1. **Every residual method beats the heuristic in every seed**, by 2 to 4% at capacity 0.85 and 1 to 3% at 1.05. This is the first time any learned policy here beats the priority heuristic.
2. **The algorithms barely differ once the heuristic is the base.** At capacity 0.85 independent learners (IPPO) are as good as anything else; MAT's autoregressive coordination brings no measurable gain. A plausible reason: the heuristic's priority rationing already resolves the capacity conflict, so what is left to learn is mostly each warehouse's own state-dependent adjustment, which independent policies can represent.
3. The curves are still going down at 480k steps, so longer training may widen the gap to the heuristic.
4. The learned corrections are state-dependent rather than a constant shift: in the seed-0 IPPO and MAPPO policies at capacity 0.85, the two high-demand warehouses have δ ≠ 0 in 80 to 100% of steps while their mean correction stays between −0.3 and 0.

## Is the residual gain from learning or from a weak baseline?

The priority heuristic above tunes a single common offset and serves warehouses in a fixed order of stockout penalty. `ormat/strong_heuristics.py` tunes the order-up-to level of each warehouse separately (coordinate descent on a separate seed) and compares three rationing rules:

| Rationing rule | Capacity 1.05 × demand | Capacity 0.85 × demand |
|---|---|---|
| priority by b, one common offset (the residual base above) | 1123 | 1272 |
| priority by b, per-warehouse levels | 1123 | 1268 |
| urgency b·(μ(L+1) − inventory position), recomputed every step | 1137 | 1379 |
| **marginal value**: each unit goes to the warehouse with the largest b·P(D > y) − h·P(D ≤ y), D ~ Poisson(μ(L+1)) | **1097** | **1224** |
| best residual RL on the priority base (above) | 1087 | 1226 |

The marginal-value rule needs no training and matches the best residual RL at capacity 0.85. Per-warehouse tuning alone changes almost nothing; the rationing rule is what matters.

Residual RL on top of the marginal-value heuristic (`--residual-base marginal`, same budget, 3 seeds; results in `results_residual_marginal/`):

| Method | Capacity 1.05 × demand | Capacity 0.85 × demand |
|---|---|---|
| marginal-value heuristic (the base) | 1097 | 1224 |
| residual IPPO | 1094 ± 7 | 1224 ± 0 |
| residual MAPPO | 1098 ± 6 | 1223 ± 1 |
| residual HAPPO | 1095 ± 1 | 1224 ± 0 |
| residual MAT-fixed | 1093 ± 3 | 1224 ± 0 |

Observations:

1. **At capacity 0.85 learning adds nothing on top of the marginal-value heuristic.** The policies do make corrections (for example, in seed 0 of residual IPPO one warehouse has δ ≠ 0 in 95% of steps), but the cost does not move, so the heuristic sits on a flat optimum for corrections of this kind.
2. At capacity 1.05 residual RL gains at most 0.4%, within the seed spread.
3. So most of the gain reported in the previous section came from replacing a weak rationing rule, not from learning. A learned policy has to be compared against the marginal-value heuristic, not the priority heuristic.

## Next steps

- Use the marginal-value heuristic as the baseline to beat from now on.
- Add the fixed ordering cost from the course model (F = 50); this is where simple rationing rules are expected to break down and learning may help.
- Compare residual learning with a model-based rollout on the heuristic (simulate each candidate correction for a few periods and pick the cheapest).
- Train residual policies longer and with 5 to 10 seeds.
- Run 5 to 10 seeds per method and test for significance.
- Change the rationing rule to first-come-first-served in decision order, so the order directly affects allocation.
- Scale to 8 or 16 warehouses and check whether the benefit of learning the order grows.
- Reuse the same framework for patient-to-hospital allocation.
