# Multi-warehouse versions of the HW3 inventory problem

The starting point is the single-warehouse inventory problem from ORIE 5570 HW3: capacity 40, fixed ordering cost 50, unit cost 1, holding cost 0.1, lost-sales penalty 10, demand distribution `[0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]` on 0 to 7, discount 0.98, and orders arrive immediately. The optimal policy is (s,S) = (3,40); `solve_single()` reproduces the homework result (V(0) = 534.71).

The problem is extended here to five two-warehouse versions. With 41×41 states and 41×41 actions, value iteration still solves each one exactly, so every version has a **true optimum** to test heuristics and MARL against.

| Version | Coupling | Optimal V(0,0) | Heuristic | Heuristic V(0,0) | Gap (at (0,0) / mean over states / max) |
|---|---|---|---|---|---|
| independent | none, costs add up | 1069.42 | each warehouse uses (3,40) | 1069.42 | 0 / 0 / 0 |
| joint_fixed | one major fixed cost of 50 when any warehouse orders, plus 10 per ordering warehouse | 966.17 | each warehouse uses (3,40) | 1096.00 | 13.4% / 15.9% / 18.5% |
| supplier | total order of both warehouses ≤ 24 per period | 1268.23 | each warehouse uses (3,40), scaled down proportionally when over the cap | 1351.73 | 6.6% / 4.1% / 7.9% |
| storage | total inventory of both warehouses ≤ 40 | 1310.51 | each warehouse uses (3,40), scaled down proportionally when over the cap | 1381.70 | 5.4% / 5.9% / 10.7% |
| serial | retailer pulls from a central warehouse, which orders from outside (arrives next period) | 621.79 | retailer (5,38) + warehouse (0,40), best by grid search | 648.21 | 4.3% / 6.6% / 10.9% |

Heuristic costs are computed exactly by solving a linear system, not by simulation. For the storage version, the mean over states only counts reachable states (total inventory ≤ 40).

## What the optimal policies look like

`results/policy_regions.png` shows, for each state (s₁, s₂) of the four two-warehouse versions, which warehouses order:

- **independent**: two straight lines s₁ ≤ 3 and s₂ ≤ 3, i.e. two copies of (3,40).
- **joint_fixed**: whenever one warehouse needs to reorder, the other one fills up too (there is no "only one orders" region), and the reorder point drops from 3 to 2, because replenishing both at once is cheaper and makes waiting a little longer worthwhile. This is the can-order structure known from the joint replenishment problem.
- **supplier**: the two warehouses never order at the same time. They take turns taking the full 24 units of one period, instead of paying two fixed costs for half each.
- **storage**: each warehouse's reorder point depends on the other's stock. The fuller the other warehouse, the lower the reorder point (it reorders later), and the order size is limited by the remaining space. Both order together only when both are nearly empty.
- **serial**: at (0,0) the warehouse orders a full 40; the retailer pulls stock only when it drops to about 5, and then takes everything the warehouse has.

These structures map onto different kinds of multi-agent coordination: joint_fixed needs synchronization, supplier and storage need staggering and deciding who goes first, and serial needs information to flow between upstream and downstream.

## Running

```bash
python -m ormat.hw3.run    # solve the five versions and evaluate the heuristics, about 30 seconds; writes results/
python -m ormat.hw3.plot   # plot the policy regions
```

## Next step

These versions can serve directly as a MARL testbed: treat each warehouse as an agent, train MAT, and compare against the exact optima here to see precisely how far MARL is from optimal. With three or more warehouses DP becomes intractable, which is where MARL is actually needed.
