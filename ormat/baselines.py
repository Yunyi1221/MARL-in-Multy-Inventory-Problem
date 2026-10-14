"""Classic operations-research baselines: base-stock (order up to S) and (s, S) policies.

Each warehouse decides on its own and ignores the supplier capacity, except for
the priority-rationing heuristic. These are the reference lines MARL has to beat.
Usage: python -m ormat.baselines
"""
import argparse
import math

import numpy as np

from ormat.envs import EnvConfig, ReplenishmentEnv


def poisson_quantile(mean, q):
    """q-quantile of Poisson(mean), without scipy."""
    k, p, cdf = 0, math.exp(-mean), math.exp(-mean)
    while cdf < q:
        k += 1
        p *= mean / k
        cdf += p
    return k


class BaseStockPolicy:
    """Order up to S when the inventory position is below s; s = S gives a base-stock policy."""

    def __init__(self, S, s=None, max_order=12):
        self.S = np.asarray(S, dtype=np.float64)
        self.s = self.S if s is None else np.asarray(s, dtype=np.float64)
        self.max_order = max_order

    def act(self, env: ReplenishmentEnv):
        pos = env.inventory_position()
        order = np.where(pos < self.s, self.S - pos, 0.0)
        return np.clip(np.round(order), 0, self.max_order).astype(np.int64)


class PriorityBaseStockPolicy(BaseStockPolicy):
    """Capacity-aware heuristic: when total orders exceed capacity, serve warehouses in
    decreasing order of stockout penalty, so low-priority warehouses order less."""

    def __init__(self, S, priority, capacity, s=None, max_order=12):
        super().__init__(S, s, max_order)
        self.order = np.argsort(-np.asarray(priority))
        self.capacity = capacity

    def act(self, env: ReplenishmentEnv):
        want = super().act(env)
        out = np.zeros_like(want)
        left = np.full(want.shape[0], self.capacity)
        for i in self.order:
            out[:, i] = np.minimum(want[:, i], left)
            left -= out[:, i]
        return out


def critical_ratio_levels(cfg: EnvConfig):
    """Newsvendor levels: S_i = b/(b+h) quantile of demand over (lead_time + 1) periods."""
    periods = cfg.lead_time + 1
    return [poisson_quantile(mu * periods, b / (b + h))
            for mu, h, b in zip(cfg.demand_mean, cfg.holding_cost, cfg.stockout_cost)]


def evaluate_policy(policy, cfg: EnvConfig, n_episodes=200, seed=12345):
    env = ReplenishmentEnv(cfg, n_envs=n_episodes, seed=seed)
    env.reset()
    total_cost = np.zeros(n_episodes)
    done = False
    while not done:
        _, _, done, info = env.step(policy.act(env))
        total_cost += info["cost"]
    return total_cost.mean(), total_cost.std()


def grid_search(cfg: EnvConfig, n_episodes=200):
    """Grid search over a common offset delta around the critical-ratio levels and a reorder gap."""
    S_cr = np.asarray(critical_ratio_levels(cfg), dtype=np.float64)
    best = None
    for delta in range(-4, 5):
        S = np.maximum(S_cr + delta, 0)
        for gap in (0, 1, 2, 4):
            policy = BaseStockPolicy(S, S - gap, cfg.max_order)
            mean, std = evaluate_policy(policy, cfg, n_episodes, seed=999)  # separate seed for the search
            if best is None or mean < best[0]:
                best = (mean, std, delta, gap, S)
    return best


def best_baselines(cfg: EnvConfig, n_episodes=500):
    """Returns {name: (mean cost, std)}; all three baselines are scored on the evaluation seed."""
    capacity = ReplenishmentEnv(cfg).capacity
    S_cr = critical_ratio_levels(cfg)
    out = {"base-stock (critical ratio)": evaluate_policy(
        BaseStockPolicy(S_cr, max_order=cfg.max_order), cfg, n_episodes)}
    _, _, delta, gap, S = grid_search(cfg)
    out[f"(s,S) grid search, delta={delta} gap={gap}"] = evaluate_policy(
        BaseStockPolicy(S, S - gap, cfg.max_order), cfg, n_episodes)
    best = None
    for d in range(-4, 5):
        S_d = np.maximum(np.asarray(S_cr) + d, 0)
        pol = PriorityBaseStockPolicy(S_d, cfg.stockout_cost, capacity, max_order=cfg.max_order)
        m, _ = evaluate_policy(pol, cfg, 200, seed=999)
        if best is None or m < best[0]:
            best = (m, d, pol)
    out[f"priority-rationing base-stock, delta={best[1]}"] = evaluate_policy(best[2], cfg, n_episodes)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--capacity-ratio", type=float, default=EnvConfig.capacity_ratio)
    args = parser.parse_args()

    cfg = EnvConfig(capacity_ratio=args.capacity_ratio)
    print(f"supplier capacity C = {ReplenishmentEnv(cfg).capacity}, mean total demand = {sum(cfg.demand_mean)}")
    for name, (mean, std) in best_baselines(cfg, args.episodes).items():
        print(f"{name}: cost per episode {mean:.1f} ± {std:.1f}")


if __name__ == "__main__":
    main()
