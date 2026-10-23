"""Stronger capacity-aware heuristics, to check whether residual RL beats a well-tuned baseline.

All policies compute a base-stock order per warehouse and then share the supplier capacity
by one of these rationing rules:

  priority   fixed order of decreasing stockout penalty b (the original heuristic)
  urgency    order of decreasing b_i * (mu_i * (L+1) - inventory position_i), recomputed every step
  marginal   unit by unit: the next unit goes to the warehouse with the largest newsvendor
             marginal value b * P(D > y) - h * P(D <= y), D ~ Poisson(mu * (L+1)),
             y = inventory position after the units already given; a warehouse never
             orders past its order-up-to level or past the point where that value
             becomes <= 0 (its critical-ratio level); stops when capacity runs out

The order-up-to levels are tuned per warehouse by coordinate descent on a separate seed
(the original heuristic only tunes one common offset).
Usage: python -m ormat.strong_heuristics --capacity-ratio 0.85
"""
import argparse
import functools
import math

import numpy as np

from ormat.baselines import critical_ratio_levels, evaluate_policy
from ormat.envs import EnvConfig, ReplenishmentEnv

RULES = ("priority", "urgency", "marginal")


def poisson_cdf_table(mean, k_max):
    """P(D <= k) for k = 0..k_max."""
    p = math.exp(-mean)
    cdf = [p]
    for k in range(1, k_max + 1):
        p *= mean / k
        cdf.append(cdf[-1] + p)
    return np.minimum(np.asarray(cdf), 1.0)


class RationedBaseStockPolicy:
    def __init__(self, cfg: EnvConfig, S, rule="priority"):
        if rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}")
        self.cfg, self.rule = cfg, rule
        self.S = np.asarray(S, dtype=np.float64)
        self.mu = np.asarray(cfg.demand_mean, dtype=np.float64)
        self.h = np.asarray(cfg.holding_cost, dtype=np.float64)
        self.b = np.asarray(cfg.stockout_cost, dtype=np.float64)
        self.capacity = ReplenishmentEnv(cfg).capacity
        self.S_cr = np.asarray(critical_ratio_levels(cfg), dtype=np.float64)
        self.k_max = int(max(self.S.max(), self.S_cr.max())) + cfg.max_order + 1
        self.cdf = np.stack([poisson_cdf_table(m * (cfg.lead_time + 1), self.k_max) for m in self.mu])

    def want(self, env: ReplenishmentEnv):
        """Order-up-to quantity per warehouse before rationing, (n_envs, n_agents)."""
        S = self.S
        if self.rule == "marginal":
            # a unit with non-positive marginal value is never ordered, which caps the
            # inventory position at the critical-ratio level
            S = np.minimum(S, self.S_cr)
        return np.clip(np.round(S - env.inventory_position()), 0, self.cfg.max_order)

    def act(self, env: ReplenishmentEnv):
        return self.ration(env, self.want(env))

    def ration(self, env: ReplenishmentEnv, want):
        """Shares the supplier capacity among the wanted quantities by this policy's rule."""
        E, N = want.shape
        out = np.zeros_like(want, dtype=np.float64)
        left = np.full(E, float(self.capacity))
        rows = np.arange(E)
        if self.rule == "marginal":
            y = env.inventory_position().copy()
            for _ in range(self.capacity):
                idx = np.clip(y, 0, self.k_max).astype(int)
                below = self.cdf[np.arange(N), idx]  # P(D <= y), per env and warehouse
                gain = self.b * (1 - below) - self.h * below
                gain = np.where((out < want) & (left[:, None] > 0), gain, -np.inf)
                i = gain.argmax(1)
                ok = np.isfinite(gain[rows, i])
                if not ok.any():
                    break
                out[rows[ok], i[ok]] += 1
                y[rows[ok], i[ok]] += 1
                left[ok] -= 1
            return out.astype(np.int64)
        if self.rule == "priority":
            order = np.tile(np.argsort(-self.b), (E, 1))
        else:
            order = np.argsort(-env.urgency(), axis=1, kind="stable")
        for k in range(N):
            i = order[:, k]
            give = np.minimum(want[rows, i], left)
            out[rows, i] = give
            left -= give
        return out.astype(np.int64)


@functools.lru_cache(maxsize=None)
def tuned_levels(cfg_key, rule):
    """Cached tune() for a config given as a sorted tuple of its fields."""
    return tune(EnvConfig(**dict(cfg_key)), rule)[0]


def tune(cfg: EnvConfig, rule, n_episodes=400, seed=999, span=6, passes=3):
    """Coordinate descent over per-warehouse offsets to the critical-ratio levels."""
    S_cr = np.asarray(critical_ratio_levels(cfg), dtype=np.float64)

    def cost(off):
        S = np.maximum(S_cr + off, 0)
        return evaluate_policy(RationedBaseStockPolicy(cfg, S, rule), cfg, n_episodes, seed)[0]

    # start from the best common offset, as in baselines.best_baselines
    best_c, best_off = min((cost(np.full(cfg.n_agents, d)), d) for d in range(-span, span + 1))
    off = np.full(cfg.n_agents, float(best_off))
    for _ in range(passes):
        changed = False
        for i in range(cfg.n_agents):
            for d in range(-span, span + 1):
                if d == off[i]:
                    continue
                trial = off.copy()
                trial[i] = d
                c = cost(trial)
                if c < best_c - 1e-9:
                    best_c, off, changed = c, trial, True
        if not changed:
            break
    return np.maximum(S_cr + off, 0), best_c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capacity-ratio", type=float, default=EnvConfig.capacity_ratio)
    p.add_argument("--episodes", type=int, default=200)
    args = p.parse_args()
    cfg = EnvConfig(capacity_ratio=args.capacity_ratio)
    for rule in RULES:
        S, tune_cost = tune(cfg, rule)
        mean, std = evaluate_policy(RationedBaseStockPolicy(cfg, S, rule), cfg, args.episodes)
        print(f"{rule:9s} S={S.astype(int).tolist()} tuning cost {tune_cost:7.1f} | eval cost {mean:7.1f} ± {std:5.1f}",
              flush=True)


if __name__ == "__main__":
    main()
