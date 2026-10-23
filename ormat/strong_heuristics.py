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

RULES = ("proportional", "priority", "urgency", "marginal")


def poisson_cdf_table(mean, k_max):
    """P(D <= k) for k = 0..k_max."""
    p = math.exp(-mean)
    cdf = [p]
    for k in range(1, k_max + 1):
        p *= mean / k
        cdf.append(cdf[-1] + p)
    return np.minimum(np.asarray(cdf), 1.0)


class RationedBaseStockPolicy:
    def __init__(self, cfg: EnvConfig, S, rule="priority", s=None):
        if rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}")
        self.cfg, self.rule = cfg, rule
        self.S = np.asarray(S, dtype=np.float64)
        self.s = None if s is None else np.asarray(s, dtype=np.float64)  # reorder points of an (s, S) policy
        self.mu = np.asarray(cfg.demand_mean, dtype=np.float64)
        self.h = np.asarray(cfg.holding_cost, dtype=np.float64)
        self.b = np.asarray(cfg.stockout_cost, dtype=np.float64)
        self.capacity = ReplenishmentEnv(cfg).capacity
        self.S_cr = np.asarray(critical_ratio_levels(cfg), dtype=np.float64)
        self.k_max = int(max(self.S.max(), self.S_cr.max())) + cfg.max_order + 1
        self.cdf = np.stack([poisson_cdf_table(m * (cfg.lead_time + 1), self.k_max) for m in self.mu])

    def want(self, env: ReplenishmentEnv):
        """Order-up-to quantity per warehouse before rationing, (n_envs, n_agents)."""
        pos = env.inventory_position()
        if self.s is not None:  # (s, S): order up to S only once the position drops below s
            return np.clip(np.round(np.where(pos < self.s, self.S - pos, 0.0)), 0, self.cfg.max_order)
        S = self.S
        if self.rule == "marginal":
            # a unit with non-positive marginal value is never ordered, which caps the
            # inventory position at the critical-ratio level
            S = np.minimum(S, self.S_cr)
        return np.clip(np.round(S - pos), 0, self.cfg.max_order)

    def act(self, env: ReplenishmentEnv):
        return self.ration(env, self.want(env))

    def ration(self, env: ReplenishmentEnv, want):
        """Shares the supplier capacity among the wanted quantities by this policy's rule."""
        E, N = want.shape
        out = np.zeros_like(want, dtype=np.float64)
        left = np.full(E, float(self.capacity))
        rows = np.arange(E)
        if self.rule == "proportional":  # the environment scales oversized orders down itself
            return want.astype(np.int64)
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
    """Cached tune() for a config given as a sorted tuple of its fields; returns (S, s)."""
    S, s, _ = tune(EnvConfig(**dict(cfg_key)), rule)
    return S, s


def tune(cfg: EnvConfig, rule, n_episodes=400, seed=999, span=6, passes=3):
    """Coordinate descent over per-warehouse levels, starting from the best common setting.

    Without a fixed ordering cost the policy is base-stock and only the offset to the
    critical-ratio level is tuned. With a fixed cost it is (s, S): the offset of S and the
    gap S - s are tuned per warehouse. Returns (S, s or None, tuning cost)."""
    S_cr = np.asarray(critical_ratio_levels(cfg), dtype=np.float64)
    N = cfg.n_agents
    use_gap = cfg.fixed_cost > 0
    offsets = range(-span, 2 * span + 1) if use_gap else range(-span, span + 1)
    gaps = range(0, cfg.max_order + 1, 2) if use_gap else (0,)

    def policy(off, gap):
        S = np.maximum(S_cr + off, 0)
        return RationedBaseStockPolicy(cfg, S, rule, s=S - gap if use_gap else None)

    def cost(off, gap):
        return evaluate_policy(policy(off, gap), cfg, n_episodes, seed)[0]

    best_c, d0, g0 = min((cost(np.full(N, d), np.full(N, g)), d, g) for d in offsets for g in gaps)
    off, gap = np.full(N, float(d0)), np.full(N, float(g0))
    for _ in range(passes):
        changed = False
        for i in range(N):
            for d in offsets:
                for g in gaps:
                    if d == off[i] and g == gap[i]:
                        continue
                    o2, g2 = off.copy(), gap.copy()
                    o2[i], g2[i] = d, g
                    c = cost(o2, g2)
                    if c < best_c - 1e-9:
                        best_c, off, gap, changed = c, o2, g2, True
        if not changed:
            break
    pol = policy(off, gap)
    return pol.S, pol.s, best_c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capacity-ratio", type=float, default=EnvConfig.capacity_ratio)
    p.add_argument("--fixed-cost", type=float, default=EnvConfig.fixed_cost)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--rules", nargs="+", choices=RULES, default=list(RULES))
    args = p.parse_args()
    cfg = EnvConfig(capacity_ratio=args.capacity_ratio, fixed_cost=args.fixed_cost)
    for rule in args.rules:
        S, s, tune_cost = tune(cfg, rule)
        mean, std = evaluate_policy(RationedBaseStockPolicy(cfg, S, rule, s), cfg, args.episodes)
        levels = f"S={S.astype(int).tolist()}" + ("" if s is None else f" s={s.astype(int).tolist()}")
        print(f"{rule:12s} {levels} tuning cost {tune_cost:7.1f} | eval cost {mean:7.1f} ± {std:5.1f}",
              flush=True)


if __name__ == "__main__":
    main()
