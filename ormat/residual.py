"""Residual learning on top of the priority-rationing heuristic.

The agents no longer choose an order quantity from scratch. Each warehouse picks a
correction delta in {-R, ..., +R} that is added to the heuristic's base-stock order
before the heuristic's priority rationing is applied:

    want_i  = base-stock order of warehouse i (order up to S_i)
    order_i = clip(want_i + delta_i, 0, max_order)
    filled  = the heuristic's rationing of the orders (priority by stockout penalty, or
              marginal-value allocation with --residual-base marginal)

With delta = 0 everywhere the policy is exactly the priority heuristic, so the heuristic
is reachable by every learner. The policy head is initialised with a bias towards
delta = 0 so training starts close to the heuristic instead of at random corrections.
Each observation is extended with the heuristic's order for that warehouse.
"""
import functools

import numpy as np
import torch

from ormat.baselines import PriorityBaseStockPolicy, critical_ratio_levels, evaluate_policy
from ormat.envs import EnvConfig, ReplenishmentEnv
from ormat.strong_heuristics import RationedBaseStockPolicy, tuned_levels


@functools.lru_cache(maxsize=None)
def _tuned_offset(cfg_key):
    """Grid-searched offset of the priority heuristic, same search as baselines.best_baselines."""
    cfg = EnvConfig(**dict(cfg_key))
    capacity = ReplenishmentEnv(cfg).capacity
    S_cr = np.asarray(critical_ratio_levels(cfg))
    best = None
    for d in range(-4, 5):
        pol = PriorityBaseStockPolicy(np.maximum(S_cr + d, 0), cfg.stockout_cost, capacity,
                                      max_order=cfg.max_order)
        m, _ = evaluate_policy(pol, cfg, 200, seed=999)
        if best is None or m < best[0]:
            best = (m, d)
    return best[1]


def heuristic_policy(cfg: EnvConfig, rule: str = "priority"):
    """The heuristic used as the residual base.

    priority  the original heuristic: one common offset to the critical-ratio levels,
              capacity served in order of decreasing stockout penalty
    marginal  per-warehouse levels ((s, S) when there is a fixed ordering cost), capacity
              given unit by unit to the warehouse with the largest newsvendor marginal
              value (see strong_heuristics.py)
    """
    key = tuple(sorted(vars(cfg).items()))
    if rule == "priority":
        S = np.maximum(np.asarray(critical_ratio_levels(cfg)) + _tuned_offset(key), 0)
    else:
        S, s = tuned_levels(key, rule)
        return RationedBaseStockPolicy(cfg, S, rule, s)
    return RationedBaseStockPolicy(cfg, S, rule)


class ResidualEnv(ReplenishmentEnv):
    """Same dynamics and costs as ReplenishmentEnv; actions are corrections to the heuristic."""

    def __init__(self, cfg: EnvConfig, n_envs: int = 1, seed: int = 0, max_delta: int = 3,
                 base_rule: str = "priority"):
        super().__init__(cfg, n_envs, seed)
        self.max_delta = max_delta
        self.n_actions = 2 * max_delta + 1  # action k means delta = k - max_delta
        self.base = heuristic_policy(cfg, base_rule)
        self.obs_dim += 1  # heuristic order

    def heuristic_order(self):
        """What the heuristic itself would order after rationing."""
        return self.base.act(self)

    def apply_delta(self, actions):
        delta = np.asarray(actions) - self.max_delta
        want = np.clip(self.base.want(self) + delta, 0, self.cfg.max_order)
        return self.base.ration(self, want)

    def step(self, actions):
        return super().step(self.apply_delta(actions))

    def _obs(self):
        base = super()._obs()
        h = self.heuristic_order()[..., None] / self._mu_scale
        return np.concatenate([base, h.astype(np.float32)], axis=-1)


def make_env(cfg: EnvConfig, n_envs: int, seed: int, residual: bool, max_delta: int = 3,
             base_rule: str = "priority"):
    if residual:
        return ResidualEnv(cfg, n_envs, seed, max_delta, base_rule)
    return ReplenishmentEnv(cfg, n_envs, seed)


def bias_towards_zero(linear, max_delta, prior):
    """Adds `prior` to the logit of delta = 0 in a policy output layer."""
    if prior:
        with torch.no_grad():
            linear.bias[max_delta] += prior
