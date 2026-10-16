"""Residual learning on top of the priority-rationing heuristic.

The agents no longer choose an order quantity from scratch. Each warehouse picks a
correction delta in {-R, ..., +R} that is added to the heuristic's base-stock order
before the heuristic's priority rationing is applied:

    want_i  = base-stock order of warehouse i (order up to S_i)
    order_i = clip(want_i + delta_i, 0, max_order)
    filled  = priority rationing of the orders (highest stockout penalty served first)

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


def heuristic_policy(cfg: EnvConfig):
    """The tuned priority-rationing base-stock policy used as the residual base."""
    offset = _tuned_offset(tuple(sorted(vars(cfg).items())))
    S = np.maximum(np.asarray(critical_ratio_levels(cfg)) + offset, 0)
    return PriorityBaseStockPolicy(S, cfg.stockout_cost, ReplenishmentEnv(cfg).capacity,
                                   max_order=cfg.max_order)


class ResidualEnv(ReplenishmentEnv):
    """Same dynamics and costs as ReplenishmentEnv; actions are corrections to the heuristic."""

    def __init__(self, cfg: EnvConfig, n_envs: int = 1, seed: int = 0, max_delta: int = 3):
        super().__init__(cfg, n_envs, seed)
        self.max_delta = max_delta
        self.n_actions = 2 * max_delta + 1  # action k means delta = k - max_delta
        self.base = heuristic_policy(cfg)
        self.obs_dim += 1  # heuristic order

    def heuristic_want(self):
        """Base-stock order of each warehouse before rationing, (n_envs, n_agents)."""
        return super(PriorityBaseStockPolicy, self.base).act(self)

    def heuristic_order(self):
        """What the heuristic itself would order after priority rationing."""
        return self.base.act(self)

    def apply_delta(self, actions):
        delta = np.asarray(actions) - self.max_delta
        want = np.clip(self.heuristic_want() + delta, 0, self.cfg.max_order)
        out = np.zeros_like(want)
        left = np.full(want.shape[0], self.base.capacity)
        for i in self.base.order:
            out[:, i] = np.minimum(want[:, i], left)
            left -= out[:, i]
        return out

    def step(self, actions):
        return super().step(self.apply_delta(actions))

    def _obs(self):
        base = super()._obs()
        h = self.heuristic_order()[..., None] / self._mu_scale
        return np.concatenate([base, h.astype(np.float32)], axis=-1)


def make_env(cfg: EnvConfig, n_envs: int, seed: int, residual: bool, max_delta: int = 3):
    if residual:
        return ResidualEnv(cfg, n_envs, seed, max_delta)
    return ReplenishmentEnv(cfg, n_envs, seed)


def bias_towards_zero(linear, max_delta, prior):
    """Adds `prior` to the logit of delta = 0 in a policy output layer."""
    if prior:
        with torch.no_grad():
            linear.bias[max_delta] += prior
