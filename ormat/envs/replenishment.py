"""Multi-warehouse replenishment environment (vectorized, pure numpy).

N warehouses share one supplier with limited capacity. At every time step:
  1. each warehouse picks an order quantity a_i in {0, 1, ..., max_order};
  2. if the total order exceeds the supplier capacity C, orders are rationed
     proportionally (rounded down);
  3. filled orders enter a pipeline of length lead_time; the head of the
     pipeline arrives into stock;
  4. demand ~ Poisson(mu_i) arrives; unmet demand is lost (lost sales);
  5. the team reward is -(holding + stockout + ordering cost), shared by all
     warehouses; the ordering cost is a per-unit cost plus an optional fixed cost
     for every warehouse that receives a non-zero order.

Warehouses are heterogeneous (different demand means and stockout penalties)
and the capacity is only slightly above the mean total demand, so "who orders
more and who orders less" has to be coordinated. This is exactly what
autoregressive decision making (MAT) is meant to handle.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EnvConfig:
    n_agents: int = 4
    horizon: int = 50
    lead_time: int = 2
    max_order: int = 12                      # action space size = max_order + 1
    demand_mean: tuple = (2.0, 4.0, 6.0, 8.0)
    holding_cost: tuple = (1.0, 1.0, 1.0, 1.0)
    stockout_cost: tuple = (2.0, 4.0, 8.0, 16.0)  # heterogeneous priority: higher penalty = more important
    order_cost: float = 0.0
    fixed_cost: float = 0.0                  # charged once per warehouse in every period it receives an order
    capacity_ratio: float = 1.05             # supplier capacity = ratio * mean total demand
    init_inventory_ratio: float = 1.0        # initial inventory = ratio * mu_i
    reward_scale: float = 0.01

    def __post_init__(self):
        for name in ("demand_mean", "holding_cost", "stockout_cost"):
            v = getattr(self, name)
            if len(v) != self.n_agents:
                raise ValueError(f"{name} has length {len(v)}, expected n_agents={self.n_agents}")


class ReplenishmentEnv:
    """Runs n_envs independent copies in a batch. Axis 0 of every array is the env, axis 1 the warehouse."""

    def __init__(self, cfg: EnvConfig, n_envs: int = 1, seed: int = 0):
        self.cfg = cfg
        self.n_envs = n_envs
        self.n_agents = cfg.n_agents
        self.n_actions = cfg.max_order + 1
        self.rng = np.random.default_rng(seed)

        self.mu = np.asarray(cfg.demand_mean, dtype=np.float64)
        self.h = np.asarray(cfg.holding_cost, dtype=np.float64)
        self.b = np.asarray(cfg.stockout_cost, dtype=np.float64)
        self.capacity = int(round(cfg.capacity_ratio * self.mu.sum()))

        # obs = [inventory, pipeline (lead_time), last demand, last filled order, mu, h, b, capacity, time left]
        self.obs_dim = 1 + cfg.lead_time + 2 + 3 + 2
        self._mu_scale = float(self.mu.max())

    # ------------------------------------------------------------------ #
    def reset(self):
        E, N, L = self.n_envs, self.n_agents, self.cfg.lead_time
        self.t = 0
        self.inventory = np.tile(np.round(self.cfg.init_inventory_ratio * self.mu), (E, 1))
        self.pipeline = np.tile(np.round(self.mu)[:, None], (E, 1, L)).astype(np.float64)
        self.last_demand = np.tile(self.mu, (E, 1))
        self.last_filled = np.tile(self.mu, (E, 1))
        return self._obs()

    def step(self, actions):
        """actions: (n_envs, n_agents) int. Returns obs, reward (n_envs,), done (bool), info."""
        cfg = self.cfg
        a = np.clip(np.asarray(actions), 0, cfg.max_order).astype(np.float64)

        # 1) proportional rationing at the supplier
        total = a.sum(axis=1, keepdims=True)
        scale = np.where(total > self.capacity, self.capacity / np.maximum(total, 1.0), 1.0)
        filled = np.floor(a * scale)

        # 2) advance the pipeline; the head arrives
        arrived = self.pipeline[:, :, 0].copy()
        self.pipeline[:, :, :-1] = self.pipeline[:, :, 1:]
        self.pipeline[:, :, -1] = filled
        self.inventory += arrived

        # 3) demand and lost sales
        demand = self.rng.poisson(self.mu, size=(self.n_envs, self.n_agents)).astype(np.float64)
        sold = np.minimum(self.inventory, demand)
        lost = demand - sold
        self.inventory -= sold

        holding = (self.h * self.inventory).sum(axis=1)
        stockout = (self.b * lost).sum(axis=1)
        ordering = cfg.order_cost * filled.sum(axis=1) + cfg.fixed_cost * (filled > 0).sum(axis=1)
        cost = holding + stockout + ordering

        self.last_demand = demand
        self.last_filled = filled
        self.last_lost = lost
        self.t += 1
        done = self.t >= cfg.horizon
        info = {
            "cost": cost,
            "holding": holding,
            "stockout": stockout,
            "ordering": ordering,
            "lost_units": lost.sum(axis=1),
            "rationed": (total[:, 0] > self.capacity).astype(np.float64),
        }
        return self._obs(), -cost * cfg.reward_scale, done, info

    # ------------------------------------------------------------------ #
    def _obs(self):
        E, N = self.n_envs, self.n_agents
        s = self._mu_scale
        cols = [
            self.inventory[..., None] / s,
            self.pipeline / s,
            self.last_demand[..., None] / s,
            self.last_filled[..., None] / s,
            np.broadcast_to(self.mu / s, (E, N))[..., None],
            np.broadcast_to(self.h / self.h.max(), (E, N))[..., None],
            np.broadcast_to(self.b / self.b.max(), (E, N))[..., None],
            np.full((E, N, 1), self.capacity / (s * N)),
            np.full((E, N, 1), 1.0 - self.t / self.cfg.horizon),
        ]
        return np.concatenate(cols, axis=-1).astype(np.float32)

    def inventory_position(self):
        """On-hand plus pipeline inventory, used by the baseline policies."""
        return self.inventory + self.pipeline.sum(axis=-1)

    def urgency(self):
        """OR urgency score per warehouse, (n_envs, n_agents): expected shortfall of the
        inventory position against demand over lead time + 1, weighted by the stockout penalty."""
        shortfall = self.mu * (self.cfg.lead_time + 1) - self.inventory_position()
        return self.b * shortfall
