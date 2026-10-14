"""The single-warehouse inventory problem from ORIE 5570 HW3 and its multi-warehouse
extensions (two warehouses, small enough to solve exactly by dynamic programming).

Original problem (as in the homework):
  each period, order a first (arrives immediately) with inventory level y = s + a <= 40;
  demand d ~ p_d on 0..7;
  cost = 50*1[a>0] + 1*a + 0.1*(y-d)^+ + 10*(d-y)^+, next state (y-d)^+, discount 0.98.

Two-warehouse versions (identical warehouses, i.i.d. demand):
  independent  fully independent warehouses, costs add up (should reduce to two copies of (3,40))
  joint_fixed  joint replenishment: one major fixed cost of 50 whenever any warehouse orders,
               plus a minor fixed cost of 10 for each warehouse that orders
  supplier     shared supplier: total order of both warehouses <= 24 per period
  storage      shared storage: total inventory level of both warehouses <= 40
               (two products in one building)
  serial       two echelons: a retailer pulls stock from a central warehouse (shipping fixed
               cost 10, arrives immediately, limited by the warehouse's stock); the warehouse
               orders from an outside supplier (fixed cost 50 + unit cost 1, arrives next
               period); holding cost 0.1 at the retailer and 0.05 at the warehouse; only the
               retailer faces demand
"""
from dataclasses import dataclass

import numpy as np

CAP = 40
P_D = np.array([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1])
GAMMA = 0.98
FIXED, VAR, HOLD, PRICE = 50.0, 1.0, 0.1, 10.0

VERSIONS = ("independent", "joint_fixed", "supplier", "storage", "serial")


@dataclass
class Params:
    minor_fixed: float = 10.0       # joint_fixed: minor fixed cost per ordering warehouse
    supplier_cap: int = 24          # supplier: cap on the total order per period
    storage_cap: int = 40           # storage: cap on the total inventory level
    ship_fixed: float = 10.0        # serial: retailer shipping fixed cost
    wh_hold: float = 0.05           # serial: warehouse holding cost


def level_tables(cap=CAP, p_d=P_D, hold=HOLD, price=PRICE):
    """Precompute, per post-order level y, the expected holding + lost-sales cost L[y]
    and the transition matrix T[y, s']."""
    L = np.zeros(cap + 1)
    T = np.zeros((cap + 1, cap + 1))
    for y in range(cap + 1):
        for d, p in enumerate(p_d):
            L[y] += p * (hold * max(y - d, 0) + price * max(d - y, 0))
            T[y, max(y - d, 0)] += p
    return L, T


# ---------------------------------------------------------------------- #
# Original single-warehouse problem
# ---------------------------------------------------------------------- #
def solve_single(tol=1e-6):
    L, T = level_tables()
    s = np.arange(CAP + 1)[:, None]
    y = np.arange(CAP + 1)[None, :]
    a = y - s
    order = np.where(a > 0, FIXED + VAR * a, 0.0)
    feasible = a >= 0
    V = np.zeros(CAP + 1)
    while True:
        G = L + GAMMA * T @ V                       # G[y]
        Q = np.where(feasible, order + G[None, :], np.inf)
        V_new = Q.min(1)
        if np.abs(V_new - V).max() < tol:
            return V_new, Q.argmin(1) - np.arange(CAP + 1)
        V = V_new


# ---------------------------------------------------------------------- #
# Two warehouses: state (s1, s2); a decision is a post-order level (y1, y2) >= (s1, s2)
# ---------------------------------------------------------------------- #
def _order_cost_2(version, prm: Params):
    """Returns (C, F): ordering cost C[s1,s2,y1,y2] and feasibility mask F, shape (41,41,41,41)."""
    n = CAP + 1
    s1 = np.arange(n)[:, None, None, None]
    s2 = np.arange(n)[None, :, None, None]
    y1 = np.arange(n)[None, None, :, None]
    y2 = np.arange(n)[None, None, None, :]
    a1, a2 = y1 - s1, y2 - s2
    F = (a1 >= 0) & (a2 >= 0)
    o1, o2 = a1 > 0, a2 > 0
    if version == "joint_fixed":
        C = np.where(o1 | o2, FIXED, 0.0) + prm.minor_fixed * (o1.astype(float) + o2) + VAR * (a1 + a2)
    else:
        C = FIXED * (o1.astype(float) + o2) + VAR * (a1 + a2)
    if version == "supplier":
        F = F & (a1 + a2 <= prm.supplier_cap)
    if version == "storage":
        # states above the cap are unreachable (inventory only goes down); let them "order nothing"
        # so value iteration stays well defined
        F = F & ((y1 + y2 <= prm.storage_cap) | ((a1 == 0) & (a2 == 0)))
    return np.broadcast_to(C, (n,) * 4), np.broadcast_to(F, (n,) * 4)


def solve_pair(version, prm=Params(), tol=1e-6, max_iter=5000):
    """Exact optimum by value iteration. Returns V[s1,s2] and optimal post-order levels Y[s1,s2] = (y1, y2)."""
    assert version in VERSIONS and version != "serial"
    n = CAP + 1
    L, T = level_tables()
    C, F = _order_cost_2(version, prm)
    Lsum = L[:, None] + L[None, :]
    V = np.zeros((n, n))
    for _ in range(max_iter):
        G = Lsum + GAMMA * T @ V @ T.T              # G[y1,y2] = expected cost + discounted expected future value
        Q = np.where(F, C + G[None, None], np.inf).reshape(n, n, n * n)
        V_new = Q.min(-1)
        if np.abs(V_new - V).max() < tol:
            break
        V = V_new
    idx = Q.argmin(-1)
    return V_new, np.stack(np.unravel_index(idx, (n, n)), -1)


def evaluate_pair(version, Y, prm=Params(), tol=1e-8):
    """Exact discounted cost of a given policy Y[s1,s2] -> (y1,y2), by solving a linear system."""
    n = CAP + 1
    L, T = level_tables()
    C, F = _order_cost_2(version, prm)
    s1, s2 = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    y1, y2 = Y[..., 0], Y[..., 1]
    if not F[s1, s2, y1, y2].all():
        raise ValueError("the policy contains infeasible actions")
    r = C[s1, s2, y1, y2] + L[y1] + L[y2]
    # P[(s1,s2) -> (t1,t2)] = T[y1,t1] * T[y2,t2]
    P = np.einsum("ia,ib->iab", T[y1.ravel()], T[y2.ravel()]).reshape(n * n, n * n)
    V = np.linalg.solve(np.eye(n * n) - GAMMA * P, r.ravel())
    return V.reshape(n, n)


def independent_sS_policy(version, s=3, S=40, prm=Params()):
    """Apply the single-warehouse optimum (s,S) to each warehouse; when a coupling constraint
    binds, scale the orders down proportionally until they are feasible."""
    n = CAP + 1
    Y = np.zeros((n, n, 2), dtype=int)
    for s1 in range(n):
        for s2 in range(n):
            y = [S if s1 <= s else s1, S if s2 <= s else s2]
            a = [y[0] - s1, y[1] - s2]
            if version == "supplier" and sum(a) > prm.supplier_cap:
                # split the capacity in proportion to the requested amounts
                tot = sum(a)
                a = [a[0] * prm.supplier_cap // tot, a[1] * prm.supplier_cap // tot]
            if version == "storage" and s1 + a[0] + s2 + a[1] > prm.storage_cap:
                room = max(prm.storage_cap - s1 - s2, 0)
                tot = sum(a)
                a = [a[0] * room // tot, a[1] * room // tot] if tot else a
            Y[s1, s2] = [s1 + a[0], s2 + a[1]]
    return Y


# ---------------------------------------------------------------------- #
# Two echelons: state (r, w) = (retailer stock, warehouse stock)
# ---------------------------------------------------------------------- #
def solve_serial(prm=Params(), tol=1e-6, max_iter=5000):
    """Decision: retailer post-order level y_r (shipment y_r - r <= w) and the warehouse's
    next-period stock w' = w - shipment + new order."""
    n = CAP + 1
    L, T = level_tables()
    r = np.arange(n)[:, None, None, None]
    w = np.arange(n)[None, :, None, None]
    yr = np.arange(n)[None, None, :, None]
    wn = np.arange(n)[None, None, None, :]
    ship = yr - r
    left = w - ship                     # warehouse stock left after shipping
    buy = wn - left                     # order from the outside supplier (arrives next period)
    F = (ship >= 0) & (left >= 0) & (buy >= 0)
    C = (np.where(ship > 0, prm.ship_fixed, 0.0) + np.where(buy > 0, FIXED + VAR * buy, 0.0)
         + prm.wh_hold * np.maximum(left, 0))
    C, F = np.broadcast_to(C, (n,) * 4), np.broadcast_to(F, (n,) * 4)
    V = np.zeros((n, n))
    for _ in range(max_iter):
        G = L[:, None] + GAMMA * T @ V              # G[y_r, w']
        Q = np.where(F, C + G[None, None], np.inf).reshape(n, n, n * n)
        V_new = Q.min(-1)
        if np.abs(V_new - V).max() < tol:
            break
        V = V_new
    idx = Q.argmin(-1)
    return V_new, np.stack(np.unravel_index(idx, (n, n)), -1)


def evaluate_serial(Y, prm=Params()):
    n = CAP + 1
    L, T = level_tables()
    r, w = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    yr, wn = Y[..., 0], Y[..., 1]
    ship = yr - r
    left = w - ship
    buy = wn - left
    assert (ship >= 0).all() and (left >= 0).all() and (buy >= 0).all()
    cost = (np.where(ship > 0, prm.ship_fixed, 0.0) + np.where(buy > 0, FIXED + VAR * buy, 0.0)
            + prm.wh_hold * left + L[yr])
    P = np.zeros((n * n, n * n))
    rows = np.arange(n * n)
    for t in range(n):
        P[rows, t * n + wn.ravel()] += T[yr.ravel(), t]
    V = np.linalg.solve(np.eye(n * n) - GAMMA * P, cost.ravel())
    return V.reshape(n, n)


def naive_serial_policy(s=3, S=40, ws=10, wS=40):
    """Retailer uses (s,S) and pulls as much as the warehouse has; the warehouse uses its own (ws, wS)."""
    n = CAP + 1
    Y = np.zeros((n, n, 2), dtype=int)
    for r in range(n):
        for w in range(n):
            ship = min(S - r, w) if r <= s else 0
            left = w - ship
            wn = wS if left <= ws else left
            Y[r, w] = [r + ship, wn]
    return Y
