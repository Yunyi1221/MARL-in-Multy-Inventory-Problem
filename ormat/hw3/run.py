"""Solve every version and compare with "each warehouse uses (3,40) on its own".
Usage: python -m ormat.hw3.run"""
import json
import os
import time

import numpy as np

from ormat.hw3.models import (Params, evaluate_pair, evaluate_serial, independent_sS_policy,
                              naive_serial_policy, solve_pair, solve_serial, solve_single)


STATES = [(0, 0), (3, 3), (3, 10), (3, 25), (10, 3), (5, 5), (10, 10), (20, 20)]


def describe_pair(Y):
    """Optimal order quantities [W1, W2] at a few representative states."""
    return {f"({a},{b})": [int(Y[a, b, 0] - a), int(Y[a, b, 1] - b)] for a, b in STATES}


def describe_serial(Y):
    """Optimal actions [shipment, warehouse order] at a few representative states (retailer, warehouse)."""
    out = {}
    for r, w in STATES:
        ship = int(Y[r, w, 0] - r)
        out[f"({r},{w})"] = [ship, int(Y[r, w, 1] - (w - ship))]
    return out


def best_naive_serial(prm):
    """Grid-search the serial heuristic: first the warehouse (ws, wS), then the retailer (s, S),
    choosing by V(0,0)."""
    best = None
    for ws in range(0, 41, 4):
        for wS in range(max(ws, 8), 41, 4):
            v = evaluate_serial(naive_serial_policy(3, 40, ws, wS), prm)[0, 0]
            if best is None or v < best[0]:
                best = (v, 3, 40, ws, wS)
    _, _, _, ws, wS = best
    for s in range(0, 9):
        for S in range(10, 41, 2):
            v = evaluate_serial(naive_serial_policy(s, S, ws, wS), prm)[0, 0]
            if v < best[0]:
                best = (v, s, S, ws, wS)
    return best[1:]


def main(out_dir=os.path.join(os.path.dirname(__file__), "results")):
    os.makedirs(out_dir, exist_ok=True)
    prm = Params()
    V1, pi1 = solve_single()
    print(f"single warehouse: V(0) = {V1[0]:.2f}, policy (s,S) = (3,40), orders in states 0-5: {pi1[:6].tolist()}")
    summary = {"single": {"V0": float(V1[0]), "policy": pi1.tolist()}}

    for version in ("independent", "joint_fixed", "supplier", "storage", "serial"):
        t0 = time.time()
        if version == "serial":
            V, Y = solve_serial(prm)
            s_r, S_r, ws, wS = best_naive_serial(prm)
            Vh = evaluate_serial(naive_serial_policy(s_r, S_r, ws, wS), prm)
            heuristic = f"retailer ({s_r},{S_r}) + warehouse ({ws},{wS}), best by grid search"
        else:
            V, Y = solve_pair(version, prm)
            Vh = evaluate_pair(version, independent_sS_policy(version, prm=prm), prm)
            heuristic = "each warehouse uses (3,40)"
        gap = (Vh - V) / V * 100
        if version == "storage":  # reachable states only
            ii, jj = np.meshgrid(np.arange(41), np.arange(41), indexing="ij")
            gap = np.where(ii + jj <= prm.storage_cap, gap, np.nan)
        print(f"\n[{version}] solved in {time.time() - t0:.1f}s")
        print(f"  optimal V(0,0) = {V[0, 0]:.2f}   {heuristic}: {Vh[0, 0]:.2f}   gap {gap[0, 0]:.2f}%"
              f"   mean gap over states {np.nanmean(gap):.2f}%  max {np.nanmax(gap):.2f}%")
        acts = describe_serial(Y) if version == "serial" else describe_pair(Y)
        print("  optimal actions at representative states:", acts)
        np.savez_compressed(os.path.join(out_dir, f"{version}.npz"), V=V, Y=Y, V_heuristic=Vh)
        summary[version] = {"V00": float(V[0, 0]), "V00_heuristic": float(Vh[0, 0]), "heuristic": heuristic,
                            "gap00_pct": float(gap[0, 0]), "gap_mean_pct": float(np.nanmean(gap)),
                            "gap_max_pct": float(np.nanmax(gap)), "actions": acts}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
