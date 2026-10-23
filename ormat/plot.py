"""Aggregate the learning curves under results/, plot them and print the final cost table.

Usage: python -m ormat.plot --results ormat/results
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

from ormat.baselines import best_baselines
from ormat.envs import EnvConfig


def method_label(mode):
    prefix = ""
    for tag, label in (("resm-", "residual (marginal base) "), ("res-", "residual ")):
        if mode.startswith(tag):
            prefix, mode = label, mode[len(tag):]
            break
    return prefix + (mode.upper() if mode in ("ippo", "mappo", "happo") else f"MAT-{mode}")


def load(results_dir):
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, "*_s*", "progress.csv"))):
        mode = os.path.basename(os.path.dirname(path)).rsplit("_s", 1)[0]
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            runs[mode].append(rows)
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="ormat/results")
    p.add_argument("--capacity-ratio", type=float, default=EnvConfig.capacity_ratio)
    p.add_argument("--ylim", type=float, nargs=2, default=None, help="zoom the cost axis, e.g. --ylim 1150 1350")
    args = p.parse_args()

    cfg = EnvConfig(capacity_ratio=args.capacity_ratio)
    baselines = best_baselines(cfg, 200)  # same number of evaluation episodes as train.py

    runs = load(args.results)
    print(f"{'method':<44}{'final eval cost':>16}{'seeds':>7}   mean decision position per warehouse")
    for name, (mean, _) in baselines.items():
        print(f"{name:<44}{mean:>16.1f}{'-':>7}")
    for mode, rs in runs.items():
        final = [float(r[-1]["eval_cost"]) for r in rs]
        ranks = ""
        if "rank_0" in rs[0][-1]:
            ranks = np.mean([[float(r[-1][f"rank_{i}"]) for i in range(cfg.n_agents)] for r in rs], axis=0)
            ranks = np.round(ranks, 2).tolist()
        print(f"{method_label(mode):<44}{np.mean(final):>9.1f} ± {np.std(final):<5.1f}{len(rs):>6}   {ranks}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the plot")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode, rs in runs.items():
        n = min(len(r) for r in rs)
        x = np.array([int(r["env_steps"]) for r in rs[0][:n]])
        y = np.array([[float(row["eval_cost"]) for row in r[:n]] for r in rs])
        m, s = y.mean(0), y.std(0)
        ax.plot(x, m, label=method_label(mode))
        ax.fill_between(x, m - s, m + s, alpha=0.2)
    labels = ["base-stock (critical ratio)", "(s,S) grid search", "priority base-stock"]
    for (mean, _), label, ls in zip(baselines.values(), labels, ("--", ":", "-.")):
        ax.axhline(mean, ls=ls, c="gray", label=label)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("eval cost per episode (lower is better)")
    if args.ylim:
        ax.set_ylim(*args.ylim)
    else:
        ax.set_ylim(top=2 * max(m for m, _ in baselines.values()))
    ax.set_title(f"capacity ratio = {cfg.capacity_ratio}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(args.results, "curves.png")
    fig.savefig(out, dpi=150)
    print(f"saved curves to {out}")


if __name__ == "__main__":
    main()
