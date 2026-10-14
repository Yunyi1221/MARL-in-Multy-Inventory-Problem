"""Plot the optimal policy regions of the two-warehouse versions: who orders in each
state (s1, s2). Usage: python -m ormat.hw3.plot"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = os.path.dirname(__file__)
TITLES = {"independent": "Independent", "joint_fixed": "Joint fixed cost",
          "supplier": "Shared supplier (<= 24/period)", "storage": "Shared storage (<= 40 total)"}
COLORS = ["#e4e6e8", "#2a78d6", "#eb6834", "#1baf7a", "#ffffff"]   # none / only W1 / only W2 / both / unreachable
LABELS = ["no order", "only W1 orders", "only W2 orders", "both order"]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.6), constrained_layout=True)
    for ax, version in zip(axes.flat, TITLES):
        d = np.load(os.path.join(HERE, "results", f"{version}.npz"))
        Y = d["Y"]
        s1, s2 = np.meshgrid(np.arange(41), np.arange(41), indexing="ij")
        o1, o2 = Y[..., 0] > s1, Y[..., 1] > s2
        cat = np.select([o1 & o2, o1, o2], [3, 1, 2], 0).astype(float)
        if version == "storage":
            cat[s1 + s2 > 40] = 4
        ax.imshow(cat.T, origin="lower", cmap=ListedColormap(COLORS), vmin=0, vmax=4,
                  extent=(-0.5, 40.5, -0.5, 40.5))
        ax.set_title(TITLES[version], fontsize=11)
        ax.set_xlabel("W1 inventory $s_1$")
        ax.set_ylabel("W2 inventory $s_2$")
        ax.set_xlim(-0.5, 20.5)
        ax.set_ylim(-0.5, 20.5)
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(COLORS, LABELS)],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), frameon=False)
    out = os.path.join(HERE, "results", "policy_regions.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
