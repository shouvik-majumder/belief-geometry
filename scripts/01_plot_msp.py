"""Step 1: draw the ground-truth belief geometry (the fractal), with no network involved.

This is the target image. Everything after this is about finding it inside a transformer.

  python scripts/01_plot_msp.py --depth 11
  python scripts/01_plot_msp.py --process z1r --depth 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bg.msp import SIMPLEX_CORNERS, belief_rgb, box_counting_dimension, msp_cloud, plot_projection  # noqa: E402
from bg.process import PROCESSES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def draw_simplex(ax) -> None:
    """Outline of the triangle of all distributions over 3 states, with its corners labelled."""
    tri = np.vstack([SIMPLEX_CORNERS, SIMPLEX_CORNERS[:1]])
    ax.plot(tri[:, 0], tri[:, 1], color="0.7", lw=1)
    for i, (x, y) in enumerate(SIMPLEX_CORNERS):
        ax.annotate(f"certain: state {i}", (x, y), textcoords="offset points",
                    xytext=(0, 8 if y > 0 else -14), ha="center", fontsize=8, color="0.4")
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", default="mess3", choices=sorted(PROCESSES))
    ap.add_argument("--depth", type=int, default=11, help="how many tokens of history to enumerate")
    ap.add_argument("--x", type=float, default=0.15, help="mess3 transition parameter")
    ap.add_argument("--alpha", type=float, default=0.6, help="mess3 emission parameter")
    args = ap.parse_args()

    process = PROCESSES[args.process](args.x, args.alpha) if args.process == "mess3" else PROCESSES[args.process]()
    cloud = msp_cloud(process, depth=args.depth)
    beliefs, probs, depth = cloud["beliefs"], cloud["probs"], cloud["depth"]
    xy, _ = plot_projection(beliefs)
    simplex_plot = beliefs.shape[1] == 3
    print(f"{process.name}: {len(beliefs)} belief states enumerated to depth {args.depth}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    # Left: every reachable belief, coloured by which state it favours.
    ax = axes[0]
    draw_simplex(ax) if simplex_plot else ax.set_aspect("equal")
    ax.scatter(xy[:, 0], xy[:, 1], s=1.0, c=belief_rgb(beliefs), alpha=0.6, linewidths=0)
    ax.set_title(f"Belief states reachable in {args.depth} tokens\n({len(beliefs)} points, coloured by belief)", fontsize=10)

    # Right: the same points weighted by how often you would actually visit them.
    ax = axes[1]
    draw_simplex(ax) if simplex_plot else ax.set_aspect("equal")
    order = np.argsort(probs)
    ax.scatter(xy[order, 0], xy[order, 1], s=2.0, c=np.log10(np.clip(probs[order], 1e-12, None)),
               cmap="viridis", alpha=0.8, linewidths=0)
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array(np.log10(np.clip(probs, 1e-12, None)))
    fig.colorbar(sm, ax=ax, fraction=0.04, label="log10 probability of that history")
    ax.set_title("Same points, coloured by how likely the history is", fontsize=10)

    fig.suptitle(f"Mixed-state presentation of {process.name}: the ground-truth geometry", fontsize=12)
    fig.tight_layout()
    out = ROOT / "figures" / f"msp_{args.process}_depth{args.depth}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"figure -> {out}")

    # A crude fractal-dimension estimate by box counting, for the record.
    if len(beliefs) > 1000:
        slope = box_counting_dimension(xy)
        print(f"box-counting dimension of the cloud ~= {slope:.2f} (2.00 would fill the triangle)")


if __name__ == "__main__":
    main()
