"""Generate every figure used in the belief-geometry summary deck.

Runs the probe analysis for both processes across all layers and all controls, caches the
numbers, and draws the figures. Defaults to CPU because the models are tiny (201k parameters),
so this can run while the GPU is busy with something else.

  python scripts/04_summary_figures.py
  python scripts/04_summary_figures.py --device cuda --force
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bg.model import TrainConfig, build_model, collect_activations  # noqa: E402
from bg.msp import SIMPLEX_CORNERS, belief_rgb, msp_cloud, to_simplex_xy  # noqa: E402
from bg.probe import token_history_features, train_test_probe  # noqa: E402
from bg.process import PROCESSES  # noqa: E402

OUT = ROOT / "figures" / "deck"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "data" / "probe_summary.json"

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
GREY, DARK, ACCENT = "0.65", "0.25", "#1f77b4"
CKPTS = {"mess3": "mess3_L4_d64_seed0.pt", "rrxor": "rrxor_L4_d64_seed0.pt"}


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}")


def load_ckpt(name: str, device: str):
    blob = torch.load(ROOT / "data" / "checkpoints" / CKPTS[name], map_location="cpu", weights_only=False)
    cfg = TrainConfig(**blob["cfg"]); cfg.device = device
    pi = blob["process"]
    process = PROCESSES[pi["name"]](pi["x"], pi["alpha"]) if pi["name"] == "mess3" else PROCESSES[pi["name"]]()
    model = build_model(cfg, n_vocab=process.n_obs)
    model.load_state_dict(blob["state_dict"]); model.eval()
    return model, process, cfg, blob


def compute(device: str, n_seq: int, seed: int) -> dict:
    """Probe R^2 for each process: every layer of the trained model, plus both controls."""
    results = {}
    for name in CKPTS:
        model, process, cfg, blob = load_ckpt(name, device)
        rng = np.random.default_rng(seed)
        tokens = process.sample(n_seq, cfg.seq_len, rng)
        Y = process.beliefs_for_tokens(tokens)[:, 1:].reshape(-1, process.n_states)

        layers = []
        for L in range(cfg.n_layers):
            a = collect_activations(model, tokens, layer=L, hook="resid_post")
            layers.append(train_test_probe(a.reshape(-1, a.shape[-1]), Y, seed=seed)["r2_test"])

        untrained = build_model(cfg, n_vocab=process.n_obs)
        au = collect_activations(untrained, tokens, layer=cfg.n_layers - 1, hook="resid_post")
        r2_untrained = train_test_probe(au.reshape(-1, au.shape[-1]), Y, seed=seed)["r2_test"]
        hist = token_history_features(tokens, process.n_obs, k=6)
        r2_hist = train_test_probe(hist, Y, seed=seed)["r2_test"]

        results[name] = {
            "layers": layers, "model": layers[-1], "untrained": r2_untrained, "token_history": r2_hist,
            "n_states": process.n_states, "n_obs": process.n_obs, "seq_len": cfg.seq_len,
            "eval_loss": blob["history"]["eval_loss"][-1], "optimal_loss": blob["history"]["optimal_loss"],
            "per_position_loss": blob["history"]["per_position_loss"],
            "myopic_entropy": blob["history"]["myopic_entropy"],
            "step": blob["history"]["step"], "loss_curve": blob["history"]["eval_loss"],
            "n_params": blob["n_params"],
        }
        print(f"  {name}: model {layers[-1]:.3f} | untrained {r2_untrained:.3f} | "
              f"token history {r2_hist:.3f} | layers {np.round(layers, 3)}")
    return results


# ------------------------------------------------------------------- figures
def fig_trajectory(seed: int = 3) -> None:
    """One sequence: how the belief walks around the simplex as tokens arrive."""
    process = PROCESSES["mess3"]()
    rng = np.random.default_rng(seed)
    tokens = process.sample(1, 12, rng)
    beliefs = process.beliefs_for_tokens(tokens)[0]
    xy = to_simplex_xy(beliefs)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    tri = np.vstack([SIMPLEX_CORNERS, SIMPLEX_CORNERS[:1]])
    ax.plot(tri[:, 0], tri[:, 1], color="0.8", lw=1)
    for i, (x, y) in enumerate(SIMPLEX_CORNERS):
        ax.annotate(f"certain:\nstate {i}", (x, y), textcoords="offset points",
                    xytext=(0, 10 if y > 0 else -22), ha="center", fontsize=8, color="0.45")
    ax.plot(xy[:, 0], xy[:, 1], "-", color=GREY, lw=1, zorder=1)
    ax.scatter(xy[:, 0], xy[:, 1], s=45, c=belief_rgb(beliefs), zorder=2, edgecolors=DARK, linewidths=0.5)
    for t, (x, y) in enumerate(xy):
        ax.annotate(str(t), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7, color=DARK)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("A belief state is a point in the triangle.\nEach token moves it. "
                 f"(one sequence, tokens {list(tokens[0][:12])})", fontsize=9)
    save(fig, "bfig01_trajectory.png")


def fig_msp() -> None:
    """The set of ALL reachable beliefs: a fractal for Mess3, a handful of points for RRXOR."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
    mess = msp_cloud(PROCESSES["mess3"](), depth=11)
    xy = to_simplex_xy(mess["beliefs"])
    ax = axes[0]
    tri = np.vstack([SIMPLEX_CORNERS, SIMPLEX_CORNERS[:1]])
    ax.plot(tri[:, 0], tri[:, 1], color="0.8", lw=1)
    ax.scatter(xy[:, 0], xy[:, 1], s=0.5, c=belief_rgb(mess["beliefs"]), alpha=0.6, linewidths=0)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Mess3: {len(xy):,} reachable beliefs\nfractal, box-counting dimension ~1.46", fontsize=9)

    rr = msp_cloud(PROCESSES["rrxor"](), depth=12)
    b = rr["beliefs"]
    centred = b - b.mean(0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    xy2 = centred @ Vt[:2].T
    ax = axes[1]
    ax.scatter(xy2[:, 0], xy2[:, 1], s=6, c=belief_rgb(b), alpha=0.7, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlabel("belief PC1"); ax.set_ylabel("belief PC2")
    ax.set_title(f"RRXOR: {len(np.unique(np.round(b, 6), axis=0))} distinct beliefs\n"
                 "discrete, 5 states so shown as PCA", fontsize=9)
    save(fig, "bfig02_msp.png")


def fig_training(res: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6))
    for row, name in enumerate(("mess3", "rrxor")):
        r = res[name]
        ax = axes[row, 0]
        ax.plot(r["step"], r["loss_curve"], "-o", ms=3, color=DARK, label="transformer")
        ax.axhline(r["optimal_loss"], color=ACCENT, ls="--", lw=1.4, label="optimal Bayesian predictor")
        ax.set_ylabel("cross-entropy (nats)")
        ax.set_title(f"{name}: training ({r['n_params']/1e3:.0f}k parameters)", fontsize=9)
        if row == 1:
            ax.set_xlabel("training step")
        ax.legend(fontsize=7.5)

        ax = axes[row, 1]
        pos = np.arange(1, len(r["per_position_loss"]) + 1)
        ax.plot(pos, r["per_position_loss"], "-o", ms=3, color=DARK, label="transformer")
        ax.plot(pos, np.array(r["myopic_entropy"])[1:], "--s", ms=3, color=ACCENT, label="optimal")
        ax.set_ylabel("cross-entropy (nats)")
        ax.set_title(f"{name}: loss position by position\n"
                     f"(final gap {r['eval_loss'] - r['optimal_loss']:+.4f} nats)", fontsize=9)
        if row == 1:
            ax.set_xlabel("token position being predicted")
        ax.legend(fontsize=7.5)
    save(fig, "bfig03_training.png")


def fig_controls(res: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    groups = ["model residual stream", "untrained model", "raw token history\n(no network)"]
    keys = ["model", "untrained", "token_history"]
    x = np.arange(len(groups)); w = 0.36
    for i, (name, colour) in enumerate((("mess3", GREY), ("rrxor", ACCENT))):
        vals = [res[name][k] for k in keys]
        ax.bar(x + (i - 0.5) * w, vals, w, color=colour, edgecolor=DARK, label=name)
        for xi, v in zip(x + (i - 0.5) * w, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8.5)
    ax.set_ylabel("held-out $R^2$ of a linear probe\nfor the belief state")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8)
    ax.set_title("The same probe, three sources of information\n"
                 "For Mess3 the no-network control WINS: the geometry is free from the input.", fontsize=9.5)
    save(fig, "bfig04_controls.png")


def fig_layers(res: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, colour, marker in (("mess3", GREY, "o"), ("rrxor", ACCENT, "s")):
        r = res[name]
        ax.plot(range(len(r["layers"])), r["layers"], f"-{marker}", color=colour, ms=5, label=f"{name}: model")
        ax.axhline(r["token_history"], color=colour, ls=":", lw=1.2)
        ax.text(len(r["layers"]) - 1.02, r["token_history"] + 0.02, f"{name}: no-network control",
                fontsize=7.5, color=colour, ha="right")
    ax.set_xticks(range(len(res["mess3"]["layers"])))
    ax.set_xlabel("transformer block (residual stream after the block)")
    ax.set_ylabel("held-out $R^2$")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, loc="center right")
    ax.set_title("Where does the belief state get built?\nFlat = copied from the input. Rising = computed.", fontsize=9.5)
    save(fig, "bfig05_layers.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--n-seq", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if CACHE.exists() and not args.force:
        res = json.loads(CACHE.read_text())
        print(f"using cached probe summary ({CACHE.name}); pass --force to recompute")
    else:
        print("computing probe summary...")
        res = compute(args.device, args.n_seq, args.seed)
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(res, indent=2))

    fig_trajectory()
    fig_msp()
    fig_training(res)
    fig_controls(res)
    fig_layers(res)
    print(f"\nfigures in {OUT}")


if __name__ == "__main__":
    main()
