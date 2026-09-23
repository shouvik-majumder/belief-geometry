"""Generate the summary figures (written to figures/summary/).

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
from bg.msp import (SIMPLEX_CORNERS, belief_rgb, box_counting_dimension,  # noqa: E402
                    msp_cloud, to_simplex_xy)
from bg.probe import token_history_features, train_test_probe  # noqa: E402
from bg.process import PROCESSES  # noqa: E402

OUT = ROOT / "figures" / "summary"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "data" / "probe_summary.json"

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
GREY, DARK, ACCENT = "0.65", "0.25", "#1f77b4"
PROCESS_NAMES = ("mess3", "rrxor")
SEEDS = (0, 1, 2)   # independently trained models per process; seed 0 is the one in the figures


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}")


def load_ckpt(name: str, device: str, seed: int = 0):
    path = ROOT / "data" / "checkpoints" / f"{name}_L4_d64_seed{seed}.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**blob["cfg"]); cfg.device = device
    pi = blob["process"]
    process = PROCESSES[pi["name"]](pi["x"], pi["alpha"]) if pi["name"] == "mess3" else PROCESSES[pi["name"]]()
    model = build_model(cfg, n_vocab=process.n_obs)
    model.load_state_dict(blob["state_dict"]); model.eval()
    return model, process, cfg, blob


def compute(device: str, n_seq: int, seed: int, eval_n: int = 40000) -> dict:
    """Probe R^2 for each process: every layer of the trained model, all layers concatenated,
    and both controls, repeated for each independently trained seed (each seed also gets its own
    fresh token sample and probe split, so the spread covers model, data and split variation)."""
    results = {}
    for name in PROCESS_NAMES:
        per_seed = []
        for s in SEEDS:
            model_s, process, cfg_s, _ = load_ckpt(name, device, s)
            tokens = process.sample(n_seq, cfg_s.seq_len, np.random.default_rng(seed + s))
            Y = process.beliefs_for_tokens(tokens)[:, 1:].reshape(-1, process.n_states)
            acts = [collect_activations(model_s, tokens, layer=L, hook="resid_post").reshape(len(Y), -1)
                    for L in range(cfg_s.n_layers)]
            layers_s = [train_test_probe(a, Y, seed=seed + s)["r2_test"] for a in acts]
            concat_s = train_test_probe(np.concatenate(acts, axis=1), Y, seed=seed + s)["r2_test"]
            untrained = build_model(cfg_s, n_vocab=process.n_obs)      # same init as the trained seed
            au = collect_activations(untrained, tokens, layer=cfg_s.n_layers - 1, hook="resid_post")
            untr_s = train_test_probe(au.reshape(len(Y), -1), Y, seed=seed + s)["r2_test"]
            hist_s = train_test_probe(token_history_features(tokens, process.n_obs, k=6), Y,
                                      seed=seed + s)["r2_test"]
            per_seed.append({"layers": layers_s, "concat": concat_s, "untrained": untr_s,
                             "token_history": hist_s})
            print(f"    {name} seed {s}: model {layers_s[-1]:.4f} | concat {concat_s:.4f} | "
                  f"untrained {untr_s:.4f} | token history {hist_s:.4f}")

        model, process, cfg, blob = load_ckpt(name, device, 0)
        layers = np.mean([p["layers"] for p in per_seed], axis=0).tolist()
        stat = lambda k: [p[k] for p in per_seed]
        r2_untrained, r2_hist = float(np.mean(stat("untrained"))), float(np.mean(stat("token_history")))

        # Recompute the per-position loss on a much larger evaluation set. With 4k sequences the
        # standard error per position (~0.008 nats) is as large as Mess3's entire dynamic range
        # (0.01 nats), so the stored curve looks noisy for reasons that have nothing to do with
        # the model.
        import torch.nn.functional as F
        big = torch.as_tensor(process.sample(int(eval_n), cfg.seq_len, np.random.default_rng(7)),
                              device=device)
        per_pos = []
        with torch.no_grad():
            for i in range(0, big.shape[0], 4096):
                chunk = big[i:i + 4096]
                lg = model(chunk)
                per_pos.append(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                                               chunk[:, 1:].reshape(-1), reduction="none")
                               .view(chunk.shape[0], -1).cpu().numpy())
        per_pos = np.concatenate(per_pos, axis=0)
        per_position = per_pos.mean(0)
        per_position_sem = per_pos.std(0, ddof=1) / np.sqrt(per_pos.shape[0])

        results[name] = {
            "layers": layers, "model": layers[-1], "untrained": r2_untrained, "token_history": r2_hist,
            "concat": float(np.mean(stat("concat"))), "seeds": list(SEEDS), "per_seed": per_seed,
            "layers_per_seed": [p["layers"] for p in per_seed],
            "model_per_seed": [p["layers"][-1] for p in per_seed],
            "untrained_per_seed": stat("untrained"), "token_history_per_seed": stat("token_history"),
            "concat_per_seed": stat("concat"),
            "n_states": process.n_states, "n_obs": process.n_obs, "seq_len": cfg.seq_len,
            "eval_loss": blob["history"]["eval_loss"][-1], "optimal_loss": blob["history"]["optimal_loss"],
            "per_position_loss": per_position.tolist(),
            "per_position_sem": per_position_sem.tolist(),
            "eval_n": int(eval_n),
            "myopic_entropy": blob["history"]["myopic_entropy"],
            "step": blob["history"]["step"], "loss_curve": blob["history"]["eval_loss"],
            "n_params": blob["n_params"],
        }
        print(f"  {name} (mean of {len(SEEDS)} seeds): model {layers[-1]:.4f} | "
              f"concat {results[name]['concat']:.4f} | untrained {r2_untrained:.4f} | "
              f"token history {r2_hist:.4f} | layers {np.round(layers, 3)}")
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
    ax.set_title("Belief trajectory of one Mess3 sequence\n"
                 f"tokens {' '.join(str(int(t)) for t in tokens[0][:12])}", fontsize=9)
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
    dim = box_counting_dimension(xy)
    ax.set_title(f"Mess3: {len(xy):,} reachable beliefs\n"
                 f"box-counting dimension ~{dim:.2f}", fontsize=9)

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
                 "first two principal components", fontsize=9)
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
        ax.errorbar(pos, r["per_position_loss"], yerr=1.96 * np.array(r["per_position_sem"]),
                    fmt="-o", ms=3, color=DARK, capsize=2, lw=1, label="transformer")
        ax.plot(pos, np.array(r["myopic_entropy"])[1:], "--s", ms=3, color=ACCENT, label="optimal")
        ax.set_ylabel("cross-entropy (nats)")
        ax.set_title(f"{name}: loss by token position (n={r['eval_n']:,}, 95% CI)", fontsize=9)
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
        spread = [np.array(res[name][f"{k}_per_seed"]) for k in keys]
        err = np.array([[v - s.min(), s.max() - v] for v, s in zip(vals, spread)]).T
        ax.bar(x + (i - 0.5) * w, vals, w, color=colour, edgecolor=DARK, label=name,
               yerr=err, capsize=3, error_kw={"lw": 1, "ecolor": DARK})
        for xi, v in zip(x + (i - 0.5) * w, vals):
            ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8.5)
    ax.set_ylabel("held-out $R^2$ of a linear probe\nfor the belief state")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    m = res["mess3"]
    ax.set_title("Linear belief probe by information source "
                 f"(mean, range over {len(m['seeds'])} seeds)", fontsize=9)
    save(fig, "bfig04_controls.png")


def fig_layers(res: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, colour, marker in (("mess3", GREY, "o"), ("rrxor", ACCENT, "s")):
        r = res[name]
        per = np.array(r["layers_per_seed"])
        ax.fill_between(range(per.shape[1]), per.min(0), per.max(0), color=colour, alpha=0.2, lw=0)
        ax.plot(range(len(r["layers"])), r["layers"], f"-{marker}", color=colour, ms=5, label=f"{name}: model")
        ax.plot(len(r["layers"]) - 0.6, r["concat"], marker, mfc="none", mec=colour, ms=7, mew=1.5)
        ax.text(len(r["layers"]) - 0.6, r["concat"] - 0.07, "all\nlayers", fontsize=6.5,
                color=colour, ha="center", va="top")
        ax.axhline(r["token_history"], color=colour, ls=":", lw=1.2)
        ax.text(len(r["layers"]) - 1.02, r["token_history"] + 0.02, f"{name}: no-network control",
                fontsize=7.5, color=colour, ha="right")
    ax.set_xticks(range(len(res["mess3"]["layers"])))
    ax.set_xlim(-0.3, len(res["mess3"]["layers"]) - 0.2)
    ax.set_xlabel("transformer block (residual stream after the block)")
    ax.set_ylabel("held-out $R^2$")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.02, 0.8))
    ax.set_title("Linear belief probe by depth", fontsize=9.5)
    save(fig, "bfig05_layers.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--n-seq", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--eval-n", type=int, default=40000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if CACHE.exists() and not args.force:
        res = json.loads(CACHE.read_text())
        print(f"using cached probe summary ({CACHE.name}); pass --force to recompute")
    else:
        print("computing probe summary...")
        res = compute(args.device, args.n_seq, args.seed, args.eval_n)
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
