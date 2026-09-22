"""Step 3: look for the fractal inside the transformer.

Procedure:
  1. sample fresh sequences and record the residual stream at every position
  2. compute the ground-truth belief at every position (pure maths, no network)
  3. fit a linear map from activations to beliefs, scored on held-out data
  4. plot the model's activations through that map, coloured the same way as the ground truth

If the two pictures match, the belief geometry is linearly present in the residual stream. Two
controls are printed alongside: shuffled labels, and the same probe on an untrained model.

  python scripts/03_belief_geometry.py --ckpt data/checkpoints/mess3_L4_d64_seed0.pt
  python scripts/03_belief_geometry.py --ckpt ... --layer 1     # look at an earlier layer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bg.model import TrainConfig, build_model, collect_activations  # noqa: E402
from bg.msp import belief_rgb, plot_projection  # noqa: E402
from bg.probe import (shuffle_control, token_history_features,  # noqa: E402
                      train_test_probe)
from bg.process import PROCESSES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module  # noqa: E402

draw_simplex = import_module("01_plot_msp").draw_simplex


def load(ckpt_path: Path):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**blob["cfg"])
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    pinfo = blob["process"]
    process = PROCESSES[pinfo["name"]](pinfo["x"], pinfo["alpha"]) if pinfo["name"] == "mess3" else PROCESSES[pinfo["name"]]()
    model = build_model(cfg, n_vocab=process.n_obs)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, process, cfg, blob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layer", type=int, default=None, help="which block's residual stream (default: last)")
    ap.add_argument("--hook", default="resid_post", choices=["resid_post", "resid_mid", "resid_pre"])
    ap.add_argument("--n-seq", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--history-k", type=int, default=6, help="tokens of raw history for the no-network control")
    args = ap.parse_args()

    model, process, cfg, blob = load(Path(args.ckpt))
    layer = cfg.n_layers - 1 if args.layer is None else args.layer
    rng = np.random.default_rng(args.seed)

    # 1 + 2: activations and ground-truth beliefs, aligned position by position.
    tokens = process.sample(args.n_seq, cfg.seq_len, rng)
    acts = collect_activations(model, tokens, layer=layer, hook=args.hook)   # [N, L, d_model]
    beliefs = process.beliefs_for_tokens(tokens)[:, 1:]                      # [N, L, n_states]
    X = acts.reshape(-1, acts.shape[-1])
    Y = beliefs.reshape(-1, beliefs.shape[-1])
    print(f"{process.name} | layer {layer} {args.hook} | {X.shape[0]} (sequence, position) pairs, d_model={X.shape[1]}")

    # 3: the probe, with controls.
    fit = train_test_probe(X, Y, seed=args.seed)
    print(f"linear probe R^2: train {fit['r2_train']:.4f}  held-out {fit['r2_test']:.4f}")
    print(f"  per belief coordinate (held out): {np.round(fit['r2_test_per_dim'], 4)}")
    print(f"  control, shuffled labels:         {shuffle_control(X, Y, seed=args.seed):.4f}  (expect ~0)")

    untrained = build_model(cfg, n_vocab=process.n_obs)
    acts_u = collect_activations(untrained, tokens, layer=layer, hook=args.hook)
    fit_u = train_test_probe(acts_u.reshape(-1, acts_u.shape[-1]), Y, seed=args.seed)
    print(f"  control, untrained model:         {fit_u['r2_test']:.4f}  (random net, embeddings + mixing)")
    hist = token_history_features(tokens, process.n_obs, k=args.history_k)
    fit_h = train_test_probe(hist, Y, seed=args.seed)
    print(f"  control, raw token history (k={args.history_k}):   {fit_h['r2_test']:.4f}  (NO network at all)")
    print("  -> read the model's R^2 against these, not against zero. The interesting quantity is")
    print("     how much the trained network adds over what the input already hands you.")

    # 4: the picture. Push activations through the probe to get predicted beliefs, then project
    #    truth and prediction with the SAME basis so the two panels are directly comparable.
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    pred_beliefs = Xb @ fit["W"]
    xy_truth, basis = plot_projection(Y)
    xy_model, _ = plot_projection(pred_beliefs, basis)
    colours = belief_rgb(Y)
    simplex_plot = Y.shape[1] == 3

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, xy, title in zip(
        axes[:2], [xy_truth, xy_model],
        ["Ground truth: Bayesian belief states", f"Transformer: residual stream, layer {layer}"],
    ):
        if simplex_plot:
            draw_simplex(ax)
        else:
            ax.set_aspect("equal"); ax.set_xlabel("belief PC1"); ax.set_ylabel("belief PC2")
        ax.scatter(xy[:, 0], xy[:, 1], s=0.6, c=colours, alpha=0.35, linewidths=0)
        ax.set_title(title, fontsize=10)

    ax = axes[2]
    ax.scatter(xy_truth[:, 0], xy_truth[:, 1], s=1.2, c="0.8", alpha=0.5, linewidths=0)
    ax.scatter(xy_model[:, 0], xy_model[:, 1], s=0.6, c=colours, alpha=0.25, linewidths=0)
    if simplex_plot:
        draw_simplex(ax)
    else:
        ax.set_aspect("equal")
    ax.set_title(f"Overlaid, grey = truth (held-out $R^2$ = {fit['r2_test']:.3f})", fontsize=10)

    # eval loss on the fixed 4096-sequence set, not the last training batch (which is noisy)
    excess = blob["history"]["eval_loss"][-1] - blob["history"]["optimal_loss"]
    fig.suptitle(f"{process.name}: belief geometry in a {cfg.n_layers}-layer, {cfg.d_model}-dim transformer "
                 f"(excess loss {excess:+.4f} nats)", fontsize=12)
    fig.tight_layout()
    out = ROOT / "figures" / f"geometry_{Path(args.ckpt).stem}_layer{layer}_{args.hook}.png"
    fig.savefig(out, dpi=160)
    print(f"figure -> {out}")

    # Layer sweep: where in the network does the belief appear?
    print(f"\nlayer sweep (held-out R^2 of the same probe, hook_{args.hook}):")
    y = process.beliefs_for_tokens(tokens[:2000])[:, 1:].reshape(-1, process.n_states)
    per_layer = []
    for L in range(cfg.n_layers):
        a = collect_activations(model, tokens[:2000], layer=L, hook=args.hook)
        per_layer.append(a.reshape(-1, a.shape[-1]))
        r2 = train_test_probe(per_layer[-1], y, seed=args.seed)["r2_test"]
        print(f"  blocks.{L}.hook_{args.hook}: {r2:.4f}")
    # Shai et al. report that for some processes the belief geometry is spread across layers,
    # so also probe all layers' residual streams concatenated.
    r2_cat = train_test_probe(np.concatenate(per_layer, axis=1), y, seed=args.seed)["r2_test"]
    print(f"  all layers concatenated: {r2_cat:.4f}")


if __name__ == "__main__":
    main()
