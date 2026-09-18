"""Step 2: train a tiny transformer from scratch on the process.

No pretrained weights, no downloads, no API. A few minutes on a 3090.

  python scripts/02_train.py                      # mess3, 4 layers, 3000 steps
  python scripts/02_train.py --process z1r --n-steps 1500
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bg.model import TrainConfig, build_model, train  # noqa: E402
from bg.process import PROCESSES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", default="mess3", choices=sorted(PROCESSES))
    ap.add_argument("--x", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the checkpoint name")
    args = ap.parse_args()

    cfg = TrainConfig(process=args.process, seq_len=args.seq_len, n_layers=args.n_layers,
                      d_model=args.d_model, n_heads=args.n_heads, n_steps=args.n_steps,
                      batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                      device="cuda" if torch.cuda.is_available() else "cpu")
    process = PROCESSES[args.process](args.x, args.alpha) if args.process == "mess3" else PROCESSES[args.process]()

    model = build_model(cfg, n_vocab=process.n_obs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{process.name}: training {cfg.n_layers}L {cfg.d_model}d transformer "
          f"({n_params/1e3:.0f}k parameters) on {cfg.device}")

    history = train(model, process, cfg)

    stem = f"{args.process}_L{cfg.n_layers}_d{cfg.d_model}_seed{cfg.seed}{args.tag}"
    ckpt_dir = ROOT / "data" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.to_dict(),
                "process": {"name": args.process, "x": args.x, "alpha": args.alpha},
                "history": history, "n_params": n_params}, ckpt_dir / f"{stem}.pt")

    final, optimal = history["eval_loss"][-1], history["optimal_loss"]
    print(f"\nfinal loss {final:.4f} nats | optimal (Bayesian) {optimal:.4f} | excess {final - optimal:+.4f}")
    print("an excess near zero means the model has effectively learned to track the belief state")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(history["step"], history["eval_loss"], marker="o", ms=3, label="transformer")
    ax.axhline(optimal, color="r", ls="--", label="optimal Bayesian predictor")
    ax.set_xlabel("training step"); ax.set_ylabel("cross-entropy (nats)")
    ax.set_title("Approaching the information-theoretic floor", fontsize=10)
    ax.legend()
    # Per-position: the optimal predictor gets better as history accumulates. If the model tracks
    # beliefs it must show the same downward staircase, not just the right average.
    ax = axes[1]
    pos = np.arange(1, len(history["per_position_loss"]) + 1)
    ax.plot(pos, history["per_position_loss"], marker="o", ms=3, label="transformer")
    ax.plot(pos, np.array(history["myopic_entropy"])[1:], marker="s", ms=3, ls="--", color="r",
            label="optimal (myopic entropy)")
    ax.set_xlabel("token position being predicted"); ax.set_ylabel("cross-entropy (nats)")
    ax.set_title("Per-position loss: does it match the Bayesian curve?", fontsize=10)
    ax.legend()
    fig.suptitle(process.name, fontsize=11)
    fig.tight_layout()
    out = ROOT / "figures" / f"loss_{stem}.png"
    fig.savefig(out, dpi=160)
    json.dump(history, open(ckpt_dir / f"{stem}_history.json", "w"), indent=2)
    print(f"checkpoint -> {ckpt_dir / (stem + '.pt')}\nfigure -> {out}")


if __name__ == "__main__":
    main()
