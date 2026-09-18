"""A tiny transformer, trained from scratch, plus the training loop.

We use TransformerLens' `HookedTransformer` rather than plain PyTorch for one reason: it gives
`run_with_cache`, which hands you every internal activation by name (`blocks.3.hook_resid_post`
and friends). That is the standard tool in mechanistic interpretability, and the naming
convention is the same one you already met in the Gemma Scope SAE work.

The models here are deliberately minuscule (about 200k parameters). They train in minutes on a
3090 and still reach the information-theoretic floor for the process, which is the point: the
task is hard in a mathematical sense, not a scale sense.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig


@dataclass
class TrainConfig:
    process: str = "mess3"
    seq_len: int = 10
    n_layers: int = 4
    d_model: int = 64
    n_heads: int = 4
    d_mlp: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 512
    n_steps: int = 3000
    eval_every: int = 250
    seed: int = 0
    device: str = "cuda"

    def to_dict(self) -> dict:
        return asdict(self)


def build_model(cfg: TrainConfig, n_vocab: int) -> HookedTransformer:
    """A small decoder-only transformer with the same architecture family as real LLMs."""
    hooked_cfg = HookedTransformerConfig(
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_head=cfg.d_model // cfg.n_heads,
        d_mlp=cfg.d_mlp,
        d_vocab=n_vocab,
        n_ctx=cfg.seq_len,
        act_fn="relu",
        normalization_type="LN",
        seed=cfg.seed,
        device=cfg.device,
    )
    return HookedTransformer(hooked_cfg)


def train(model: HookedTransformer, process, cfg: TrainConfig, verbose: bool = True) -> dict:
    """Next-token prediction on freshly sampled sequences (so there is no train/test split to
    worry about: every batch is new data straight from the process).

    We log two things:
      loss            the model's cross-entropy, in nats
      optimal_loss    the same quantity for a perfect Bayesian predictor (the myopic entropy)
    The gap between them is what "has it learned the process?" means, quantitatively.
    """
    rng = np.random.default_rng(cfg.seed)
    device = cfg.device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # The loss scores predictions of tokens at positions 1..L-1 (position 0 has no context), so
    # the matching theoretical floor is the myopic entropy at those same positions. Getting this
    # index right matters: off by one and the model appears to beat the optimal predictor.
    myopic = process.myopic_entropy(cfg.seq_len)
    optimal = float(myopic[1:].mean())

    # A fixed evaluation set, so the reported loss is not the noise of one training batch.
    eval_tokens = torch.as_tensor(process.sample(4096, cfg.seq_len, np.random.default_rng(cfg.seed + 999)),
                                  device=device)

    @torch.no_grad()
    def evaluate() -> tuple[float, np.ndarray]:
        model.eval()
        logits = model(eval_tokens)
        per_pos = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            eval_tokens[:, 1:].reshape(-1),
            reduction="none",
        ).view(eval_tokens.shape[0], -1).mean(0)
        model.train()
        return float(per_pos.mean().item()), per_pos.cpu().numpy()

    history = {"step": [], "loss": [], "eval_loss": [], "optimal_loss": optimal,
               "myopic_entropy": myopic.tolist()}
    t0 = time.perf_counter()
    for step in range(1, cfg.n_steps + 1):
        tokens = torch.as_tensor(process.sample(cfg.batch_size, cfg.seq_len, rng), device=device)
        logits = model(tokens)  # [B, L, n_vocab]
        # predict token t+1 from positions 0..L-2
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            tokens[:, 1:].reshape(-1),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.eval_every == 0 or step == 1:
            ev, per_pos = evaluate()
            history["step"].append(step)
            history["loss"].append(float(loss.item()))
            history["eval_loss"].append(ev)
            if verbose:
                print(f"  step {step:5d}/{cfg.n_steps}  eval loss {ev:.4f}  "
                      f"optimal {optimal:.4f}  excess {ev - optimal:+.4f}  "
                      f"[{time.perf_counter() - t0:.0f}s]")
    _, per_pos = evaluate()
    history["per_position_loss"] = per_pos.tolist()
    history["seconds"] = time.perf_counter() - t0
    return history


@torch.no_grad()
def collect_activations(model: HookedTransformer, tokens: np.ndarray, layer: int | None = None,
                        hook: str = "resid_post", batch_size: int = 512) -> np.ndarray:
    """Run the model and return one activation vector per (sequence, position).

    layer=None means the last layer. The default hook `resid_post` is the residual stream after a
    block, the same place the ESR project steered and the same place Gemma Scope SAEs are trained.
    Returns [n_seq, seq_len, d_model] as a numpy array.
    """
    layer = model.cfg.n_layers - 1 if layer is None else layer
    name = f"blocks.{layer}.hook_{hook}"
    outs = []
    for i in range(0, len(tokens), batch_size):
        batch = torch.as_tensor(tokens[i : i + batch_size], device=model.cfg.device)
        _, cache = model.run_with_cache(batch, names_filter=[name])
        outs.append(cache[name].float().cpu().numpy())
    return np.concatenate(outs, axis=0)
