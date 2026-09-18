"""Linear probes: is the belief state written linearly in the residual stream?

A *probe* is a small supervised model fitted from activations to a quantity you care about. Here
the quantity is the ground-truth belief (3 numbers summing to 1) and the probe is ordinary least
squares, so the question is precise:

    is there a fixed linear map W with  belief ≈ activation @ W + b ?

Linear matters. Any sufficiently flexible probe can extract almost anything from almost anything,
which tells you nothing about how the model uses it. A *linear* readout is the operation the rest
of the network can actually perform cheaply, so a good linear fit is evidence that the belief is
represented, not merely recoverable.

Two controls are included, because a probe with no control is a story, not a result:
  shuffle    fit the same probe to shuffled labels (destroys any real relationship)
  untrained  fit the same probe to a randomly initialised model's activations
"""
from __future__ import annotations

import numpy as np


def fit_linear_probe(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-6) -> dict:
    """Least-squares map from activations X [N, d] to targets Y [N, k], with a bias term.

    Returns the weights, the predictions, and R^2 both overall and per target dimension.
    R^2 = 1 means the activations determine the target exactly; 0 means no better than predicting
    the mean; negative means worse than the mean.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    A = Xb.T @ Xb + ridge * np.eye(Xb.shape[1])
    W = np.linalg.solve(A, Xb.T @ Y)  # [d + 1, k]
    pred = Xb @ W
    resid = ((Y - pred) ** 2).sum(axis=0)
    total = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    r2_per_dim = 1.0 - resid / np.clip(total, 1e-12, None)
    r2 = 1.0 - resid.sum() / np.clip(total.sum(), 1e-12, None)
    return {"W": W, "pred": pred, "r2": float(r2), "r2_per_dim": r2_per_dim}


def train_test_probe(X: np.ndarray, Y: np.ndarray, test_frac: float = 0.2, seed: int = 0,
                     ridge: float = 1e-6) -> dict:
    """Same, but fitted on one half of the data and scored on the other.

    With 64 activation dimensions and tens of thousands of points overfitting is not a real
    worry here, but reporting held-out R^2 is the habit worth having.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    test, train = idx[:n_test], idx[n_test:]
    fit = fit_linear_probe(X[train], Y[train], ridge=ridge)
    Xb = np.concatenate([X[test], np.ones((len(test), 1))], axis=1)
    pred = Xb @ fit["W"]
    resid = ((Y[test] - pred) ** 2).sum(axis=0)
    total = ((Y[test] - Y[test].mean(axis=0)) ** 2).sum(axis=0)
    return {
        "W": fit["W"],
        "r2_train": fit["r2"],
        "r2_test": float(1.0 - resid.sum() / np.clip(total.sum(), 1e-12, None)),
        "r2_test_per_dim": 1.0 - resid / np.clip(total, 1e-12, None),
        "test_idx": test,
        "pred_test": pred,
    }


def shuffle_control(X: np.ndarray, Y: np.ndarray, seed: int = 0, **kw) -> float:
    """Held-out R^2 after shuffling the labels. Should sit near zero."""
    rng = np.random.default_rng(seed)
    return train_test_probe(X, Y[rng.permutation(len(Y))], seed=seed, **kw)["r2_test"]


def token_history_features(tokens: np.ndarray, n_vocab: int, k: int = 6) -> np.ndarray:
    """The control that matters: features built from the raw input, with no network at all.

    For each position we one-hot encode the last `k` tokens (each at its own offset) plus the
    position index. A linear probe on these says how much of the belief state is trivially
    available from the input itself.

    Why this is the crucial control: for Mess3 the belief update is close to an exponentially
    decaying count of recent tokens, so a linear function of recent one-hot tokens already
    approximates it well. If this baseline scores as high as the model's residual stream, a high
    probe R^2 on the model tells you almost nothing about what the model computes. Always ask
    "compared to what?" before believing a probe.

    tokens: [B, L] -> features [B * L, k * n_vocab + L]
    """
    B, L = tokens.shape
    feats = np.zeros((B, L, k * n_vocab + L), dtype=np.float32)
    for offset in range(k):
        for t in range(L):
            src = t - offset
            if src >= 0:
                feats[np.arange(B), t, offset * n_vocab + tokens[:, src]] = 1.0
    for t in range(L):
        feats[:, t, k * n_vocab + t] = 1.0  # position one-hot
    return feats.reshape(B * L, -1)


def projection_2d(W: np.ndarray) -> np.ndarray:
    """Turn a probe that predicts a 3-state belief into a 2D plotting map.

    The probe gives activation -> belief; composing with the barycentric projection gives
    activation -> (x, y) directly, which is the 2D subspace of the residual stream in which the
    belief geometry lives. Shape [d + 1, 2].
    """
    from .msp import SIMPLEX_CORNERS

    return W @ SIMPLEX_CORNERS
