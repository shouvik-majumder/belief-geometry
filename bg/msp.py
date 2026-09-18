"""The mixed-state presentation (MSP): the set of belief states the process can reach.

Start from the stationary belief, feed in every possible token sequence, and collect the beliefs
you land on. That cloud of points is the MSP. For Mess3 it is a fractal (a Sierpinski-like
structure inside the triangle of all 3-state distributions).

This module computes that ground-truth cloud. It needs no neural network at all: it is pure
probability theory, and it is the picture we will later go looking for inside a transformer.
"""
from __future__ import annotations

import numpy as np

from .process import HMMProcess

# Corners of the 2-simplex (all distributions over 3 states), drawn as an equilateral triangle.
SIMPLEX_CORNERS = np.array([[0.0, 1.0], [-np.sqrt(3) / 2, -0.5], [np.sqrt(3) / 2, -0.5]])


def to_simplex_xy(beliefs: np.ndarray) -> np.ndarray:
    """Map beliefs over 3 states to 2D plotting coordinates (barycentric projection).

    beliefs: [..., 3] -> [..., 2]. A belief of (1,0,0) lands on the top corner, and so on.
    """
    return np.asarray(beliefs) @ SIMPLEX_CORNERS


def plot_projection(beliefs: np.ndarray, basis: np.ndarray | None = None):
    """2D coordinates for plotting beliefs over any number of states.

    With 3 states we use the exact barycentric triangle. With more states the simplex does not fit
    on a page, so we fall back to PCA: the first two principal components of the belief cloud.
    Returns (xy, basis) so the same basis can be reused for a second cloud (essential when
    comparing ground truth with model activations).
    """
    b = np.asarray(beliefs, dtype=float)
    if b.shape[-1] == 3:
        return b @ SIMPLEX_CORNERS, None
    if basis is None:
        centred = b - b.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        basis = (b.mean(0), Vt[:2].T)
    mean, comps = basis
    return (b - mean) @ comps, basis


def msp_cloud(process: HMMProcess, depth: int, min_prob: float = 0.0) -> dict:
    """Enumerate belief states reachable within `depth` tokens.

    Returns a dict with:
      beliefs  [N, n_states]  the belief after each token sequence
      probs    [N]            how likely that sequence is
      depth    [N]            how many tokens were consumed to get there

    Branching is `n_obs ** depth`, so depth 10 on Mess3 is 59,049 points: plenty for a picture.
    `min_prob` prunes branches that are too unlikely to matter, which keeps deeper runs cheap.
    """
    b0 = process.stationary_distribution()
    frontier = [(b0, 1.0)]
    beliefs, probs, depths = [b0], [1.0], [0]
    for d in range(1, depth + 1):
        new_frontier = []
        for b, p in frontier:
            p_tok = process.token_probs(b)
            for s in range(process.n_obs):
                p_next = p * p_tok[s]
                if p_next <= min_prob:
                    continue
                b_next = process.update_belief(b, s)
                new_frontier.append((b_next, p_next))
                beliefs.append(b_next)
                probs.append(p_next)
                depths.append(d)
        frontier = new_frontier
        if not frontier:
            break
    return {
        "beliefs": np.array(beliefs),
        "probs": np.array(probs),
        "depth": np.array(depths),
    }


def unique_beliefs(beliefs: np.ndarray, decimals: int = 6) -> np.ndarray:
    """Collapse numerically identical beliefs (Z1R revisits the same few points endlessly)."""
    return np.unique(np.round(beliefs, decimals), axis=0)


def belief_rgb(beliefs: np.ndarray) -> np.ndarray:
    """Colour each belief by its own coordinates: state 0 -> red, 1 -> green, 2 -> blue.

    This is the trick that makes the comparison convincing later. If the transformer's activations
    carry the belief, colouring its points this way reproduces the same colour pattern as the
    ground-truth fractal, not just the same silhouette.
    """
    b = np.clip(np.asarray(beliefs, dtype=float), 0, 1)
    if b.shape[-1] == 3:
        return b
    # More than 3 states: group coordinates into three colour channels so the colouring still
    # carries real information about which states the belief favours.
    n = b.shape[-1]
    idx = np.array_split(np.arange(n), 3)
    out = np.stack([b[..., i].sum(-1) for i in idx], axis=-1)
    return out / np.clip(out.max(), 1e-12, None)
