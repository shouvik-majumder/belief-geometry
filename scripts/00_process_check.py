"""Step 0: check the process itself, before any neural network is involved.

Nothing here needs a GPU. The point is to convince yourself the probability machinery is right,
so that later, when the transformer's activations do or do not match, you know the fault is not
in the maths.

  python scripts/00_process_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bg.msp import msp_cloud, unique_beliefs  # noqa: E402
from bg.process import mess3, z1r  # noqa: E402


def check(process, seq_len: int = 10, n_seq: int = 20_000, seed: int = 0) -> None:
    print(f"\n=== {process.name} ===")
    print(f"states={process.n_states}  tokens={process.n_obs}")

    # 1. The tensor is a valid joint distribution: from any state, all (token, next state)
    #    outcomes together have probability 1.
    print(f"row sums of T (should all be 1): {process.T.sum(axis=(0, 2))}")

    # 2. The stationary distribution really is stationary.
    pi = process.stationary_distribution()
    print(f"stationary distribution: {np.round(pi, 4)}  (pi @ A - pi = {np.abs(pi @ process.transition_matrix - pi).max():.2e})")

    # 3. Sampling matches theory: empirical token frequencies should match the stationary
    #    predictive distribution, and empirical belief-weighted predictions should be calibrated.
    rng = np.random.default_rng(seed)
    tokens = process.sample(n_seq, seq_len, rng)
    emp = np.bincount(tokens.ravel(), minlength=process.n_obs) / tokens.size
    print(f"token frequencies: empirical {np.round(emp, 4)}  vs theory {np.round(process.token_probs(pi), 4)}")

    # 4. Beliefs are valid distributions, and predicting with them beats predicting with the prior.
    beliefs = process.beliefs_for_tokens(tokens)
    assert np.allclose(beliefs.sum(-1), 1.0), "beliefs must be distributions"
    p_next = np.einsum("bti,sij->bts", beliefs[:, :-1], process.T)  # predicted token dists
    nll_belief = -np.log(np.clip(np.take_along_axis(p_next, tokens[..., None], axis=2)[..., 0], 1e-300, None)).mean()
    p_prior = process.token_probs(pi)
    nll_prior = -np.log(np.clip(p_prior[tokens], 1e-300, None)).mean()
    print(f"cross-entropy (nats): belief-tracking {nll_belief:.4f}  vs ignoring history {nll_prior:.4f}")
    print("  -> the gap is exactly what the transformer has to learn to capture")

    # 5. The myopic entropy curve: how hard each position is for a perfect predictor.
    me = process.myopic_entropy(seq_len)
    print(f"optimal per-position loss: {np.round(me, 4)}")

    # 6. How many distinct belief states exist? Fractal vs finite is visible already here.
    cloud = msp_cloud(process, depth=min(8, seq_len))
    uniq = unique_beliefs(cloud["beliefs"])
    print(f"belief states reachable within 8 tokens: {len(cloud['beliefs'])} visits, {len(uniq)} distinct")
    print("  -> Mess3 keeps generating new ones (fractal); Z1R saturates at a handful (discrete)")


if __name__ == "__main__":
    check(mess3())
    check(z1r())
    print("\nprocess check OK")
