"""Hidden Markov processes, belief states, and optimal prediction.

The whole project rests on one idea. A *hidden Markov process* emits tokens you can see while
hopping between states you cannot. To predict the next token as well as possible, you must track
a probability distribution over which state the process is in right now, given everything you
have seen. That distribution is the **belief state**.

Two facts make this useful for interpretability:

1. The belief state is *the* optimal summary of the past. Any predictor that achieves the
   theoretical minimum loss must, in effect, compute it.
2. For a well-chosen process the set of reachable belief states has a striking shape (for Mess3,
   a fractal). So if a trained transformer tracks beliefs, that shape should be visible inside
   its activations. That is the claim of Shai et al. 2024, and what we reproduce here.

Notation used throughout:
  n_states  number of hidden states
  n_obs     number of distinct tokens (the vocabulary)
  T         tensor of shape [n_obs, n_states, n_states] with
            T[s, i, j] = P(next state = j AND emit token s | current state = i)
  belief    row vector of shape [n_states] summing to 1
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HMMProcess:
    """A hidden Markov process specified by its transition-emission tensor `T`."""

    T: np.ndarray  # [n_obs, n_states, n_states]
    name: str = "process"

    def __post_init__(self) -> None:
        self.T = np.asarray(self.T, dtype=np.float64)
        assert self.T.ndim == 3 and self.T.shape[1] == self.T.shape[2], "T must be [n_obs, n_states, n_states]"
        row_sums = self.T.sum(axis=(0, 2))  # for each current state: total probability
        assert np.allclose(row_sums, 1.0), f"rows of T must sum to 1, got {row_sums}"

    # ------------------------------------------------------------------ basics
    @property
    def n_obs(self) -> int:
        return self.T.shape[0]

    @property
    def n_states(self) -> int:
        return self.T.shape[1]

    @property
    def transition_matrix(self) -> np.ndarray:
        """P(next state | current state), ignoring which token was emitted. Shape [n_states, n_states]."""
        return self.T.sum(axis=0)

    def stationary_distribution(self) -> np.ndarray:
        """The long-run distribution over hidden states: the left eigenvector of the transition
        matrix with eigenvalue 1. This is our belief before seeing any token."""
        A = self.transition_matrix
        vals, vecs = np.linalg.eig(A.T)  # left eigenvectors of A == right eigenvectors of A.T
        i = int(np.argmin(np.abs(vals - 1.0)))
        v = np.real(vecs[:, i])
        return v / v.sum()

    # ----------------------------------------------------------------- beliefs
    def token_probs(self, belief: np.ndarray) -> np.ndarray:
        """P(next token | belief). Shape [n_obs].

        Summing T[s] over the next state gives the probability of emitting s from each current
        state; weighting by the belief gives the predictive distribution. This is exactly what an
        optimal next-token predictor outputs, so it is the target the transformer is trained on.
        """
        return np.einsum("i,sij->s", belief, self.T)

    def update_belief(self, belief: np.ndarray, obs: int) -> np.ndarray:
        """Bayes rule: fold one observed token into the belief.

        Unnormalised posterior over the *next* state is `belief @ T[obs]`; renormalising gives the
        new belief. Iterating this is the only thing a perfect predictor needs to do.
        """
        b = belief @ self.T[obs]
        s = b.sum()
        return b / s if s > 0 else np.full_like(b, 1.0 / len(b))

    def update_beliefs_batch(self, beliefs: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """Vectorised `update_belief` over a batch. beliefs [B, n_states], obs [B] -> [B, n_states]."""
        b = np.einsum("bi,bij->bj", beliefs, self.T[obs])
        return b / np.clip(b.sum(axis=1, keepdims=True), 1e-300, None)

    def beliefs_for_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """Ground-truth belief at every position of a batch of token sequences.

        tokens: [B, L]. Returns [B, L + 1, n_states] where
          beliefs[:, 0]     = stationary distribution (nothing seen yet)
          beliefs[:, t + 1] = belief after seeing tokens[:, :t + 1]

        Convention that matters later: a transformer's residual stream at position t has seen
        tokens 0..t, so it should encode beliefs[:, t + 1].
        """
        B, L = tokens.shape
        out = np.empty((B, L + 1, self.n_states))
        out[:, 0] = self.stationary_distribution()
        for t in range(L):
            out[:, t + 1] = self.update_beliefs_batch(out[:, t], tokens[:, t])
        return out

    # ---------------------------------------------------------------- sampling
    def sample(self, n_seq: int, seq_len: int, rng: np.random.Generator) -> np.ndarray:
        """Sample token sequences from the process. Returns [n_seq, seq_len] int array."""
        # Flatten (emit s, go to j) into one categorical draw per step, per sequence.
        flat = self.T.transpose(1, 0, 2).reshape(self.n_states, self.n_obs * self.n_states)
        states = rng.choice(self.n_states, size=n_seq, p=self.stationary_distribution())
        tokens = np.empty((n_seq, seq_len), dtype=np.int64)
        for t in range(seq_len):
            cum = np.cumsum(flat[states], axis=1)
            r = rng.random((n_seq, 1)) * cum[:, -1:]
            idx = (r > cum).sum(axis=1)
            tokens[:, t], states = np.divmod(idx, self.n_states)
        return tokens

    # ------------------------------------------------------------- information
    def myopic_entropy(self, max_len: int) -> np.ndarray:
        """Expected cross-entropy (in nats) of the *optimal* predictor at each position.

        Position 0 is predicted from the stationary belief and is hardest; later positions are
        easier because the belief has sharpened. The transformer's per-position loss should
        approach this curve, which is how we check it has really learned the process.
        """
        beliefs = np.array([self.stationary_distribution()])
        weights = np.array([1.0])
        out = np.empty(max_len)
        for t in range(max_len):
            p_tok = np.einsum("bi,sij->bs", beliefs, self.T)  # [n_beliefs, n_obs]
            with np.errstate(divide="ignore", invalid="ignore"):
                h = -np.nansum(np.where(p_tok > 0, p_tok * np.log(p_tok), 0.0), axis=1)
            out[t] = float(weights @ h)
            # expand the belief tree one step, weighting each branch by how likely it is
            new_b, new_w = [], []
            for b, w, row in zip(beliefs, weights, p_tok):
                for s in range(self.n_obs):
                    if row[s] > 1e-12:
                        new_b.append(self.update_belief(b, s))
                        new_w.append(w * row[s])
            beliefs, weights = np.array(new_b), np.array(new_w)
        return out


# --------------------------------------------------------------------- library
def mess3(x: float = 0.15, alpha: float = 0.6) -> HMMProcess:
    """The Mess3 process (Marzen & Crutchfield): 3 states, 3 tokens, belief states form a fractal.

    Mechanism: the state stays put with probability `1 - 2x` and jumps to each other state with
    probability `x`; the token emitted reveals the state it moved *into*, correctly with
    probability `alpha` and wrongly with probability `(1 - alpha) / 2` each.

    So every token is weak, noisy evidence about the current state. Beliefs never collapse to
    certainty, they keep getting nudged around the simplex, and the set of beliefs you can reach
    is self-similar: a fractal. That is what makes it such a good test image for interpretability.
    """
    beta = (1.0 - alpha) / 2.0
    stay = 1.0 - 2.0 * x
    T = np.zeros((3, 3, 3))
    for s in range(3):
        for i in range(3):
            for j in range(3):
                p_transition = stay if i == j else x
                p_emission = alpha if s == j else beta
                T[s, i, j] = p_transition * p_emission
    return HMMProcess(T, name=f"mess3(x={x},alpha={alpha})")


def z1r() -> HMMProcess:
    """Zero-One-Random: emits 0, then 1, then a coin flip, forever. 3 states, 2 tokens.

    A useful contrast to Mess3: here tokens are *decisive*, so after a couple of observations the
    belief is exactly one of three corners of the simplex. Discrete belief geometry, not a fractal.
    """
    T = np.zeros((2, 3, 3))
    T[0, 0, 1] = 1.0  # state 0 emits "0" and moves to state 1
    T[1, 1, 2] = 1.0  # state 1 emits "1" and moves to state 2
    T[0, 2, 0] = 0.5  # state 2 emits a random bit and returns to state 0
    T[1, 2, 0] = 0.5
    return HMMProcess(T, name="z1r")


def rrxor() -> HMMProcess:
    """Random-Random-XOR: emit two fair coin flips, then their XOR, forever. 5 states, 2 tokens.

    This is the important counterexample to Mess3. To predict the third token you must remember
    the first two *and combine them nonlinearly* (XOR). No weighted count of recent tokens can do
    that, so the "raw token history" control fails here while a real state-tracker succeeds.

    States: 0 = start of a block, 1 = saw first bit 0, 2 = saw first bit 1,
            3 = third token must be 0, 4 = third token must be 1.
    """
    S, A0, A1, F, T_ = 0, 1, 2, 3, 4
    T = np.zeros((2, 5, 5))
    T[0, S, A0] = 0.5          # first bit 0
    T[1, S, A1] = 0.5          # first bit 1
    T[0, A0, F] = 0.5          # 0 XOR 0 = 0
    T[1, A0, T_] = 0.5         # 0 XOR 1 = 1
    T[0, A1, T_] = 0.5         # 1 XOR 0 = 1
    T[1, A1, F] = 0.5          # 1 XOR 1 = 0
    T[0, F, S] = 1.0           # forced 0, block restarts
    T[1, T_, S] = 1.0          # forced 1, block restarts
    return HMMProcess(T, name="rrxor")


PROCESSES = {"mess3": mess3, "z1r": z1r, "rrxor": rrxor}
