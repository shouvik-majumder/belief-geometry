"""Build the belief-geometry summary deck (data-club style: goal, method, finding).

  python scripts/05_make_deck.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from deckbuild import Deck  # noqa: E402

FIG = ROOT / "figures" / "deck"
RAW = ROOT / "figures"
OUT = ROOT / "deck"


def build() -> Deck:
    d = Deck()

    d.title_slide(
        "Do transformers build a Bayesian belief state?",
        ["Replication of Shai et al., Transformers Represent Belief State Geometry, NeurIPS 2024",
         "Tiny models trained from scratch: 201k parameters, 90 seconds each, one GPU",
         "Shouvik Majumder  |  data club, September 2026"],
    )

    d.slide(
        "The question",
        bullets=[
            "A hidden Markov process emits tokens you see while hopping between states you cannot.",
            "To predict the next token optimally you must track a posterior over the hidden state: the belief state.",
            "For a well chosen process the set of reachable beliefs has a striking shape, computable exactly in advance.",
            "So: train a transformer on that process, and look for the shape inside its activations.",
            "Why this is a good teaching problem: the ground truth is known, so the analysis cannot quietly lie to you.",
        ],
        notes="Contrast with the ESR project, where every measurement depended on a judge model.",
    )

    d.slide("A belief state is a point in a triangle", image=FIG / "bfig01_trajectory.png",
            bullets=["Three hidden states, so the belief is three numbers summing to one: a point in a 2-simplex.",
                     "Each observed token applies Bayes rule and moves the point."])

    d.slide("The set of all reachable beliefs", image=FIG / "bfig02_msp.png",
            bullets=["Mess3: every token is weak evidence, beliefs never collapse, and the reachable set is a fractal.",
                     "RRXOR: tokens are decisive, so only 36 distinct beliefs exist. Two very different target shapes."])

    d.slide("The models learn the processes exactly", image=FIG / "bfig03_training.png",
            bullets=["Cross-entropy reaches the information-theoretic floor to within 0.001 nats.",
                     "The per-position curve matches the Bayesian staircase, not just its average."])

    d.slide("The fractal is in the residual stream", image=RAW / "geometry_mess3_L4_d64_seed0_layer3_resid_post.png",
            bullets=["A linear map from 64 activation dimensions to the 3 belief coordinates, fitted on half the data and scored on the other half.",
                     "Held-out R-squared 0.98, and the colours match, not just the silhouette."],
            notes="This is the replication. The next slide is why it is not enough.")

    d.slide("...but a control with no network at all does better", image=FIG / "bfig04_controls.png",
            bullets=["Control: the same linear probe fitted to one-hot codes of the last six tokens.",
                     "For Mess3 it scores 0.988 against the model's 0.982, because the belief update is close to a decaying token count.",
                     "A high probe score is not evidence of computation unless a baseline with the same access does worse."])

    d.slide("RRXOR: a process where the shortcut fails",
            image=RAW / "geometry_rrxor_L4_d64_seed0_layer3_resid_post.png",
            bullets=["Two random bits, then their XOR. No weighted count of recent tokens can represent it.",
                     "The no-network control collapses to 0.13 while the model reaches 0.56."])

    d.slide("Where the belief state gets built", image=FIG / "bfig05_layers.png",
            bullets=["Mess3: flat across layers, already present at block 0. Consistent with copying from the input.",
                     "RRXOR: rises monotonically with depth. That is the signature of a quantity being computed."])

    d.slide(
        "Summary",
        bullets=[
            "Replicated: a 201k-parameter transformer trained only to predict tokens carries the Bayesian belief-state geometry linearly in its residual stream.",
            "Added: a no-network control showing that for Mess3 the headline result is largely free from the input, and the probe score is flat across layers.",
            "The contrast between Mess3 and RRXOR separates 'the geometry is recoverable' from 'the model computes it'.",
            "Next: does the probe improve with a nonlinear readout, when during training does the geometry appear, and does intervening on it change predictions as Bayes says it should?",
        ],
    )
    return d


if __name__ == "__main__":
    deck = build()
    path = deck.save(OUT / "belief_state_geometry.pptx")
    previews = deck.preview(OUT / "preview")
    print(f"deck  -> {path}")
    print(f"slides: {len(deck.spec)}   previews -> {previews[0].parent}")
