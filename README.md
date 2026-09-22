# Belief-state geometry in tiny transformers

Do transformers trained only to predict the next token build the *Bayesian posterior over hidden
states* internally, and does that posterior's geometry show up in the residual stream?

Replication and extension of Shai et al., *Transformers Represent Belief State Geometry in their
Residual Stream*, NeurIPS 2024 ([arXiv:2405.15943](https://arxiv.org/abs/2405.15943)).

Everything here is free and local: models are trained from scratch in about 90 seconds on one
RTX 3090, there are no downloads, no gated weights and no API calls.

## Setup

```powershell
conda activate beliefgeom
cd D:\dev\belief-geometry
python scripts/00_process_check.py
```

The environment was created with:

```powershell
conda create -n beliefgeom python=3.12 -y
conda activate beliefgeom
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformer_lens numpy scipy matplotlib pandas tqdm scikit-learn einops jupyter
```

## The four steps

| Script | What it does | Cost |
|---|---|---|
| `00_process_check.py` | Verifies the probability machinery before any network exists | seconds, CPU |
| `01_plot_msp.py` | Draws the ground-truth belief geometry (the fractal) | seconds, CPU |
| `02_train.py` | Trains a 4-layer, 64-dim transformer from scratch on the process | ~90 s, GPU |
| `03_belief_geometry.py` | Probes the residual stream for the belief state, with controls | ~1 min, GPU |

```powershell
python scripts/01_plot_msp.py --depth 11                 # Mess3 fractal
python scripts/02_train.py --n-steps 4000                # train on Mess3
python scripts/03_belief_geometry.py --ckpt data/checkpoints/mess3_L4_d64_seed0.pt

python scripts/02_train.py --process rrxor --seq-len 12 --n-steps 4000
python scripts/03_belief_geometry.py --ckpt data/checkpoints/rrxor_L4_d64_seed0.pt
```

## What we found

**Both models learn the process exactly.** Cross-entropy sits on the information-theoretic floor
(excess under 0.001 nats), and the per-position loss follows the Bayesian "myopic entropy"
staircase rather than merely matching its average.

**The fractal is there.** Projecting the Mess3 model's residual stream through a linear probe
reproduces the ground-truth belief geometry, coloured correctly, at held-out R² = 0.98.

**But the headline number is mostly an illusion, and that is the real lesson.**

Held-out R^2, mean over three independently trained seeds (range in brackets). Each seed also gets
its own token sample and probe split. Recomputed by `04_summary_figures.py --force`.

| Process | Model, last layer | Model, all 4 layers concatenated | Untrained model | Raw token history, no network | Layer sweep |
|---|---|---|---|---|---|
| Mess3 | 0.980 [0.976-0.983] | 0.992 [0.990-0.993] | 0.895 [0.881-0.905] | **0.988** [0.988-0.988] | flat: 0.98 at every layer |
| RRXOR | 0.547 [0.523-0.563] | **0.712** [0.685-0.735] | 0.184 [0.177-0.189] | 0.132 [0.128-0.135] | rising: 0.32 -> 0.42 -> 0.50 -> 0.55 |

For **Mess3** a linear function of the last few one-hot tokens predicts the belief state about as
well as the trained network does: slightly better than any single layer, slightly worse than all
layers stacked together. And the probe scores the same at layer 0 as at layer 3. The belief update
for Mess3 is close to an exponentially decaying token count, so the geometry is nearly free from
the input. The picture is real, but it is weak evidence that the network computes anything.
(An earlier version of this README said the raw history "beats" the model, from one seed and the
last layer only. "Matches" is what three seeds and the concatenated probe support.)

For **RRXOR**, where the third token is the XOR of the previous two, no weighted token count can
work. The no-network baseline collapses to 0.13, the model reaches 0.55 at the last layer, and the
probe improves monotonically with depth. That rising profile is the signature of a quantity being
*built* layer by layer, which is what "the model represents the belief state" should mean.

**RRXOR's belief state is spread across layers.** Probing all four residual streams at once lifts
R^2 from 0.55 to 0.71, in every seed. So the last-layer 0.55 was not a ceiling: part of the belief
is carried in earlier layers and not copied forward. Shai et al. report a similar layer-distributed
representation for RRXOR and probe concatenated layers for it. Mess3 gains almost nothing from concatenation (0.98 -> 0.99),
consistent with its geometry being available from the input at every depth.

Takeaway worth carrying to every future probing experiment: **a probe result means nothing without
a baseline that shares the probe's access to the input.** Ask "compared to what?" before believing
any R².

## Layout

```
bg/process.py   HMM processes (Mess3, RRXOR, Z1R), Bayes updates, myopic entropy
bg/msp.py       mixed-state presentation: the reachable belief cloud, simplex/PCA projection
bg/model.py     tiny HookedTransformer, training loop, activation capture
bg/probe.py     linear probes, shuffled-label and raw-token-history controls
scripts/        the four steps above
figures/        output plots
data/checkpoints/  trained models and loss histories
```

## Where to take it

Ordered roughly by effort. Items 1 to 3 are self-contained experiments; 4 to 6 are open enough to
become a workshop paper.

1. **What is the remaining RRXOR variance?** Concatenating layers takes R² from 0.55 to 0.71.
   Is the rest non-linearly encoded, or absent? Try a two-layer MLP probe on the concatenated
   streams: a big gap means non-linear encoding, no gap means the information is not there.
2. **When does the geometry appear during training?** Save checkpoints every 100 steps and plot
   probe R² against step next to the loss curve. Does the geometry form before, during, or after
   the loss drops?
3. **Where does it live?** Sweep `--hook resid_pre/resid_mid/resid_post` and every layer; ablate
   individual attention heads and watch which ones the probe R² depends on. This is your ESR
   ablation methodology in a setting with ground truth.
4. **Causal test.** Add the belief direction to the residual stream at run time (the steering hook
   you already wrote for Gemma) and check whether the output distribution moves the way Bayes says
   it should. That turns a correlation into a mechanism.
5. **Harder processes.** Longer memory, more states, non-ergodic; see *Constrained Belief Updates
   Explain Geometric Structures in Transformer Representations*
   ([arXiv:2502.01954](https://arxiv.org/abs/2502.01954)) for what is already known.
6. **The control paper.** The Mess3-versus-RRXOR contrast above is a small but genuine
   methodological result. Quantifying "how much belief geometry is free from the input" across a
   family of processes would be a useful short paper on its own.
