# Chess Endgame MCTS (KRK) - Research Documentation

## Overview

This project investigates Monte Carlo Tree Search (MCTS) guided by a neural policy-value network for the King-and-Rook vs King (KRK) endgame. The work focuses on efficient search primitives, legal-action masking in policy networks, and root-level exploration strategies. **Status: ongoing; currently integrating Gumbel-Top-k sampling at the root and optimizing memory efficiency.**

## Motivation

KRK is a minimal yet non-trivial testbed for researching MCTS + RL:

- **Compact domain**: finite, deterministic, ~1000 reachable positions; allows full exploration without prohibitive cost.
- **Realistic constraints**: typical researcher setup (limited GPU, single machine); experiments remain reproducible and fast.
- **Methodological clarity**: small network and short horizons reveal PUCT dynamics, backup stability, and policy prior quality without confounding factors.
- **Algorithm validation**: prototype and benchmark search improvements (masked softmax, Gumbel-Top-k) before scaling to general chess or other games.

## Method

### Environment

- **Game**: KRK endgame on python-chess; terminal rewards: +1 for White mate, -1 for Black mate, 0 for draw (50-move rule or repetition).
- **State representation**: 8×8 board as multi-channel tensor (piece occupancy, turn, move count).
- **Depth cap**: maximum ~100 plies to avoid infinite loops in rare cases; terminal states checked on every step.

### Neural Network

**Architecture**: lightweight encoder (CNN or ResNet stem) + two heads.

- **Policy head**: Dense layer outputting logits over a fixed global action space (no softmax activation).
  - Action space: 64 × 64 from-square and to-square pairs (4096 actions total).
  - Rationale: fixed size matches the board; promotions disabled for KRK.
  - **Key insight**: by outputting raw logits instead of probabilities, we avoid wasting softmax mass on illegal moves and gain numerical stability for downstream masked softmax and Gumbel operations.

- **Value head**: Dense layer + tanh activation predicting expected outcome ∈ [-1, 1].

### Action Mapping & Legal-Action Masking

- **Deterministic global mapping**: uci → index via a precomputed JSON (move_uci_string ↔ action_index).
- **Legal indexing**: at each node, env.get_legal_actions() returns chess.Move objects; we convert to indices via the mapping and gather policy logits at those indices only.
- **Masked softmax**: priors over legal moves are computed as:
  - \(z = \text{logits}[\text{legal\_indices}] - \max(\text{logits}[\text{legal\_indices}])\)
  - \(p = \text{softmax}(z)\)
  - This ensures \(\sum_a p(a) = 1\) only over legal moves, concentrating all probability mass appropriately.

### Search: PUCT + Backup

**Standard MCTS loop**:
1. **Selection**: traverse tree from root using PUCT until reaching a leaf node.
   - \(\text{PUCT}(a) = Q(a) + c \cdot P(a) \cdot \sqrt{\frac{N}{1 + N(a)}}\)
   - \(Q(a) = V\_\text{sum}(a) / N(a)\) (average backed-up values).
   - \(P(a)\) = prior from masked softmax over legal actions.

2. **Expansion**: sample one legal move uniformly at random, evaluate with the network, and add a new child node.

3. **Backup**: propagate value \(v\) back up the path, flipping sign at each ply (alternating maximizer/minimizer).
   - \(V\_\text{sum}(a) \mathrel{+}= v\) and \(N(a) \mathrel{+}= 1\) at each visited node.

**Stability measures**:
- Value clamping: clip network output to [-0.99, 0.99] to avoid numerical issues.
- Stable averaging: Q(a) computed as V_sum / visits with explicit division guards.

### Root-Level Exploration: Gumbel-Top-k (In Progress)

**Goal**: select a small, diverse set of high-probability root actions, then allocate simulations across them.

**Algorithm** (Gumbel-Max trick):
- For each legal action at the root, draw i.i.d. Gumbel noise \(g_i \sim \text{Gumbel}(0, 1)\).
- Compute perturbed scores: \(s_i = \text{logits}[i] + g_i\).
- Select top-k actions by argmax over perturbed scores; these are distributed according to softmax(logits) without replacement.

**Why it helps**:
- Prevents early bias toward a single greedy action.
- Ensures exploration of promising alternatives proportional to policy confidence.
- Avoids pre-normalizing logits; operates directly on raw scores for numerical stability.
- Enables future variants (e.g., top-k-then-visit-proportional allocation).

**Current status**: helper functions written; integration into the main search loop in progress.

## Training

### Data Generation

- Self-play: agent plays against itself using MCTS (e.g., 100–500 simulations per move).
- Replay buffer: store (state, policy_target, value_target, outcome) tuples.
- Policy targets: visit distribution at root or high-value nodes, masked to legal actions.
- Value targets: actual game outcome (discounted if needed, or raw terminal reward).

### Loss & Optimization

- **Policy loss**: cross-entropy between network logits and target distribution (computed over legal actions only).
- **Value loss**: MSE between predicted and target value.
- **Combined loss**: \(L = L_\text{policy} + \lambda L_\text{value}\) with optional regularization.

### Evaluation

- **Benchmark**: win rate vs. baseline (random or simple heuristic).
- **Endgame solving**: spot-check a few key positions against tablebases (e.g., via python-chess).
- **Ablation metrics**: depth visited, terminal nodes reached, average backup value magnitude.

## Implementation Status

### Completed

-  Fixed global action mapping (4096 actions, uci ↔ index JSON).
-  Legal-action indexing and batch gathering from logits.
-  Masked softmax over legal logits.
-  Model with logits-only policy head and value head.
-  PUCT node selection and single-leaf expansion.
-  Value backup with sign alternation and stability guards.
-  Environment with legal move generation, terminal checks, step function.

### In Progress

-  **Gumbel-Top-k root selection**: implementing noise sampling, perturbed score ranking, and candidate aggregation.
-  **Memory optimization**: profiling RAM usage during large rollouts; exploring state-dict checkpointing or compressed node storage.
-  **Simulation throughput**: benchmarking simulation rate on limited hardware; investigating bottlenecks (env.step, network inference, masking).

### Planned

-  Full training loop: self-play, replay buffer management, mini-batch training, periodic checkpoint and evaluation.
-  Ablation studies:
  - Impact of Gumbel-Top-k vs. greedy root selection.
  - Effect of c_puct, simulation budget, and depth limits.
  - Policy prior quality: masked softmax vs. uniform baseline.
-  Comparison with tablebases and simple heuristics (e.g., distance-to-mate).
-  Scaling experiments: vary network size, action space (if moving to general chess), and simulation count.


## Quick Start

### Install

```bash
git clone https://github.com/JamalEddineEb/chess-endgame-mcts.git
cd chess-endgame-mcts
pip install -r requirements.txt
```

### Train

```bash
python -m src.train
```

### Play demonstration

```bash
python -m src.play
```

## Key Design Decisions

1. **Logits over probabilities**: avoid global softmax, enable efficient masking and Gumbel operations.
2. **Fixed global action space**: simplifies indexing, deterministic across runs, compatible with fixed network head.
3. **Legal-action masking at nodes**: ensure policy mass is only on reachable actions; simplify PUCT and Gumbel computation.
4. **Single-leaf expansion**: each simulation adds one new node; reduces memory overhead and allows stable value backup.
5. **Gumbel-Top-k at root**: diverse candidate selection; proportional to policy confidence; prepares for future allocation strategies.

## Known Limitations & Future Work

- **Small domain**: KRK is solvable; the goal is methodological clarity, not solving from scratch.
- **Hardware constraints**: currently runs on CPU + modest GPU; future work may explore distributed search or batch inference.
- **No transposition handling**: KRK has many transpositions; storing only unique nodes could improve efficiency (future).
- **Sparse rewards**: terminal rewards only; intermediate value signal may be weak; consider auxiliary targets or reward shaping in future iterations.

## References & Inspiration

- AlphaZero (Silver et al., 2017): neural MCTS policy priors and self-play training.
- Gumbel-Max trick (Gumbel, 1954; Lakshminarayanan et al., 2017): for categorical sampling without replacement.
- Mu Zero (Schaal et al., 2019): environment model + value network; relevant for understanding backup stability.
- python-chess: robust game logic and UCI parsing.

