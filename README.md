# Rook–King Endgame Agent: MCTS + Neural Network

A research-oriented project exploring whether a compact policy–value network can reduce Monte Carlo Tree Search (MCTS) simulations while reliably solving the rook vs. king (KRK) endgame under limited compute. The codebase is structured for reproducible experiments, ablations, and extensions to other elementary endgames.

## 1. Problem Statement

Given randomized legal KRK positions, the agent must produce a move sequence that checkmates within a move cap while minimizing mate length and simulation cost. KRK is chosen for its controlled branching factor, interpretable optimal strategies (edge and cornering), and unambiguous terminal criteria.

## 2. Method Overview

- Search: UCT-based MCTS with policy priors and value estimates from a neural network, replacing long rollouts to cut simulation budgets.
- Network: Lightweight policy–value model over a masked move space, trained from self-play (policy targets from visit counts; value targets from outcomes or mate-distance proxies).
- Integration: Policy priors guide selection; the value head evaluates leaf nodes; optional Dirichlet noise at the root for exploration; temperature scheduling for early training stability.

## 3. Repository Layout

```
.
├── train.py                       # Self-play, training, evaluation, checkpointing
├── environment.py                 # RookKingEnv: legality, terminal checks, randomized starts
├── mcts.py                        # MCTSAgent: selection/expansion/evaluation/backup
├── chess_renderer.py              # Board visualization (qualitative inspection)
├── move_mapping.json              # Stable move-to-index mapping for masked policy
├── dqn_model_checkpoint.weights.h5# Saved weights (created after training)
└── README.md                      # Project documentation
```

## 4. Installation

```bash
git clone https://github.com/JamalEddineEb/chess-endgame-mcts
cd chess-endgame-mcts
pip install -r requirements.txt
```

Requirements:
- Python 3.8+
- TensorFlow
- numpy
- memory-profiler
- Optional GPU (recommended)

## 5. Usage

Train from randomized KRK positions:

```bash
python train.py
```

Behavior:
- Checkpoints are saved to `dqn_model_checkpoint.weights.h5`.
- Training logs simulation counts, episode summaries, and mate statistics.
- If a checkpoint exists, training resumes by default.

## 6. Evaluation Protocol

Evaluate on a held-out set of randomized KRK positions at a fixed simulations-per-move budget. Report:
- Mate rate: percentage of positions solved within the move cap.
- Mean mate plies: average length for solved instances.
- Efficiency: mates per minute at the chosen simulation budget.


## 7. Design Details

- State encoding: board planes for pieces and side-to-move; legal-move masking restricts policy outputs to valid actions.
- Targets: policy from normalized visit counts; value from terminal outcome or a mate-distance proxy.
- Exploration: optional root Dirichlet noise; temperature control early in training.
- Determinism: set seeds for environment resets, network initialization, and sampling to support reproducibility.

## 8. Example Output (illustrative)

```
Num GPUs Available:  1
Loading model from dqn_model_checkpoint.weights.h5
episode 0
1 moves made
2 moves made
Episode: 0/500, Score: 1, epsilon: 0.95
mates : 0 / 1
saved
```

## 10. Limitations and Next Steps

Limitations:
- Endgame-only scope simplifies signals and may overstate generalization to midgame.
- Correct legality, check detection, and masking are critical; small bugs cause large variance.
- No tablebases used; incorporating them could assist curriculum or calibrate value targets.

Planned work:
- Deeper MCTS parameterization and improved rollouts.
- Extend to KQK and KRK with obstacles.
- Web-based evaluation UI with FEN input and PGN export.

## 11. Reproducibility

- Fix seeds for environment resets, network initialization, and MCTS sampling.
- Save model after each epoch; evaluate using frozen checkpoints to avoid leakage.
- Log configuration (simulation budget, network size, training steps) alongside results.