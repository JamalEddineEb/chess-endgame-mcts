# 🌟 Project Brief: Gumbel AlphaZero for KRK Endgame

## 💡 1. Project Overview

This project implements a sophisticated **Gumbel AlphaZero (GAZ)** agent to master the **King and Rook vs. King (KRK)** chess endgame. The agent combines a deep convolutional neural network with a state-of-the-art tree search algorithm, demonstrating superior sample efficiency compared to traditional AlphaZero and MuZero.

### Key Features
* **Gumbel-Max Trick:** Uses Gumbel noise sampling for action selection, enabling effective exploration with a minimal number of simulations ($N$).
* **Sequential Halving:** Implements a highly efficient search budget allocation strategy for MCTS. 
* **Two-Headed Neural Network:** A modern architecture with separate heads for **Policy** (move selection) and **Value** (board evaluation).
* **Strong Opponent Baseline:** Trains against the powerful **Stockfish** engine to guarantee high-quality self-play data.

---

## 🧠 2. Algorithmic Detail: The Gumbel Advantage

The agent's strength lies in its ability to achieve high-quality policy improvement with a minimal simulation budget. This is achieved by focusing computational effort only on the most promising moves.

### Advanced Search Strategy: Sequential Halving
Instead of deep, uniform search, the agent uses a **Sequential Halving** process over the **Gumbel Top-$k$** candidates selected at the root. This progressively eliminates low-value actions, concentrating simulations where they matter most.

### Policy Target Construction (The High-Quality Label)
The network is trained to match an "improved policy" ($\pi'$) derived directly from the search results, where the original network logits ($\mathbf{L}$) are corrected by the search's completed Q-values ($\mathbf{Q}$):

$$\pi' \propto \text{softmax}(\mathbf{L} + \sigma(\mathbf{Q}))$$

* **$\mathbf{Q}$ (Completed Q-Values):** Uses the search-improved value ($V/N$) for visited actions, and the network's value ($V_\pi$) as a reliable fallback for unvisited actions.
* **Impact:** This high-quality, aggregated label allows the agent to learn the optimal policy with drastically fewer training episodes.

---

## 📈 3. Key Performance and Technical Architecture

### Architecture Diagram
The codebase is organized into modular files, reflecting best practices for Deep Reinforcement Learning frameworks.



* **`mcts_agent.py`**: The core intelligence, containing the **MCTS algorithm**, the two-headed **Keras/TensorFlow model definition**, and the $\pi'$ label computation logic.
* **`environment.py`**: The game environment, managing the board state, rule checking, and running the **Stockfish** opponent.
* **`train.py`**: The primary self-play loop for data generation, collecting the training samples $(\mathbf{s}, \mathbf{\pi'}, \mathbf{z})$.

### Performance Indicators (KPIs)
The agent's success is defined by rapid convergence to a super-human level on the KRK task.

* **Win Rate:** Should quickly exceed 95% against Stockfish.
* **Average Game Length:** The number of moves to mate should decrease significantly as the policy learns optimal forcing sequences.



---

## 🚀 4. Getting Started

### Prerequisites
* Python 3.8+
* `tensorflow`, `keras`, `python-chess`, `PyQt5`
* **Stockfish** executable must be installed and correctly linked in `environment.py`.

### Execution
| Command | Purpose |
| :--- | :--- |
| `python -m src.train` | Starts the self-play loop and trains the model, saving weights to `model_checkpoint.weights.h5`. |
| `python -m src.play` | Runs the trained agent in a real-time demo, rendering the moves via PyQt5/SVG. |

---
