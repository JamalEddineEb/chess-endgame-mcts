import random
import json
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

from src.mcts_agent import MCTSAgent
from src.environment import RookKingEnv
from src.chess_renderer import ChessRenderer

# --- setup ---
state_size = 8 * 8 * 3  # 8x8 board with 3 channels
agent = MCTSAgent(state_size, n_simulations=2)

model_file = "model_checkpoint.weights.h5"
print(f"Loading model from {model_file}")
agent.load(model_file)

env = RookKingEnv(stage=2)

# --- Qt app / UI ---
app = QApplication.instance() or QApplication(sys.argv)
renderer = ChessRenderer(env.board)
renderer.show()
renderer.update_board()  # initial paint

MOVE_DELAY_MS = 800  # visual pacing between moves

def play_step():
    if env.board.is_game_over():
        print(f"Game over: {env.board.result()}")
        return

    # Agent move via MCTS
    move, _, _ = agent.simulate(env)
    if move is None:
        print("No legal move; stopping.")
        return

    env.step_with_opponent(move)
    renderer.update_board()

    if env.board.is_game_over():
        print(f"Game over: {env.board.result()}")
        return

    QTimer.singleShot(MOVE_DELAY_MS, play_step)



# start loop after small delay so first frame shows
QTimer.singleShot(MOVE_DELAY_MS, play_step)

sys.exit(app.exec_())
