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
agent = MCTSAgent(state_size, c_puct=0.2)

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
    # 1) Stop if terminal before agent moves
    if env.board.is_game_over():
        print(f"Game over: {env.board.result()}")
        return

    # 2) Agent move (consider offloading if slow)
    action = agent.act(env)
    if action is None:
        print("Agent returned no move; stopping.")
        return
    env.step(action)
    renderer.update_board()

    # 3) Check terminal after agent move
    if env.board.is_game_over():
        print(f"Game over: {env.board.result()}")
        return

    # 4) Delay a bit for readability, then opponent move
    def do_opponent():
        # Opponent move; handle failure gracefully
        try:
            env.oponent_step()  # must perform one legal move for the side to move
        except Exception as e:
            print("Opponent step failed:", e)
            return
        renderer.update_board()

        # 5) Check terminal after opponent move
        if env.board.is_game_over():
            print(f"Game over: {env.board.result()}")
            return

        # 6) Schedule next agent turn
        QTimer.singleShot(MOVE_DELAY_MS, play_step)

    QTimer.singleShot(MOVE_DELAY_MS, do_opponent)


# start loop after small delay so first frame shows
QTimer.singleShot(MOVE_DELAY_MS, play_step)

sys.exit(app.exec_())
