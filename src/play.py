import random
import json
import numpy as np
import time
from src.mcts_agent import MCTSAgent
from src.environment import RookKingEnv
from src.chess_renderer import ChessRenderer

state_size = 8 * 8 * 3  # 8x8 board with 3 channels
agent = MCTSAgent(state_size,c_puct=0.2)
batch_size = 128
episodes = 50000
target_update_frequency = 10
checkpoint_frequency = 10
model_file = "model_checkpoint.weights.h5"
move_mapping_file = "move_mapping.json"
print(f"Loading model from {model_file}")
agent.load(model_file)
env = RookKingEnv(stage=2)
renderer = ChessRenderer(gui_mode=True)

def random_move(board):
    """Select a random legal move from the board."""
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves) if legal_moves else None

def play_game(agent):
    """Play a game with the model vs a random opponent."""
    renderer.render_board(env.board)
    done = False
    step = 0
    while not done:
        print(done," not done")
        time.sleep(1)  # Pause for visibility
        
        # Use the agent's act method to determine the action
        action = agent.act(env) 

        print(f"Agent's move: {action}")

        env.step(action)
        renderer.render_board(env.board)  # Show the board after the agent's move

        step += 1

    if env.board.is_checkmate():
        print("Checkmate! The model wins!")
    elif env.board.is_stalemate():
        print("Stalemate! It's a draw.")
    else:
        print("Game over!")

play_game(agent) 