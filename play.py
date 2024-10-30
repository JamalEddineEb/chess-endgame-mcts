import chess
import random
import json
import numpy as np
import time
from mcts import MCTSAgent
from environment import RookKingEnv

state_size = 8 * 8 * 3  # 8x8 board with 3 channels
agent = MCTSAgent(state_size)
batch_size = 128
episodes = 50000
target_update_frequency = 10
checkpoint_frequency = 10
model_file = "dqn_model_checkpoint.weights.h5"
move_mapping_file = "move_mapping.json"
print(f"Loading model from {model_file}")
agent.load(model_file)
agent.load_move_mapping(move_mapping_file)
env = RookKingEnv(stage=2,demo_mode=True)
with open(move_mapping_file, "r") as file:
    move_mapping = json.load(file)
def random_move(board):
    """Select a random legal move from the board."""
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves) if legal_moves else None

def play_game(agent):
    """Play a game with the model vs a random opponent."""
    done = False
    step = 0
    while not done:
        time.sleep(1)  # Pause for visibility
        legal_moves = env.get_legal_actions()

        # Use the agent's act method to determine the action
        action = agent.act(env.get_fen())  # Assuming act takes state and legal moves
        print(f"Agent's move: {action}")

        next_state, reward, done = env.step(action)
        env.render_board()  # Show the board after the agent's move

        state = next_state  # Update the state
        step += 1

    if env.board.is_checkmate():
        print("Checkmate! The model wins!")
    elif env.board.is_stalemate():
        print("Stalemate! It's a draw.")
    else:
        print("Game over!")

play_game(agent)  # Use the agent to play

play_game(agent.model)  # Use the trained model to play
