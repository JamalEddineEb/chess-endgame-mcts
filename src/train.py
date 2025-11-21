import os
import chess
import numpy as np
import random

from src.mcts_agent import MCTSAgent
from src.environment import RookKingEnv


def train_agent():
    env = RookKingEnv(stage=2,demo_mode=False)
    state_size = (8 , 8 , 3)  # 8x8 board with 3 channels
    agent = MCTSAgent(state_size)
    batch_size = 800
    episodes = 500
    target_update_frequency = 2
    checkpoint_frequency = 1

    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    model_file = "model_checkpoint.weights.h5"

    if os.path.exists(model_file):
        print(f"Loading model from {model_file}")
        agent.load(model_file)
    else:
        print("No model found, training a new one.")


    for e in range(episodes):
        env.reset()
        moves_made = 0
        game_samples = []  # list of (state, improved_policy, player_color)

        print("episode ", e)

        max_moves = 50  # prevent endless shuffling

        while not env.done and moves_made < max_moves:
            state = env.get_state()
            current_player = env.board.turn

            # Run search at the root
            move, improved_policy, v_pi = agent.simulate(env)

            if move is None:
                # No legal moves
                break

            # Store (s, π′, player) for this position
            game_samples.append((state, improved_policy, current_player))

            # Play move
            _, reward, done = env.step(move)
            moves_made += 1

            print(f"Move {moves_made}: {move}, done={done}")
            print(env.board.unicode())

        # ----- game finished or max_moves reached -----
        # Use final game result as value target z
        result = env.board.result(claim_draw=True)  # "1-0","0-1","1/2-1/2","*"
        
        z_white = 0.0
        if result == "1-0":
            z_white = 1.0
        elif result == "0-1":
            z_white = -1.0
        elif result == "1/2-1/2":
            z_white = 0.0
        else:
            # game truncated or unknown
            z_white = 0.0

        # Push all (state, π′, z) into replay memory
        for s, pi, player in game_samples:
            if player == chess.WHITE:
                z = z_white
            else:
                z = -z_white
            agent.memory.append((s, pi, z))

        print(
            f"Episode: {e}/{episodes}, "
            f"moves: {moves_made}, "
            f"mates: {env.mates}/{e+1}, "
            f"final result: {result}"
        )

        # --- TRAINING STEP -----------------------------------------
        if len(agent.memory) >= batch_size:
            agent.replay(batch_size)
            agent.save(model_file)

        # Update target network periodically (optional)
        if e % target_update_frequency == 0:
            agent.update_target_model()

            

    return agent

if __name__ == "__main__":
    agent = train_agent()