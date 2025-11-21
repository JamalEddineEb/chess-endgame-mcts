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
    batch_size = 10
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
        total_reward = 0.0
        moves_made = 0
        game_samples = []  # list of (state, improved_policy)

        print("episode ", e)

        max_moves = 50  # prevent endless shuffling

        while not env.done and moves_made < max_moves:
            state = env.get_state()

            # Run search at the root
            move, improved_policy, v_pi = agent.simulate(env)

            if move is None:
                # No legal moves
                break

            # Store (s, π′) for this position
            game_samples.append((state, improved_policy))

            # Play move in real environment vs Stockfish
            _, reward, done = env.step_with_opponent(move)
            total_reward += reward
            moves_made += 1

            print(f"Move {moves_made}: {move}, done={done}")
            print(env.board.unicode())

        # ----- game finished or max_moves reached -----
        # Use final game result as value target z
        result = env.board.result(claim_draw=True)  # "1-0","0-1","1/2-1/2","*"
        if result == "1-0":
            z = 1.0   # we beat Stockfish
        elif result == "0-1":
            z = -1.0  # we lost
        elif result == "1/2-1/2":
            z = 0.0   # draw / stalemate
        else:
            # game truncated by move limit: treat as failed win vs Stockfish
            z = -0.5

        # Push all (state, π′, z) into replay memory
        for s, pi in game_samples:
            agent.memory.append((s, pi, z))

        print(
            f"Episode: {e}/{episodes}, "
            f"Score (sum of rewards): {total_reward}, "
            f"moves: {moves_made}, "
            f"mates: {env.mates}/{e+1}, "
            f"final result: {result}, z={z}"
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