import os

from src.mcts_agent import MCTSAgent
from src.environment import RookKingEnv
from src.chess_renderer import ChessRenderer

def train_agent():
    env = RookKingEnv(stage=2,demo_mode=True)
    chess_renderer = ChessRenderer()
    state_size = 8 * 8 * 3  # 8x8 board with 3 channels
    agent = MCTSAgent(state_size)
    batch_size = 1024
    episodes = 500
    target_update_frequency = 2
    checkpoint_frequency = 1

    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    model_file = "dqn_model_checkpoint.weights.h5"
    move_mapping_file = "move_mapping.json"

    if os.path.exists(model_file):
        print(f"Loading model from {model_file}")
        agent.load(model_file)
        agent.load_move_mapping(move_mapping_file)
    else:
        agent.load_move_mapping(move_mapping_file)
        print("No model found, training a new one.")


    for e in range(episodes):
        chess_renderer.render_board(env.board)
        total_reward = 0
        moves_made = 0
        env.reset()
        print("episode ",e)

        while True:
            fen = env.get_fen()
            action = agent.act(fen)
            next_state, reward, done = env.step(action)
            fen = env.get_fen()

            # agent.remember(state, action, reward, next_state,fen, done)
            total_reward += reward
            moves_made += 1
            print(moves_made,"moves made\n\n\n")

            # env.render_board()

            if done or moves_made > 5:  # Prevent infinite games
                print(f"Episode: {e}/{episodes}, Score: {total_reward}, epsilon: {agent.epsilon}")
                break
        print("mates : ",env.mates,"/",e+1)

        if e % checkpoint_frequency == 0:
            agent.replay(batch_size)
            agent.save(model_file)
            print("saved")
            agent.save_move_mapping()
        # env.render_board()

        # Update target network periodically
        if e % target_update_frequency == 0:
            agent.update_target_model()
            

    return agent

if __name__ == "__main__":
    agent = train_agent()