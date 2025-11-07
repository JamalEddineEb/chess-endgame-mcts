import chess
import numpy as np
import random
from collections import deque
from keras import layers, models
import json

from src.environment import RookKingEnv
from src.node import MCTSNode
from src.utils.utilities import *

class MCTSAgent():
    def __init__(self, state_size, c_puct=1000.0, n_simulations=100):
        self.state_size = state_size
        self.c_puct = c_puct
        self.memory = deque(maxlen=10000)
        self.n_simulations = n_simulations
        self.move_mapping = {}
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0.9998
        self.epsilon_min = 0.1
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.reverse_move_mapping = {}
        self.engine = chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish")

    def _build_model(self):
        # Input layer for an 8x8 chessboard with 3 channels
        input_layer = layers.Input(shape=(8, 8, 3))

        # Convolutional Block 1
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        # Convolutional Block 2
        x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)


        # Flatten the output from the convolutional layers
        x = layers.Flatten()(x)

        # Policy head for move selection 
        policy_output = layers.Dense(len(chess.SQUARES) * len(chess.SQUARES), activation='softmax', name='policy')(x)

        # Value head for estimating the value of the current board state
        value_output = layers.Dense(1, activation='tanh', name='value')(x)

        # Create the model with both policy and value heads
        model = models.Model(inputs=input_layer, outputs=[policy_output, value_output])
        model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'], metrics=['accuracy','accuracy'])

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())



    def get_state(self, board):
        # Create a more informative state representation
        state = np.zeros((8, 8, 3), dtype=np.float32)  # 3 channels: WK, BK, WR

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = 7 - chess.square_rank(square)
                file = chess.square_file(square)
                if piece.piece_type == chess.KING:
                    channel = 0 if piece.color == chess.WHITE else 1
                    state[rank, file, channel] = 1.0
                elif piece.piece_type == chess.ROOK and piece.color == chess.WHITE:
                    state[rank, file, 2] = 1.0

        return state
    

    def act(self, fen):
        root = MCTSNode(chess.Board(fen))
        self.simulate(root, self.n_simulations)

        # Use UCB1 to select the best move (explore-exploit tradeoff)
        best_move = root.best_child(self.c_puct).move

        return best_move

    def simulate(self, root, n_simulations, k_root=8, depth_limit=64):
        env = RookKingEnv()
        env.board = root.board.copy()

        # 1) Root: Gumbel-Top-k to select candidate first moves
        legal_root = list(env.get_legal_actions())
        if not legal_root:
            return  # terminal

        candidates, _, _ = gumbel_top_k_root_candidates(
            self.model, self.get_state, env, legal_root, k=k_root
        )

        # Build or retrieve child nodes for candidates (defer full expansion)
        for mv in candidates:
            if mv not in root.children:
                # create a placeholder child with prior set from policy later
                child_node = MCTSNode(board=None, parent=root, move=mv, prior=0.0)
                root.children[mv] = child_node

        # 2) Allocate simulations evenly among candidates (simple default)
        sims_per_cand = max(1, n_simulations // max(1, len(candidates)))

        # 3) Run simulations, one leaf expansion per simulation
        for mv in candidates:
            for _ in range(sims_per_cand):
                path = [root]
                env.board = root.board.copy()

                # Force the first move to be the candidate mv
                if mv in root.children:
                    child = root.children[mv]
                else:
                    # Should not happen, but guard
                    child = None

                # Step env with the candidate move to sync state
                next_state, reward, done = env.step(mv)

                if child is None or child.board is None:
                    # initialize child's board/state
                    if child is None:
                        child = MCTSNode(board=env.board.copy(), parent=root, move=mv, prior=0.0)
                        root.children[mv] = child
                    else:
                        child.board = env.board.copy()
                path.append(child)

                # Selection down the tree from the candidate child
                node = child
                depth = 1
                while (node.children and all(ch.visits > 0 for ch in node.children.values())
                    and not done and depth < depth_limit):
                    node = puct_select_child(node, self.c_puct)
                    # sync env with the node's move from its parent
                    next_state, reward, done = env.step(node.move)
                    if node.board is None:
                        node.board = env.board.copy()
                    path.append(node)
                    depth += 1

                # If terminal reached
                if done:
                    # terminal reward in root perspective
                    leaf_value = float(reward)
                    backup_path(path, leaf_value)
                    continue

                # If node has no children yet, expand once
                if not node.children:
                    leaf_value = expand_leaf(node, env, self.get_state, self.model)
                    backup_path(path, leaf_value)
                    continue

                # Otherwise, pick an unvisited child to expand
                unvisited = [ch for ch in node.children.values() if ch.visits == 0]
                if unvisited:
                    u = random.choice(unvisited)
                    # step env to u
                    next_state, reward, done = env.step(u.move)
                    if u.board is None:
                        u.board = env.board.copy()
                    path.append(u)
                    if done:
                        leaf_value = float(reward)
                    else:
                        leaf_value = expand_leaf(u, env, self.get_state, self.model)
                    backup_path(path, leaf_value)
                else:
                    # All visited: expand leaf anyway (rare here)
                    leaf_value = expand_leaf(node, env, self.get_state, self.model)
                    backup_path(path, leaf_value)


    def remember(self, state, action, reward, next_state,fen, done):
        self.memory.append((state, action, reward, next_state,fen, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            print(len(self.memory), "memory")
            return
        print("Replaying...")

        minibatch = random.sample(self.memory, batch_size)

        # Reshape states and next_states to match model input
        states = np.array([x[0] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])

        # Predict policy and value for current states
        current_policy, current_value = self.model.predict(states, verbose=0)
        future_policy, future_value = self.target_model.predict(next_states, verbose=0)

        # Update Q-values for actions taken
        for i, (state, action, reward, next_state, fen, done) in enumerate(minibatch):
            # Extract the action index
            move_idx = self.move_mapping[action.uci()]

            # Calculate target based on the policy output
            if done:
                target_value = reward
            else:
                # Get maximum value from future states
                max_future_value = future_value[i][0]
                target_value = reward + self.gamma * max_future_value

            # Update the value output for the action taken
            current_value[i] = target_value
            
            # Update the policy with a one-hot vector for the action taken
            target_policy = np.zeros_like(current_policy[i])  # Create zero vector for target policy
            target_policy[move_idx] = 1  # Set the action taken to 1
            current_policy[i] = target_policy  # Update current policy

        self.model.fit(states, [current_policy, current_value], epochs=5, verbose=1)



    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def save_move_mapping(self, filename="move_mapping.json"):
        with open(filename, 'w') as f:
            json.dump({k: str(v) for k, v in self.move_mapping.items()}, f)

    def load_move_mapping(self, filename="move_mapping.json"):
        with open(filename, 'r') as f:
            move_mapping_loaded = json.load(f)
            self.move_mapping = {k: int(v) for k, v in move_mapping_loaded.items()}
            self.reverse_move_mapping = {v: k for k, v in self.move_mapping.items()}
