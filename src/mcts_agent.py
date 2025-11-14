import chess
import numpy as np
import random
from collections import deque
from keras import layers, models
import json

from src.chess_renderer import ChessRenderer
from src.environment import RookKingEnv
from src.mcts_node import MCTSNode
from src.utils.utilities import *

class MCTSAgent():
    def __init__(self, state_size, c_puct=1000.0, n_simulations=50):
        self.state_size = state_size
        self.c_puct = c_puct
        self.memory = deque(maxlen=10000)
        self.n_simulations = n_simulations
        self.move_mapping = MoveMapping()
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0.9998
        self.epsilon_min = 0.1
        self.learning_rate = 0.001
        self.c_scale = 1
        self.c_visit = 50
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
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
    

    def act(self, env):
        best_move = self.simulate(env)

        return best_move.move
    
    def sequential_halving_phase(self, root, candidates, env, budget_per_candidate):
        """
        Run one phase of Sequential Halving for the given candidate moves.
        """
        # Run rollouts for each candidate
        for node in candidates:
            child = root.children[node.move]
            for _ in range(budget_per_candidate):
                self.rollout_from_candidate(root, child, env, node.move)

        scores = {}

        max_nb = max([a.visits for a in candidates])


        for child in candidates:
            scores[child] = (self.c_visit + max_nb)*self.c_scale*child.prior

        sorted_moves = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Keep top half
        n_keep = max(1, len(scores) // 2)
        return [node for node, _ in sorted_moves[:n_keep]]
    

    def rollout_from_candidate(self, root, child, env, candidate_move, depth_limit=64):
        path = [root]
        depth0 = len(env.board.move_stack)  # snapshot before rollout
        
        # Force candidate move
        env.step(candidate_move)
        path.append(child)
        node = child
        
        # Expand child if not already
        if not node.expanded:
            _ = node.expand_leaf(env, self.model)
        
        depth = 1
        # Selection with deterministic sequential policy
        while node.expanded and not env.done and depth < depth_limit:
            node = select_child_sequential_policy(node)  # <-- Equation 14
            env.step(node.move)
            path.append(node)
            depth += 1

        # Rewind exactly
        env.go_back(depth0)


    def simulate(self, env, k_root=16):
        """
        Root-level MCTS simulation with Gumbel-top-k and Sequential Halving.
        """
        root = MCTSNode()
        root.expand_leaf(env, self.model)

        candidates = gumbel_top_k_root_candidates(
            root, k=k_root
        )

        # Sequential Halving
        current_candidates = list(candidates)

        P = math.floor(math.log2(k_root))+1

        n_candidates = len(current_candidates)

        for phase_number in range(P):
            n_candidates = len(current_candidates)
            if n_candidates <= 1:
                break
            budget_per_candidate = max(1, self.n_simulations // (P*k_root*2**phase_number))
            current_candidates = self.sequential_halving_phase(root, current_candidates, env, budget_per_candidate)
            


        # Return best move among remaining candidates
        best_candidate = current_candidates[0]

        return best_candidate


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            print(len(self.memory), "memory")
            return
        print("Replaying...")

        minibatch = self.memory

        # Reshape states and next_states to match model input
        states = np.array([x[0] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])

        # Predict policy and value for current states
        current_policy, current_value = self.model.predict(states, verbose=0)
        _, future_value = self.target_model.predict(next_states, verbose=0)

        target_value = np.zeros_like(current_value) 

        # Update Q-values for actions taken
        for i, (_, action, reward, _, done) in enumerate(minibatch):
            # Extract the action index
            move_idx = self.move_mapping.get_index(action.uci())

            # Calculate target based on the policy output
            if done:
                target_value[i] = reward
            else:
                # Get maximum value from future states
                max_future_value = future_value[i][0]
                target_value[i] = reward + self.gamma * max_future_value

            # Update the value output for the action taken
            
            # Update the policy with a one-hot vector for the action taken
            target_policy = np.zeros_like(current_policy[i])  # Create zero vector for target policy
            target_policy[move_idx] = 1  # Set the action taken to 1
            current_policy[i] = target_policy  # Update current policy

        current_value = target_value

        self.model.fit(states, [current_policy, current_value], epochs=5, verbose=1)
        self.memory.clear()



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
