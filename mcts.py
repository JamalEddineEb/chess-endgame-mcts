import chess
import numpy as np
import random
import math
from collections import deque
from keras import layers, models
from keras.optimizers import Adam
from memory_profiler import profile
from environment import RookKingEnv
from node import MCTSNode
from tensorflow.keras.callbacks import EarlyStopping
import json

class MCTSAgent():
    def __init__(self, state_size, c_puct=1000.0, n_simulations=300):
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
        self.engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")

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

        # Policy head for move selection (outputs probabilities for each possible move)
        policy_output = layers.Dense(len(chess.SQUARES) * len(chess.SQUARES), activation='softmax', name='policy')(x)

        # Value head for estimating the value of the current board state
        value_output = layers.Dense(1, activation='tanh', name='value')(x)

        # Create the model with both policy and value heads
        model = models.Model(inputs=input_layer, outputs=[policy_output, value_output])
        model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'], metrics=['accuracy','accuracy'])

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def update_move_mapping(self, legal_moves):
        for move in legal_moves:
            move_uci = move.uci()
            if move_uci not in self.move_mapping:
                idx = len(self.move_mapping)
                self.move_mapping[move_uci] = idx
                self.reverse_mapping[idx] = move_uci

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
        root = MCTSNode(chess.Board(fen), agent=self)
        self.simulate(root, self.n_simulations)

        # Use UCB1 to select the best move (explore-exploit tradeoff)
        best_move = root.best_child(self.c_puct).move
        print(best_move)

        return best_move

    def simulate(self, root, n_simulations):
        node = root
        env = RookKingEnv()
        env.board = node.board.copy()  # Set env's state to node's board state
        policy = self.model.predict(np.expand_dims(self.get_state(env.board), axis=0), verbose=0)[0][0]

        # Selection and Expansion
        while node.is_fully_expanded():
            node = node.best_child(self.c_puct)
            env.board = node.board  # Keep environment state in sync

        # Expand node by adding children
        legal_moves = list(env.get_legal_actions())
        node.expand(legal_moves, policy)

        # Prepare for simulation
        states_to_predict = []

        for _ in range(n_simulations):
            done = False  # Reset done for each simulation
            depth = 0    # Reset depth for each simulation
            
            # Reset the environment state to the root node's board state
            env.board = node.board.copy()
            
            child = node.best_child(self.c_puct)

            while not done and depth<20:
                # Ensure the child represents a valid move
                best_move = child.move  
                
                # Take the move in the environment
                next_state, reward, done = env.step(best_move)
                self.remember(env.get_state(), best_move, reward, next_state, env.get_fen(), done)
                states_to_predict.append(self.get_state(env.board))  # Capture the board state for batch prediction

                # Update the child node based on the outcome
                child.visits += 1
                child.value += reward  # Update value based on the outcome

                depth += 1

        # Use batch prediction for values
        if states_to_predict:
            values = self.model.predict(np.array(states_to_predict), verbose=0)

            # Backpropagation: update each node's value based on the predicted values
            for i, value in enumerate(values):
                node.update(value[0][0])  # Update the current node's value
                
                # Optionally traverse back through the tree to update parent nodes
                parent = node.parent
                while parent is not None:
                    parent.update(value[0][0])  # Use the predicted value for backpropagation
                    parent = parent.parent




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
                max_future_value = np.max(future_value[i])
                target_value = reward + self.gamma * max_future_value

            # Update the value output for the action taken
            current_value[i] = target_value
            
            # Update the policy with a one-hot vector for the action taken
            target_policy = np.zeros_like(current_policy[i])  # Create zero vector for target policy
            target_policy[move_idx] = 1  # Set the action taken to 1
            current_policy[i] = target_policy  # Update current policy

        self.model.fit(states, [current_policy, current_value], epochs=10, verbose=1)



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
