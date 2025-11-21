import chess
import numpy as np
import json
from collections import deque
from keras import layers, models
import math
from src.mcts_node import MCTSNode
from src.utils.utilities import *
from src.utils.move_mapping import MoveMapping


class MCTSAgent():
    def __init__(self, state_size, n_simulations=20):
        self.state_size = state_size
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
        best_move, _, _ = self.simulate(env)
        return best_move

    
    def sequential_halving_phase(self, root, candidates, env, budget_per_candidate):
        """
        One phase of Sequential Halving on the current candidate children.

        For each candidate child:
        - Run 'budget_per_candidate' rollouts starting from that child.
        - Use mean Q = value / visits as its score.
        - Keep the top half of candidates by score.
        """
        # Rollouts
        for child in candidates:
            for _ in range(budget_per_candidate):
                self.rollout_from_candidate(root, child, env)

        # Compute scores using completed Q-values
        scores = {}
        max_visits = max(c.visits for c in candidates) if candidates else 0

        for child in candidates:
            if child.visits > 0:
                # Value is from child's perspective (opponent), so negate for root
                q = -child.value / child.visits
            else:
                q = 0.0  # if never visited, treat as neutral
            # Eq. (8) style scaling: (c_visit + maxN) * c_scale * q
            scores[child] = (self.c_visit + max_visits) * self.c_scale * q

        # Keep top half
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        n_keep = max(1, len(sorted_items) // 2)

        return [child for child, _ in sorted_items[:n_keep]]

    

    def rollout_from_candidate(self, root, child, env, depth_limit=64):
        """
        One rollout:
        - Play 'child.move' (1 ply).
        - Follow deterministic non-root policy until leaf / terminal / depth_limit.
        - Backup leaf value along the path (negating at each step).
        """
        path = [root, child]

        # Snapshot to restore tree state exactly after rollout
        baseline = len(env.board.move_stack)

        if child.move not in env.board.legal_moves:
            return

        # 1) Apply the candidate move (1 ply)
        _, reward, done = env.step(child.move)

        node = child
        depth = 1
        
        leaf_value = 0.0

        # If move caused termination (e.g. mate), reward is for the player who moved (child)
        if done:
             leaf_value = reward
             backup_path(path, leaf_value)
             env.go_back(baseline)
             return

        # Expand child once if needed
        if not node.expanded:
            # expand_leaf returns value for the current player (who is about to move from node)
            # i.e. the player at 'node'.
            leaf_value = node.expand_leaf(env, self.model)
            backup_path(path, leaf_value)
            env.go_back(baseline)
            return

        # 2) Selection / expansion loop
        while not env.done and depth < depth_limit:
            # If node is not expanded, expand as leaf and stop
            if not node.expanded or not node.children:
                leaf_value = node.expand_leaf(env, self.model)
                backup_path(path, leaf_value)
                env.go_back(baseline)
                return

            # Choose next child using deterministic non-root policy (Eq. 14 style)
            next_child = select_child_sequential_policy(node)
            path.append(next_child)

            if next_child.move not in env.board.legal_moves:
                # Optional debug:
                # print("Skip illegal next_child in rollout:", next_child.move)
                break

            # Play that move
            _, reward, done = env.step(next_child.move)
            node = next_child
            depth += 1
            
            if done:
                leaf_value = reward
                break

        # 3) Terminal handling or depth limit
        if not done:
             # Depth limit hit but game not over: evaluate with value head
            leaf_value = node.expand_leaf(env, self.model)

        backup_path(path, leaf_value)

        # 4) Rewind environment to original root state
        env.go_back(baseline)


    def compute_policy_improvement(self, root, state):

        # Step 1: get raw logits + value head vπ
        # --- FIX: convert policy probabilities -> pseudo-logits ---

        policy_probs, v_pi = self.model.predict(state[None], verbose=0)
        policy_probs = policy_probs[0]     # shape = (4096,)
        v_pi = float(v_pi[0])

        # numerical safety
        eps = 1e-8
        policy_probs = np.clip(policy_probs, eps, 1.0)

        # convert probabilities to log-probabilities (pseudo-logits)
        logits = np.log(policy_probs)


        move_mapping = self.move_mapping

        # Step 2: Compute Q-values for visited actions, fallback to vπ for unvisited.
        Q = {}
        visits = []
        for move, child in root.children.items():
            if child.visits > 0:
                # Negate for root perspective
                q_val = -child.value / child.visits
            else:
                q_val = v_pi
            Q[move] = q_val
            visits.append(child.visits)

        max_visits = max(visits) if len(visits) > 0 else 0

        # Step 3: Build completedQ vector in policy_logits shape
        completedQ = np.zeros_like(logits, dtype=np.float32)
        for move, q in Q.items():
            idx = move_mapping.get_index(move.uci())
            completedQ[idx] = q

        # Step 4: σ(q) from Eq. (8) in paper
        sigma = (self.c_visit + max_visits) * self.c_scale * completedQ

        # Step 5: Improved policy π' = softmax(logits + σ(q))
        improved_logits = logits + sigma
        # Mask illegal moves before softmax
        legal_mask = np.full_like(improved_logits, -1e9)
        for move in root.children.keys():
            idx = move_mapping.get_index(move.uci())
            legal_mask[idx] = 0

        improved_logits = improved_logits + legal_mask



        improved_policy = np.exp(improved_logits - np.max(improved_logits))
        improved_policy /= np.sum(improved_policy)

        return improved_policy, v_pi

    def simulate(self, env, k_root=4):
        """"
        Run Gumbel-style root search and return:
        - best_move: chess.Move
        - improved_policy: np.array shape (4096,) over global action space
        - v_pi: scalar value prediction at root
        """
        root = MCTSNode()
        root_value = root.expand_leaf(env, self.model)

        if not root.children:
            # No legal moves
            return None, None, root_value

        # Gumbel-Top-k root candidates
        k = min(k_root, len(root.children))
        candidates = gumbel_top_k_root_candidates(root, k=k)

        current_candidates = list(candidates)
        # Number of halving phases ~ ceil(log2(k))
        P = max(1, math.ceil(math.log2(k)))

        for phase in range(P):
            n_cand = len(current_candidates)
            if n_cand <= 1:
                break

            # Simple budget allocation: spread n_simulations across phases & candidates
            budget_per_candidate = max(
                1,
                self.n_simulations // (P * n_cand)
            )

            current_candidates = self.sequential_halving_phase(
                root,
                current_candidates,
                env,
                budget_per_candidate,
            )

        # Choose best candidate by mean action-value Q = value / visits
        def mean_q(child):
            # Negate for root perspective
            return -child.value / child.visits if child.visits > 0 else -1e9

        best_child = max(current_candidates, key=mean_q)

        # --- POLICY IMPROVEMENT AT ROOT ---------------------------------
        state = env.get_state()
        improved_policy, v_pi = self.compute_policy_improvement(root, state)

                # Mask illegal moves BEFORE selecting the final move
        legal_mask = np.zeros_like(improved_policy)
        for move in root.children.keys():
            idx = self.move_mapping.get_index(move.uci())
            legal_mask[idx] = 1

        # Apply mask
        improved_policy = improved_policy * legal_mask

        # Renormalize
        if improved_policy.sum() == 0:
            # fallback: uniform over legal moves
            for move in root.children.keys():
                idx = self.move_mapping.get_index(move.uci())
                improved_policy[idx] = 1.0
        improved_policy /= improved_policy.sum()


        return best_child.move, improved_policy, v_pi


    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            print(len(self.memory), "memory < batch_size; skipping replay")
            return

        print("Replaying...")

        # Sample a minibatch
        import random
        minibatch = random.sample(self.memory, batch_size)
        minibatch = self.memory

        states = np.array([x[0] for x in minibatch], dtype=np.float32)
        targets_pi = np.array([x[1] for x in minibatch], dtype=np.float32)
        targets_v = np.array([x[2] for x in minibatch], dtype=np.float32)

        self.model.fit(
            states,
            [targets_pi, targets_v],
            epochs=3,
            verbose=1,
        )





    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


