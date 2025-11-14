import math
import numpy as np

from src.utils.move_mapping import MoveMapping

class MCTSNode:
    slots = ("parent","children","visits","value","prior","move","expanded")
    def __init__(self,prior=0,move=None):
        self.children = {}
        self.visits = 0
        self.value = 0
        self.prior = prior
        self.move = move
        self.expanded = False


    def is_fully_expanded(self):
        return len(self.children) > 0 and all(child.visits > 0 for child in self.children.values())

    def best_child(self, c_puct):
        best_score = -float('inf')
        best_child = None
        scores = []  # List to hold (move, ucb1_score, child) tuples for printing

        for move, child in self.children.items():
            ucb1_score = (child.value / (child.visits + 1)) + c_puct * child.prior * (math.sqrt(self.visits) / (child.visits + 1))
            
            # Store the score for later printing
            scores.append((move, ucb1_score, child))
            
            # Determine the best child as before
            if ucb1_score > best_score:
                best_score = ucb1_score
                best_child = child

        # Sort scores and print the top 5
        scores.sort(key=lambda x: x[1], reverse=True)  # Sort by UCB1 score
        # print("Top 5 Children by UCB1 Score:")
        # for i, (move, score, child) in enumerate(scores[:5]):
        #     print(f"{i + 1}: Move {move} -> UCB1 Score: {score:.4f}, prior ",child.prior)
        # print("best child : ",best_child.move)

        return best_child



    def expand(self, legal_moves, priors):
        for move in legal_moves:
            if move not in self.children:
                self.children[move] = MCTSNode(prior=priors.get(move,0.0),move=move)

    def expand_leaf(self, env, model):
        move_mapping = MoveMapping()

        # Evaluate leaf with the network
        policy, value = model.predict(
            np.expand_dims(env.get_state(), axis=0), verbose=0
        )
        policy = policy[0]
        value = float(value[0])

        # Create children with priors for legal moves
        legal_moves = list(env.get_legal_actions())
        priors = {}
        # normalize priors across legal moves
        total_p = 0.0
        for mv in legal_moves:
            mv_idx = move_mapping.get_index(mv.uci())
            p = max(1e-12, policy[mv_idx])
            priors[mv] = p
            total_p += p
        for mv in legal_moves:
            priors[mv] /= total_p

        self.expand(legal_moves, priors) 
        self.expanded = True
        return value

    def update(self, value):
        self.visits += 1
        self.value += value