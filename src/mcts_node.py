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