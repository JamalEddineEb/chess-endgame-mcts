import numpy as np
import math

from src.utils.move_mapping import MoveMapping, gather_legal_logits



def sample_gumbel(shape, eps=1e-9):
    # U ~ Uniform(0,1) clipped for stability
    U = np.random.uniform(low=eps, high=1.0 - eps, size=shape)
    return -np.log(-np.log(U))



def gumbel_top_k_root_candidates(root, k):
    # Gumbel-Top-k scores
    g = sample_gumbel(len(root.children))

    scores = {}
    for i, (move, child) in enumerate(root.children.items()):
        scores[child] = child.prior + g[i]

    sorted_moves = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top_k_children = [child for child, _ in sorted_moves[:k]]

    return top_k_children



def puct_select_child(node, c_puct):
    best, best_score = None, -float('inf')
    sqrt_N = math.sqrt(max(1, node.visits))
    for move, child in node.children.items():
        Q = 0.0 if child.visits == 0 else (child.value_sum / child.visits)
        U = c_puct * child.prior * (sqrt_N / (1 + child.visits))
        score = Q + U
        if score > best_score:
            best, best_score = child, score
    return best


def ceil_log2(x):
    return math.ceil(math.log2(max(1, x)))

def backup_path(path, leaf_value, to_play_sign=1):
    v = leaf_value
    for n in reversed(path):
        n.visits += 1
        n.value += v



        
