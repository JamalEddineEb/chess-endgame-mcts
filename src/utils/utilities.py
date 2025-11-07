import numpy as np
import math

from src.utils.move_mapping import MoveMapping, gather_legal_logits

def sample_gumbel(shape, eps=1e-9):
    # U ~ Uniform(0,1) clipped for stability
    U = np.random.uniform(low=eps, high=1.0 - eps, size=shape)
    return -np.log(-np.log(U))


def gumbel_top_k_root_candidates(model, get_state, env, legal_moves, k):
    # Compute policy over all moves then restrict to legal
    move_mapping = MoveMapping()
    policy_logits, _ = model.predict(np.expand_dims(get_state(env.board), axis=0), verbose=0)
    policy_logits = policy_logits[0]

    legal_moves = list(env.get_legal_actions())
    legal_idx = move_mapping.batch_moves_to_indices(legal_moves)

    # Build arrays aligned with legal_moves
    legal_logits = gather_legal_logits(policy_logits,legal_idx)

    # Gumbel-Top-k scores
    g = sample_gumbel(legal_logits.shape)
    scores = legal_logits + g

    # Top-k indices without replacement
    k_eff = min(k, len(legal_moves))
    topk_idx = np.argpartition(-scores, kth=k_eff-1)[:k_eff]
    # Optional: sort the top-k by score descending
    topk_idx = topk_idx[np.argsort(-scores[topk_idx])]

    candidates = [legal_moves[i] for i in topk_idx]
    return candidates, policy_logits, scores

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

def expand_leaf(node, env, get_state, model):
    move_mapping = MoveMapping()

    # Evaluate leaf with the network
    policy, value = model.predict(
        np.expand_dims(get_state(env.board), axis=0), verbose=0
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

    node.expand(legal_moves, priors) 
    return value


def backup_path(path, leaf_value, to_play_sign=1):
    v = leaf_value
    for n in reversed(path):
        n.visits += 1
        n.value += v
