import math

class MCTSNode:
    def __init__(self, board,agent,parent=None, prior=0,move=None):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.prior = prior
        self.agent = agent
        self.move = move


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
        print("Top 5 Children by UCB1 Score:")
        for i, (move, score, child) in enumerate(scores[:5]):
            print(f"{i + 1}: Move {move} -> UCB1 Score: {score:.4f}")
        print("best child : ",best_child.move)

        return best_child



    def expand(self, legal_moves, policy):
        for move in legal_moves:
            if move not in self.children:
                # Create a new board state by pushing the move
                new_board = self.board.copy()
                new_board.push(move)
                move_idx = self.agent.move_mapping[move.uci()]
                self.children[move] = MCTSNode(new_board, agent=self.agent,parent=self, prior=policy[move_idx],move=move)

    def update(self, value):
        self.visits += 1
        self.value += value