import chess
import chess.engine
import numpy as np
import random
import chess

from src.chess_renderer import ChessRenderer

class RookKingEnv:
    def __init__(self, stage=1,demo_mode=False):
        self.stage = stage
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish")
        self.engine.configure({"Skill Level": 20})
        self.mates = 0
        self.steps = 0
        self.demo_mode = demo_mode
        self.reset()

    def reset(self):
        self.board.reset()
        self.board.clear_board()

        if self.stage == 1:
            # Stage 1: Mate in 1 positions
            self._generate_mate_in_one()
        elif self.stage == 2:
            self._set_fixed_mate_position()
        elif self.stage == 3:
            # Stage 3: More challenging positions
            self._generate_medium_mate()
        elif self.stage == 4:
            self._generate_random_position()

        self.steps = 0
        self.done = False
        return self.get_state()

    def _generate_mate_in_one(self):
        while True:
            # Decide randomly whether to place kings on same rank or different ranks
            same_rank = random.choice([True, False])
            same_rank = True
            
            if same_rank:
                # Place kings on the same rank
                king_rank = random.randint(0, 7)

                # Place black king
                black_king_file = random.choice([0,7])
                
                if black_king_file == 7:
                    white_king_file = black_king_file - 2
                    rook_file = 6
                else:
                    white_king_file = black_king_file + 2
                    rook_file = 1

                rook_rank = random.randint(0,7)
                while abs(rook_rank-king_rank) < 2:
                    rook_rank = random.randint(0,7)
                    
                black_king_square = chess.square(black_king_file, king_rank)
                white_king_square = chess.square(white_king_file, king_rank)
                
                # Place rook on the same rank for a back-rank mate
                rook_square = chess.square(rook_file, rook_rank)

                
            else:
                # Place kings on the same file
                king_file = random.choice([0, 7])

                # Place black king on the chosen file
                black_king_rank = random.choice([0, 7])
                
                if black_king_rank == 7:
                    white_king_rank = black_king_rank - 2
                    rook_rank = 6
                else:
                    white_king_rank = black_king_rank + 2
                    rook_rank = 1

                rook_file = random.randint(0, 7)
                while abs(rook_file - king_file) < 2:
                    rook_file = random.randint(0, 7)
                    
                black_king_square = chess.square(king_file, black_king_rank)
                white_king_square = chess.square(king_file, white_king_rank)
                
                # Place rook on the same file for a vertical checkmate
                rook_square = chess.square(rook_file, rook_rank)
                print(king_file, black_king_rank, white_king_rank, rook_file, rook_rank)
            
            # Set pieces and verify position
            self.board.clear_board()
            self.board.set_piece_at(black_king_square, chess.Piece(chess.KING, chess.BLACK))
            self.board.set_piece_at(white_king_square, chess.Piece(chess.KING, chess.WHITE))
            self.board.set_piece_at(rook_square, chess.Piece(chess.ROOK, chess.WHITE))
            self.board.turn = chess.WHITE
            
            if self._is_mate_in_one():
                break

    def _set_fixed_mate_position(self):
        # Example position for checkmate in one
        self.board.clear_board()
        self.board.set_piece_at(chess.H2, chess.Piece(chess.KING, chess.BLACK))
        self.board.set_piece_at(chess.A7, chess.Piece(chess.ROOK, chess.WHITE))
        self.board.set_piece_at(chess.E3, chess.Piece(chess.KING, chess.WHITE))
        self.board.turn = chess.WHITE

    def _generate_easy_mate(self):
        while True:
            # Place black king near the edge but not on it
            black_king_rank = random.choice([1, 6])
            black_king_file = random.randint(1, 6)
            black_king_square = chess.square(black_king_file, black_king_rank)

            # Place white king at a controlling distance
            white_king_rank = black_king_rank + (-2 if black_king_rank == 6 else 2)
            white_king_file = black_king_file
            white_king_square = chess.square(white_king_file, white_king_rank)

            # Place rook to control the rank
            rook_file = random.choice([0, 7])  # Place on A or H file
            rook_rank = black_king_rank
            rook_square = chess.square(rook_file, rook_rank)

            self.board.set_piece_at(black_king_square, chess.Piece(chess.KING, chess.BLACK))
            self.board.set_piece_at(white_king_square, chess.Piece(chess.KING, chess.WHITE))
            self.board.set_piece_at(rook_square, chess.Piece(chess.ROOK, chess.WHITE))
            self.board.turn = chess.WHITE

            if self._is_valid_position() and not self._is_mate_in_one():
                break

            self.board.clear_board()

    def _generate_medium_mate(self):
        while True:
            # Place black king in the center area
            black_king_rank = random.randint(2, 5)
            black_king_file = random.randint(2, 5)
            black_king_square = chess.square(black_king_file, black_king_rank)

            # Place white king at a reasonable distance
            while True:
                white_king_rank = black_king_rank + random.choice([-2, -1, 1, 2])
                white_king_file = black_king_file + random.choice([-2, -1, 1, 2])
                if 0 <= white_king_rank <= 7 and 0 <= white_king_file <= 7:
                    break
            white_king_square = chess.square(white_king_file, white_king_rank)

            # Place rook at a strategic position
            while True:
                rook_file = random.randint(0, 7)
                rook_rank = random.randint(0, 7)
                rook_square = chess.square(rook_file, rook_rank)
                if (chess.square_distance(rook_square, black_king_square) >= 2 and
                    chess.square_distance(rook_square, white_king_square) >= 2):
                    break

            self.board.set_piece_at(black_king_square, chess.Piece(chess.KING, chess.BLACK))
            self.board.set_piece_at(white_king_square, chess.Piece(chess.KING, chess.WHITE))
            self.board.set_piece_at(rook_square, chess.Piece(chess.ROOK, chess.WHITE))
            self.board.turn = chess.WHITE

            if self._is_valid_position() and not self._is_mate_in_one():
                break

            self.board.clear_board()

    def _generate_random_position(self):
        while True:
            # Completely random positions but maintaining basic chess rules
            black_king_square = chess.square(random.randint(0, 7), random.randint(0, 7))

            while True:
                white_king_square = chess.square(random.randint(0, 7), random.randint(0, 7))
                if chess.square_distance(black_king_square, white_king_square) >= 2:
                    break

            while True:
                rook_square = chess.square(random.randint(0, 7), random.randint(0, 7))
                if (rook_square != black_king_square and
                    rook_square != white_king_square):
                    break

            self.board.set_piece_at(black_king_square, chess.Piece(chess.KING, chess.BLACK))
            self.board.set_piece_at(white_king_square, chess.Piece(chess.KING, chess.WHITE))
            self.board.set_piece_at(rook_square, chess.Piece(chess.ROOK, chess.WHITE))
            self.board.turn = chess.WHITE

            if self._is_valid_position():
                break

            self.board.clear_board()

    def _is_valid_position(self) -> bool:
        """Check if the current position is valid."""
        if self.board.is_checkmate() or self.board.is_stalemate():
            return False

        # Make sure kings are not in check
        if self.board.is_check():
            return False

        # Make sure kings are not adjacent
        white_king_square = self.board.king(chess.WHITE)
        black_king_square = self.board.king(chess.BLACK)
        if chess.square_distance(white_king_square, black_king_square) < 2:
            return False

        return True

    def _is_mate_in_one(self) -> bool:
        """Check if the position is mate in one for white."""
        if self.board.is_checkmate():
            return False

        legal_moves = list(self.board.legal_moves)
        for move in legal_moves:
            self.board.push(move)
            is_mate = self.board.is_checkmate()
            self.board.pop()
            if is_mate:
                return True
        return False

    def get_fen(self):
        """Return the current state of the board."""
        return self.board.fen()


    def are_kings_adjacent(self, square1, square2):
        rank1, file1 = chess.square_rank(square1), chess.square_file(square1)
        rank2, file2 = chess.square_rank(square2), chess.square_file(square2)
        return max(abs(rank1 - rank2), abs(file1 - file2)) <= 1

    def get_state(self):
        # Create a more informative state representation
        state = np.zeros((8, 8, 3), dtype=np.float32)  # 3 channels: WK, BK, WR

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                rank = 7 - chess.square_rank(square)  # Invert rank to match the array representation
                file = chess.square_file(square)
                if piece.piece_type == chess.KING:
                    channel = 0 if piece.color == chess.WHITE else 1
                    state[rank, file, channel] = 1.0
                elif piece.piece_type == chess.ROOK and piece.color == chess.WHITE:
                    state[rank, file, 2] = 1.0

        # Return the state without flattening
        return state

    def go_back(self, baseline):
        self.done = False
        while len(self.board.move_stack)>baseline:
            self.board.pop()


    def step(self, action):
        if action not in self.board.legal_moves:
            print("wrong move : ", action)
            print(self.board.unicode())
            return self.get_state(), -10, True

        self.board.push(action)

        if self.demo_mode:
          self.renderer.render_board(self.board)
        reward = -0.5

        # reward structure
        if self.board.is_checkmate():
            print("checkmate!!!!!!!!!")
            reward = 150.0
            self.mates += 1
            self.done = True
        elif self.board.is_stalemate() or self.board.is_game_over():
            reward = -100.0
            self.done = True
        if self.board.is_insufficient_material():
            reward = -180.0
            self.done = True

        return self.get_state(), reward, self.done
    
    def oponent_step(self):
        if not self.done:
            self.opponent_move()
            if self.board.is_stalemate() or self.board.is_game_over():
                self.done = True

    def calculate_position_reward(self):
        reward = -20.0
        black_king_square = self.board.king(chess.BLACK)
        white_king_square = self.board.king(chess.WHITE)
        rook_squares = self.board.pieces(chess.ROOK, chess.WHITE)

        # Reward for restricting black king's mobility
        black_king_moves = sum(1 for _ in self.board.legal_moves)
        reward -= black_king_moves * 1

        # Reward for keeping the black king near the edge
        rank = chess.square_rank(black_king_square)
        file = chess.square_file(black_king_square)
        distance_from_center = abs(3.5 - rank) + abs(3.5 - file)
        reward += distance_from_center

        # Reward for keeping kings close
        king_distance = chess.square_distance(black_king_square, white_king_square)
        reward += (8 - king_distance)

        for rook_square in rook_squares:
          if self.board.is_attacked_by(chess.BLACK, rook_square):
              if not self.board.is_attacked_by(chess.WHITE, rook_square):
                  reward -= 10

        return reward

    def opponent_move(self):
        if not self.board.is_game_over():
            result = self.engine.play(self.board, chess.engine.Limit(time=1.0))
            self.board.push(result.move)
        else:
          # self.render_board()
          print("over")
          return


    def get_legal_actions(self):
        return list(self.board.legal_moves)

    def step_with_opponent(self, action):
        state, reward, done = self.step(action)

        if not done:
            self.oponent_step()

        # After Stockfish reply, check terminal
        if self.board.is_game_over():
            result = self.board.result(claim_draw=True)

            if result == "1-0":         # white mates
                reward = 150.0
            elif result == "1/2-1/2":   # stalemate/draw
                reward = -50.0
            else:
                reward = -150.0         # should not happen in KRK

            self.done = True

        return self.get_state(), reward, self.done

