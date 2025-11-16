import chess
import chess.svg

from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QWidget, QVBoxLayout



class ChessRenderer(QWidget):
    def __init__(self, board):
        super().__init__()

        self.setWindowTitle("Chess Board")
        self.resize(800, 800)

        # Layout (smoother and cleaner than manual geometry)
        layout = QVBoxLayout(self)

        self.svg_widget = QSvgWidget()
        layout.addWidget(self.svg_widget)

        # Prepare board
        self.board = board

        # Initial render
        self.update_board()

    def update_board(self):
        chessboardSvg = chess.svg.board(self.board).encode("UTF-8")
 
        self.svg_widget.load(chessboardSvg)
        self.svg_widget.update()
