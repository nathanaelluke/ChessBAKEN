import chess
import torch
import torch.nn as nn
import numpy as np

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, input):
        x = self.features(input)
        output = self.classifier(x)
        return output

class Evaluator:
    """
    This class represents the network that evaluates positions
    to give a win probability, which is used to evaluate the leaf nodes
    in the tree search.
    """
    def __init__(self, model_path: str) -> None:
        """
        Initializes the Evaluator with a given model path.
        Args:
            model_path (str): The path to the model file.
        """
        self.model = ChessNet()
        self.model.load_state_dict(torch.load(model_path))
    
    def evaluate_position(self, board: chess.Board) -> float:
        """
        Given a board state, returns the evaluation score for that position.
        A score of 1.0 indicates white is completely winning,
        and -1.0 indicates black is completely winning.
        0.0 indicates a draw or equal position.
        Args:
            board (chess.Board): The current board state.
        Returns:
            float: The evaluation score for the position.
        """
        fen = board.fen()
        encoded_board = self._encode_board_only(fen)
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(encoded_board, dtype=torch.float32).unsqueeze(0)
            output = self.model.forward(input_tensor)
            score = output.item()
        return score

    def _encode_board_only(self, fen: str) -> np.ndarray:
        encoded_data = np.zeros((12, 8, 8), dtype=np.float32)

        turn_plane_index = 12

        try:
            parts = fen.split(' ')
            piece_placement = parts[0]
            turn_indicator = parts[1] # 'w' or 'b'
        except IndexError:
            print(f"Warning: Invalid FEN string format (missing parts): {fen}")
            return encoded_data
        if turn_indicator == 'b':
            piece_to_index = {
                'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
                'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11
            }
        else:
            piece_to_index = {
                'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
            }


        rows = piece_placement.split('/')
        if len(rows) != 8:
            print(f"Warning: FEN string piece placement does not have 8 ranks: {fen}")
            return encoded_data

        for row_idx, row in enumerate(rows):
            col_idx = 0
            for char in row:
                if col_idx >= 8:
                    print(f"Warning: Col index {col_idx} exceeded 8 in row {row_idx} for FEN {fen}. Skipping rest.")
                    break
                if char.isdigit():
                    col_idx += int(char)
                else:
                    if char in piece_to_index:
                        piece_index = piece_to_index[char]
                        encoded_data[piece_index, row_idx, col_idx] = 1.0
                    else:
                        print(f"Warning: Unexpected character '{char}' in FEN piece placement: {fen}")
                    col_idx += 1
            if col_idx != 8:
                print(f"Warning: Row {row_idx} in FEN {fen} did not sum to 8 columns (processed {col_idx} columns).")

        if turn_indicator == 'b':
            encoded_data = np.rot90(encoded_data, k=2, axes=(1, 2)).copy()

        return encoded_data