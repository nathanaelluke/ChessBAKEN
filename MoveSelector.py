import chess
import torch
import torch.nn as nn

class myFCN(nn.Module):
    def __init__(self, in_size: int, hidden_sizes: list[int], out_size: int) -> None:
        
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_sizes
        self.out_size = out_size

        self.lin1 = nn.Linear(in_size, hidden_sizes[0])
        self.lin2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.lin3 = nn.Linear(hidden_sizes[1], out_size)

        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.relu(self.bn1(self.lin1(x)))
        x = self.relu(self.bn2(self.lin2(x)))
        x = self.softmax(self.lin3(x))
        return x

    def forwardEval(self, x):
        x = self.relu(self.bn1(self.lin1(x)))
        x = self.relu(self.bn2(self.lin2(x)))
        x = self.lin3(x)
        return x

class MoveSelector:
    """
    This class represents the network that gives move probabilities
    for a given board state. It is used to guide the tree search.
    """
    def __init__(self, model_path: str) -> None:
        """
        Initializes the MoveSelector with a given model path.
        Args:
            model_path (str): The path to the model file.
        """
        self.model = torch.load(model_path, weights_only=False)
    
    def get_move_probabilities(self, board: chess.Board) -> list[tuple[str, float]]:
        """
        Given a board state, returns a list of tuples containing all legal moves
        and their probabilities according to the model, sorted by probability (highest first).
        Args:
            board (chess.Board): The current board state.
        Returns:
            list[tuple[str, float]]: A list of tuples (move, probability) sorted by probability.
        """
        pass
