import chess
import torch
import torch.nn as nn

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
        self.model = torch.load(model_path, weights_only=False)
    
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
        pass