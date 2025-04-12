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
        boardFEN = board.fen().split(' ')[0]
        moves = ' '.join([move.uci() for move in board.legal_moves])
        moveMap = self._getMappingMatrix()

        encodedMoves = [[]]
        encodedMoves[0] = self._encodeSelector(boardFEN, moves, moveMap)

        X = torch.tensor(encodedMoves).float()
        torch.set_printoptions(threshold=torch.inf)
        self.model.eval()
        with torch.no_grad():
            modelMoves = self.model.forwardEval(X)

        decodedMoves = self._decodeLegalMoves(X[0][-1792:], modelMoves[0], moveMap, -100)

        softmax = nn.Softmax(dim=0)
        normalizedScores = list(softmax(torch.tensor([x[1] for x in decodedMoves])))
        for i in range(len(decodedMoves)):
            decodedMoves[i] = (decodedMoves[i][0], normalizedScores[i])

        # print(self._sumOutput(decodedMoves))
        decodedMoves = sorted(decodedMoves, key=lambda x: x[1], reverse=True)
        return decodedMoves

    def _encodeSelector(self, boardFEN, moves, moveMap):
        # Encode board position
        encodedBoard = [0 for _ in range(768)]
        rowsFEN = boardFEN.split('/')
        for i in range(8):
            col = 0
            for j in range(len(rowsFEN[i])):
                if rowsFEN[i][j].isdigit():
                    col += ord(rowsFEN[i][j])-48
                else:
                    match rowsFEN[i][j]:
                        case 'P':
                            idx = (8*6*2*col) + (6*2*i) + (0*2) + 0
                            encodedBoard[idx] = 1
                        case 'N':
                            idx = (8*6*2*col) + (6*2*i) + (1*2) + 0
                            encodedBoard[idx] = 1
                        case 'B':
                            idx = (8*6*2*col) + (6*2*i) + (2*2) + 0
                            encodedBoard[idx] = 1
                        case 'R':
                            idx = (8*6*2*col) + (6*2*i) + (3*2) + 0
                            encodedBoard[idx] = 1
                        case 'Q':
                            idx = (8*6*2*col) + (6*2*i) + (4*2) + 0
                            encodedBoard[idx] = 1
                        case 'K':
                            idx = (8*6*2*col) + (6*2*i) + (5*2) + 0
                            encodedBoard[idx] = 1
                        case 'p':
                            idx = (8*6*2*col) + (6*2*i) + (0*2) + 1
                            encodedBoard[idx] = 1
                        case 'n':
                            idx = (8*6*2*col) + (6*2*i) + (1*2) + 1
                            encodedBoard[idx] = 1
                        case 'b':
                            idx = (8*6*2*col) + (6*2*i) + (2*2) + 1
                            encodedBoard[idx] = 1
                        case 'r':
                            idx = (8*6*2*col) + (6*2*i) + (3*2) + 1
                            encodedBoard[idx] = 1
                        case 'q':
                            idx = (8*6*2*col) + (6*2*i) + (4*2) + 1
                            encodedBoard[idx] = 1
                        case 'k':
                            idx = (8*6*2*col) + (6*2*i) + (5*2) + 1
                            encodedBoard[idx] = 1
                    col += 1

        # Encode possible moves
        encodedMoves = [0 for _ in range(1792)]
        for m in moves.split(' '):
            startCol = ord(str(m)[0])-97  # e.g. c -> 2
            startRow = ord(str(m)[1])-49  # e.g. 8 -> 7
            endCol = ord(str(m)[2])-97
            endRow = ord(str(m)[3])-49
            encodedMoves[(moveMap[startCol][startRow][endCol][endRow])] = 1

        # Combine into single list
        return encodedBoard + encodedMoves
    
    def _decodeLegalMoves(self, original, input, moveMap, threshold = 0):
        indices = [i for i, x in enumerate(original) if x == 1]
        moves = []
        for h in indices:#range(len(input)):
            if input[h] > threshold: # NOTE: may need to change this condition based on selector NN output
                # Search moveMap to find the actual move squares
                found = False
                for i in range(8):
                    for j in range(8):
                        for k in range(8):
                            for l in range(8):
                                if moveMap[i][j][k][l] == h:
                                    moves.append((chr(i+97) + chr(j+49) + chr(k+97) + chr(l+49), input[h].item()))
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
        return moves
    
    def _getMappingMatrix(self):
        moveMap = [[[[-1 for l in range(8)] for k in range(
            8)] for j in range(8)] for i in range(8)]
        offset = 0
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    for l in range(8):
                        if self._canMove((i, j), (k, l)):
                            moveMap[i][j][k][l] = offset
                            offset += 1
        return moveMap
    
    def _canMove(self, start, end):
        # check duplicate
        if start[0] == end[0] and start[1] == end[1]:
            return False

        # check row or col
        if start[0] == end[0] or start[1] == end[1]:
            return True

        # check diagonal
        if abs(start[0]-end[0]) == abs(start[1]-end[1]):
            return True

        # check knight
        knightCases = [(1, 2), (1, -2), (-1, 2), (-1, -2),
                    (2, 1), (2, -1), (-2, 1), (-2, -1)]
        for c in knightCases:
            if start[0] + c[0] == end[0] and start[1] + c[1] == end[1]:
                return True

        return False
    
    def _sumOutput(self, input):
        sum = 0.0
        for i in input:
            sum += i[1]
        return sum