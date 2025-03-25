import chess
import csv

# The number of possible pieces and square states (8*8*6*2)
NUM_BOARD_NODES = 768
# The number of possible moves from any square to any other
NUM_MOVE_NODES = 1792


# Determines whether any piece could ever move from one square to another
# Note: pass in start and end coordinates as tuples containing two ints
def canMove(start, end):
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


# Generate the mapping matrix (only needs to be done once)
def getMappingMatrix():
    moveMap = [[[[0 for l in range(8)] for k in range(
        8)] for j in range(8)] for i in range(8)]
    offset = 0
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for l in range(8):
                    if canMove((i, j), (k, l)):
                        moveMap[i][j][k][l] = offset
                        offset += 1
    return moveMap

# Encode a board position and legal moves into a binary array
def encodeSelector(boardFEN, moves, moveMap):
    # Encode board position
    encodedBoard = [0 for _ in range(NUM_BOARD_NODES)]
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
    encodedMoves = [0 for _ in range(NUM_MOVE_NODES)]
    for m in moves.split(' '):
        startCol = ord(str(m)[0])-97  # e.g. c -> 2
        startRow = ord(str(m)[1])-49  # e.g. 8 -> 7
        endCol = ord(str(m)[2])-97
        endRow = ord(str(m)[3])-49
        encodedMoves[(moveMap[startCol][startRow][endCol][endRow])] = 1

    # Combine into single list
    return encodedBoard + encodedMoves


# Decode board position from a binary array of length 768
def decodePosition(input):
    # Create matrix of board
    boardMatrix = [['.' for _ in range(8)] for _ in range(8)]

    # Populate board
    for i in range(len(input)):
        if input[i] == 1:
            piece = i % 12
            # Piece: P = 0, R = 1, N = 2, B = 3, Q = 4, K = 5
            # Color: W = 0, B = 1
            pieceLabel = '.'
            match piece:
                case 0:
                    pieceLabel = 'P'
                case 1:
                    pieceLabel = 'p'
                case 2:
                    pieceLabel = 'N'
                case 3:
                    pieceLabel = 'n'
                case 4:
                    pieceLabel = 'B'
                case 5:
                    pieceLabel = 'b'
                case 6:
                    pieceLabel = 'R'
                case 7:
                    pieceLabel = 'r'
                case 8:
                    pieceLabel = 'Q'
                case 9:
                    pieceLabel = 'q'
                case 10:
                    pieceLabel = 'K'
                case 11:
                    pieceLabel = 'k'
            boardMatrix[int(i / (6*2)) % 8][int(i / (8*6*2))] = pieceLabel

    # Convert to FEN notation
    boardFEN = ''
    for i in range(len(boardMatrix)):
        empty = 0
        for j in range(len(boardMatrix[i])):
            if boardMatrix[i][j] == '.':
                empty += 1
            else:
                if empty > 0:
                    boardFEN += str(empty)
                    empty = 0
                boardFEN += boardMatrix[i][j]
        if empty > 0:
            boardFEN += str(empty)
        if i < len(boardMatrix) - 1:
            boardFEN += '/'

    return chess.Board(boardFEN)


# Decode moves from a binary array of length 1792
# Returns a list of moves with start and end squares
def decodeMoves(input, moveMap):
    moves = []
    for h in range(len(input)):
        if input[h] > 0: # NOTE: may need to change this condition based on selector NN output
            # Search moveMap to find the actual move squares
            found = False
            for i in range(8):
                for j in range(8):
                    for k in range(8):
                        for l in range(8):
                            if moveMap[i][j][k][l] == h:
                                moves.append(chr(i+97) + chr(j+49) + chr(k+97) + chr(l+49))
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
    return moves

### TESTING ###

# Test the encoder and decoder
def testEncodeDecode():
    # Encode board
    moveMap = getMappingMatrix()
    with open('MoveSelector/moveSelectorDataset.csv', 'r') as file:
        data = list(csv.reader(file))
    curr = data[0]
    encoded = encodeSelector(curr[1], curr[2], moveMap)

    # Decode board
    decodedBoard = decodePosition(encoded[0:NUM_BOARD_NODES])
    decodedMoves = decodeMoves(encoded[-NUM_MOVE_NODES:], moveMap)

    # Compare original and decoded position/moves
    print('Original position and legal moves: ')
    print(chess.Board(curr[1]))
    print(sorted(curr[2].split(' ')))
    print('---------------')
    print('Decoded position and legal moves: ')
    print(decodedBoard)
    print(decodedMoves)

testEncodeDecode()