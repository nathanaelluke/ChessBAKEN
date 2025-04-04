# these are just the functions from the notebook
# i pasted them here so i could import them for the tree search

import chess
import numpy as np

def king_safety(board):
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)

    # Helper function to evaluate safety of a king given its position and color
    def evaluate_king(board, king_square, color):
        if king_square is None:
            return 0

        # Get the king's coordinates
        safety_score = 1.0
        file = chess.square_file(king_square)
        rank = chess.square_rank(king_square)

        # Check where the pawns are shielding the king
        pawn_squares = [
            chess.square(file + file_offset, rank + (1 if color == chess.WHITE else -1))
            for file_offset in [-1, 0, 1]
            if 0 <= file + file_offset < 8 and 0 <= rank + (1 if color == chess.WHITE else -1) < 8
        ]

        # Adjust the safety score by the amount of pawns protecting the king
        pawn_shield = sum(1 for sq in pawn_squares if board.piece_at(sq) and board.piece_at(sq).symbol().lower() == 'p')
        safety_score -= (3 - pawn_shield) * 0.1

        # Check for open files that the king is on
        file = chess.square_file(king_square)
        open_file_penalty = 0.2 if all(board.piece_at(chess.square(file, r)) is None for r in range(8)) else 0
        safety_score -= open_file_penalty

        enemy_pieces = sum(1 for sq in chess.SQUARES if board.is_attacked_by(not color, sq) and chess.square_distance(sq, king_square) <= 2)
        safety_score -= min(enemy_pieces * 0.05, 0.3)

        # Check the mobility of the king
        king_moves = len(list(board.legal_moves))
        mobility_penalty = 0.2 if king_moves < 2 else 0
        safety_score -= mobility_penalty

        # Penalize if the king is in check
        if board.is_check():
            safety_score -= 0.3

        return max(0, min(safety_score, 1))

    white_safety = evaluate_king(board, white_king_square, chess.WHITE)
    black_safety = evaluate_king(board, black_king_square, chess.BLACK)

    return white_safety, black_safety

def piece_safety(board):
    # Assign values to pieces
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }

    white_safety = 0
    black_safety = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Check if the piece is under attack
            attackers = board.attackers(not piece.color, square)
            if attackers:
                # Penalize the score if the piece is under attack
                piece_value = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    white_safety -= piece_value
                else:
                    black_safety -= piece_value

    return white_safety, black_safety

def evaluate_material(board):
    # Assign values to pieces
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }

    white_score = 0
    black_score = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Retrieve the material value of the pieces
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_score += value
            else:
                black_score += value

    return white_score, black_score

def fen_to_array(fen):
    # Set a mapping to all of the pieces
    piece_map = {
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        '.': 0
    }

    # Create a board
    board = chess.Board(fen)
    board_array = np.zeros(64, dtype=int)

    # Fill the board with pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        board_array[square] = piece_map.get(piece.symbol(), 0) if piece else 0

    # Calculate features for the given board position
    white_king_safety, black_king_safety = king_safety(board)
    white_piece_safety, black_piece_safety = piece_safety(board)
    white_material, black_material = evaluate_material(board)

    return np.append(board_array, [
        white_king_safety, black_king_safety,
        white_piece_safety, black_piece_safety,
        white_material, black_material
    ])