import chess
import pygame

#@title Setup board and print legal moves
board = chess.Board()
print(board)
legMov = list(board.legal_moves)
board = chess.Board()
print(f"There are {len(legMov)} legal moves")
