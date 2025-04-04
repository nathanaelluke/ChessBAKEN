import chess
import chess.pgn
import pygame
import torch
import PositionEvaluation.EvaluatorBAKEN as Eval
import numpy as np 

# Grabs images from image directory
def loadImages():
    pieces = {}
    imagePath = "./Images"
    pieceMap = {
        'p': 'b_pawn.png', 'r': 'b_rook.png', 'n': 'b_knight.png', 'b': 'b_bishop.png', 'q': 'b_queen.png', 'k': 'b_king.png',
        'P': 'w_pawn.png', 'R': 'w_rook.png', 'N': 'w_knight.png', 'B': 'w_bishop.png', 'Q': 'w_queen.png', 'K': 'w_king.png'
    }
    
    for piece, filename in pieceMap.items():
        pieces[piece] = pygame.transform.scale(
            pygame.image.load(f"{imagePath}/{filename}"), (50, 50))
    
    return pieces

# Draws the pygame board
def drawBoard(screen, board, pieceImages, selSquare, playerTurn):
    colors = [pygame.Color("darkgreen"), pygame.Color("lightgray")]  # Improved visibility
    squareSize = 50
    
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            if selSquare == (col, row):
                color = pygame.Color("blue")
            
            piece = board.piece_at(chess.square(col, 7 - row))
           
            if board.is_checkmate():
                if not board.turn and str(piece) == 'k':
                    color = pygame.Color("red")
                elif board.turn and str(piece) == 'K':
                    color = pygame.Color("red")
            elif board.is_check():
                if not board.turn and str(piece) == 'k':
                    color = pygame.Color("yellow")
                elif board.turn and str(piece) == 'K':
                    color = pygame.Color("yellow")
            
            pygame.draw.rect(screen, color, pygame.Rect(col * squareSize, row * squareSize, squareSize, squareSize))
            
            if piece:
                screen.blit(pieceImages[piece.symbol()], (col * squareSize, row * squareSize))

# Displays a game
def displayGame():
    pygame.init()
    squareSize = 50 # Size of board
    board = chess.Board() # Blank board
    screen = pygame.display.set_mode((squareSize * 8, squareSize * 8))
    pieceImages = loadImages() # Images from image directory
    selSquare = None # The square the player has selected
    playerTurn = chess.WHITE  # Whose turn
    dragging = False # If a piece has been clicked
    running = True # If the game is running

    model = Eval.ChessNet()
    model.load_state_dict(torch.load("PositionEvaluation/EvalModelv0_1.pt"))
    model.eval()
    
    while running:
        screen.fill(pygame.Color("black"))
        drawBoard(screen, board, pieceImages, selSquare, playerTurn)
        pygame.display.flip()
        
        # The code below was for testing the board's functionality.
        # All we would need to do for our purposes is to run 
        # drawBoard with the board we want to display.
        ######################## FOR TESTING ########################
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col, row = x // squareSize, y // squareSize
                square = chess.square(col, 7 - row)
                print(f"Square: {square}")
                piece = board.piece_at(square)
                print(f"Piece: {piece}")
                if piece and piece.color == playerTurn:
                    selSquare = (col, row)
                    dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and selSquare and dragging:
                x, y = event.pos
                newCol, newRow = x // squareSize, y // squareSize
                
                # Stops crash when placing piece on same square
                if newCol == col and newRow == row:
                    break

                move = chess.Move.from_uci(f"{chess.square_name(chess.square(col, 7 - row))}{chess.square_name(chess.square(newCol, 7 - newRow))}")
                print(f"Move: {move}")
                if move in board.legal_moves:
                    board.push(move)
                    print(board)
                    turn = 1 if board.turn == chess.WHITE else 0
                    mat = Eval.material_balance(board)
                    mate = Eval.is_mate(board)
                    encoding = Eval.encode_board_only(board.fen().split()[0])
                    inp = encoding + [turn] + [mat] + [mate]
                    e = torch.tensor(np.array(inp), dtype=torch.float32)
                    pos = e.unsqueeze(0)

                    with torch.no_grad():
                        output = model(pos)
                        prediction = output
                    print(f"Prediction: {prediction}")
                    
                    playerTurn = not playerTurn
                selSquare = None
                dragging = False
            
            # Allows board to be reset with ctrl + R
            if event.type == pygame.KEYDOWN:
                if (event.key == pygame.K_r and 
                    (pygame.key.get_mods() & pygame.KMOD_CTRL)):
                    board = chess.Board()
                    pieceImages = loadImages()
                    selSquare = None
                    playerTurn = chess.WHITE
                    dragging = False
        ######################## FOR TESTING ########################

    print("Result:", board.result())

# Kick off a game
if __name__ == "__main__":
    displayGame()

