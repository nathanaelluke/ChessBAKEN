import chess
import chess.pgn
import pygame
import sys
import os

# Grabs images from image directory
def loadImages():
    pieces = {}
    imagePath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Images")
    piece_map = {
        'p': 'b_pawn.png', 'r': 'b_rook.png', 'n': 'b_knight.png', 'b': 'b_bishop.png', 'q': 'b_queen.png', 'k': 'b_king.png',
        'P': 'w_pawn.png', 'R': 'w_rook.png', 'N': 'w_knight.png', 'B': 'w_bishop.png', 'Q': 'w_queen.png', 'K': 'w_king.png'
    }
    
    for piece, filename in piece_map.items():
        pieces[piece] = pygame.transform.scale(pygame.image.load(os.path.join(imagePath, filename)), (100, 100))
    
    return pieces

# Draws the pygame board
def drawBoard(screen, board, pieceImages, selSquare, player_turn):
    colors = [pygame.Color("darkgreen"), pygame.Color("lightgray")]  # Improved visibility
    square_size = 100
    
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
            
            pygame.draw.rect(screen, color, pygame.Rect(col * square_size, row * square_size, square_size, square_size))
            
            if piece:
                screen.blit(pieceImages[piece.symbol()], (col * square_size, row * square_size))

# Displays a game
def displayGame():
    pygame.init()
    square_size = 100 # Size of board
    board = chess.Board() # Blank board
    screen = pygame.display.set_mode((square_size * 8, square_size * 8))
    pieceImages = loadImages() # Images from image directory
    selSquare = None # The square the player has selected
    player_turn = chess.WHITE  # Whose turn
    dragging = False # If a piece has been clicked
    running = True # If the game is running

    while running:
        screen.fill(pygame.Color("black"))
        drawBoard(screen, board, pieceImages, selSquare, player_turn)
        pygame.display.flip()
        
        # The code below was for testing the board's functionality.
        # All we would need to do for our purposes is to run 
        # drawBoard with the board we want to display.
        ######################## FOR TESTING ########################
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col, row = x // square_size, y // square_size
                square = chess.square(col, 7 - row)
                print(f"Square: {square}")
                piece = board.piece_at(square)
                print(f"Piece: {piece}")
                if piece and piece.color == player_turn:
                    selSquare = (col, row)
                    dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and selSquare and dragging:
                x, y = event.pos
                new_col, new_row = x // square_size, y // square_size
                
                # Stops crash when placing piece on same square
                if new_col == col and new_row == row:
                    break

                move = chess.Move.from_uci(f"{chess.square_name(
                    chess.square(col, 7 - row))}{
                        chess.square_name(chess.square(new_col, 7 - new_row))}")
                print(f"Move: {move}")
                if move in board.legal_moves:
                    board.push(move)
                    player_turn = not player_turn
                selSquare = None
                dragging = False
            
            # Allows board to be reset with ctrl + R
            if event.type == pygame.KEYDOWN:
                if (event.key == pygame.K_r and 
                    (pygame.key.get_mods() & pygame.KMOD_CTRL)):
                    board = chess.Board()
                    pieceImages = loadImages()
                    selSquare = None
                    player_turn = chess.WHITE
                    dragging = False
        ######################## FOR TESTING ########################
    print("Result:", board.result())

# Kick off a game
if __name__ == "__main__":
    displayGame()

