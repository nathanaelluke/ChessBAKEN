import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import os
import csv
import random
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from IPython.display import display, SVG

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.cached_games = []
        self.cache_size = 1000
        self.current_file = 0
        self.current_offset = 0
        self.load_games()

    def get_next_game(self):
        if (self.cached_games == []):
            self.load_games()
        if (self.cached_games == []):
            return None
        return self.cached_games.pop(0)

    def load_games(self):
        files = os.listdir(self.data_path)
        while len(self.cached_games) < self.cache_size and self.current_file < len(files):
            file_path = f"{self.data_path}/{files[self.current_file]}"
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    f.seek(self.current_offset)
                    while len(self.cached_games) < self.cache_size:
                        start_offset = f.tell()
                        try:
                            game = chess.pgn.read_game(f)
                            self.current_offset = f.tell()
                        except Exception as e_read:
                            print(f"Error reading game from {files[self.current_file]} at offset {start_offset}: {e_read}")
                            self.current_offset = 0
                            break

                        if game is None:
                            self.current_file += 1
                            self.current_offset = 0
                            break
                        self.cached_games.append(game)

            except FileNotFoundError:
                print(f"Warning: File not found {file_path}. Skipping.")
                self.current_file += 1
                self.current_offset = 0
            except Exception as e_open:
                print(f"Error opening or seeking in file {file_path}: {e_open}")
                self.current_file += 1
                self.current_offset = 0
            if self.current_offset == 0 and len(self.cached_games) < self.cache_size:
                if game is None:
                   pass
                else:
                    self.current_file += 1

def is_mate(board):
  if board.is_checkmate():
      if board.turn == chess.WHITE:
          return -1
      else:
          return 1
  return 0

def encode_board_only(fen: str) -> list:
    encoded = [0] * 768
    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    rows = fen.split('/')
    for row_idx, row in enumerate(rows):
        col_idx = 0
        for char in row:
            if char.isdigit():
                col_idx += int(char)
            else:
                piece_index = piece_to_index[char]
                square_index = row_idx * 8 + col_idx
                flat_index = square_index * 12 + piece_index
                encoded[flat_index] = 1
                col_idx += 1
    return encoded

def load_encoded_data(csv_file):
    X, y = [], []
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fen = row["pos"]
            turn = int(row["turn"])
            mat = material_balance(chess.Board(fen))
            mate = is_mate(chess.Board(fen))
            result = int(row["result"])
            board_encoded = encode_board_only(fen)
            input_vector = board_encoded + [turn] + [mat] + [mate]
            X.append(input_vector)
            y.append(result)
    return np.array(X), np.array(y)

def material_balance(board):
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    return (
        chess.popcount(white & board.pawns) - chess.popcount(black & board.pawns) +
        3 * (chess.popcount(white & board.knights) - chess.popcount(black & board.knights)) +
        3 * (chess.popcount(white & board.bishops) - chess.popcount(black & board.bishops)) +
        5 * (chess.popcount(white & board.rooks) - chess.popcount(black & board.rooks)) +
        9 * (chess.popcount(white & board.queens) - chess.popcount(black & board.queens))
    )

class ChessNet(nn.Module):
      def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(771, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

      def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x


def label_output(value):
    if value <= -0.33:
        return -1
    elif value <= 0.33:
        return 0
    else:
        return 1

def display_chess_board(fen):
    board = chess.Board(fen)
    svg_data = chess.svg.board(board, size=250)
    display(SVG(svg_data))

def read_and_display_first_game(pgn_file):
    itr = 0
    with open(pgn_file) as f:
        game = chess.pgn.read_game(f)
        if game:
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                itr+=1
                if itr == 80:
                    encoding = encode_board_only(board.fen().split()[0])
                    turn = 1 if board.turn == chess.WHITE else 0
                    mat = material_balance(chess.Board(fen))
                    mate = is_mate(chess.Board(fen))
                    inp = encoding + [turn] + [mat] + [mate]
                    e = torch.tensor(np.array(inp), dtype=torch.float32)

                    print(turn)

                    model.eval()

                    pos = e.unsqueeze(0)

                    with torch.no_grad():
                        output = model(pos)
                        prediction = output
                    print(prediction)

                    result_str = game.headers.get("Result", "*")
                    if result_str == "1-0":
                        print('White won')
                    elif result_str == "0-1":
                        print('Black won')
                    elif result_str == "1/2-1/2":
                        print('Tie')
                    else:
                        continue

                    display_chess_board(board.fen())
        else:
            print("No games found in the PGN file.")


if __name__ == "__main__":
    data_loader = DataLoader("../KingBase2019-pgn")
    num_games = 50000
    game_count = 0

    with open("game_positions_mini.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["pos", "turn", "result"])

        for i in range(num_games):
            game = data_loader.get_next_game()
            if game is None:
                break

            result_str = game.headers.get("Result", "*")
            if result_str == "1-0":
                result = 1
            elif result_str == "0-1":
                result = -1
            elif result_str == "1/2-1/2":
                result = 0
            else:
                continue

            board = game.board()
            all_moves = list(game.mainline_moves())

            for move in all_moves[:-20]:
                board.push(move)

            last_20_moves = all_moves[-20:]

            for move in last_20_moves:
                if move in board.legal_moves:
                    board.push(move)
                    pos = board.fen().split()[0]
                    turn = 1 if board.turn == chess.WHITE else 0
                    writer.writerow([pos, turn, result])
                else:
                    print("Illegal move encountered:", move)
                    continue

            game_count += 1
            if game_count % 1000 == 0:
                print(f"Processed {game_count} games...")

    print("Done.")

    X, y = load_encoded_data("game_positions_mini.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training size:", len(X_train))
    print("Test size:", len(X_test))

    # Create the model, loss, and optimizer
    model = ChessNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert np arrays to tensors
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32).view(-1, 1)

    loss_vals = []
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        loss_vals.append(loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")


    plt.plot(loss_vals)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()

    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test = torch.tensor(np.array(y_test), dtype=torch.float32).view(-1, 1)

    model.eval()

    with torch.no_grad():
        output = model(X_test)

    predictions = output.numpy().flatten()
    predicted_labels = np.array([label_output(val) for val in predictions])

    true_labels = y_test.numpy().flatten()
    true_labels_labeled = np.array([label_output(val) for val in true_labels])

    accuracy = accuracy_score(true_labels_labeled, predicted_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    criterion = nn.MSELoss()
    test_loss = criterion(output, y_test)
    print(f"Test Loss (MSE): {test_loss.item():.4f}")

    model.eval()

    for i in range(20):
        pos = X_test[i]
        true = y_test[i]

        pos = pos.unsqueeze(0)

        with torch.no_grad():
            output = model(pos)
            prediction = output

        print(f"Position {i+1}:")
        print(f"True Value: {true}")
        print(f"Predicted Value: {prediction}\n")

    torch.save(model.state_dict(), "EvalModelv1_0.pt")
