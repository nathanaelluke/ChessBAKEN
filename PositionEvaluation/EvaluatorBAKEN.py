import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import os
import csv
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from IPython.display import display, SVG
from itertools import dropwhile, takewhile
import math

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

# Currently unused. Helped network understand that being in mate always 
# means the position is lost and mating means it is won.
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

# Currently unused. When input into the network it helped it understand
# material value better. Without it, it still doesn't get materal 
# importance.
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
        self.fc1 = nn.Linear(769, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 1)
        self.tanh = nn.Tanh()

      def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.tanh(self.fc6(x))
        return x



def label_output(value):
    if value <= -0.33:
        return -1
    elif value <= 0.33:
        return 0
    else:
        return 1

if __name__ == "__main__":
    data_loader = DataLoader("../KingBase2019-pgn")
    num_games = 2000000
    game_count = 0
   
    # This takes forever to run, I would suggest only running it once,
    # then commenting this block of code out. It adds 2M games to a 
    # csv.
    '''with open("game_positions_mini.csv", "w", newline='') as csvfile:
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

            #for move in all_moves[:-20]:
            #    board.push(move)

            #last_20_moves = all_moves[-20:]

            for move in all_moves:
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

    print("Done.")''' 

    # Create the model, loss function, and optimizer
    model = ChessNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_vals = []
    batch = 0

    filename = "game_positions_mini.csv"
    chunksize = 10000 # Essentially the batch size. Maybe play around with it.
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        X, y = [], []
        model.train()
        for _, row in chunk.iterrows():
            fen = row["pos"]
            turn = int(row["turn"])
            # In eariler models I input the meterial balance and if the player 
            # was mated.
            mat = math.tanh(material_balance(chess.Board(fen))/10) # Scale mat to between -1 and 1
            mate = is_mate(chess.Board(fen))
            result = int(row["result"])
            board_encoded = encode_board_only(fen)
            input_vector = board_encoded + [turn] + [mat] + [mate]
            X.append(input_vector)
            y.append(result)

        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert numpy arrays to tensors
        X_train = torch.tensor(np.array(X), dtype=torch.float32)
        y_train = torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1)
       
        for epoch in range(1000):
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            loss_vals.append(loss.item())

        if batch % 1000 == 0:
            print(f"batch {batch}, Loss: {loss.item()}")

            X_test = torch.tensor(np.array(X), dtype=torch.float32)
            y_test = torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1)

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
        batch = batch + 1

    # Plot training loss curve
    plt.plot(loss_vals)
    plt.xlabel("batchs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()

    torch.save(model.state_dict(), "EvalModelv4_1.pt")
