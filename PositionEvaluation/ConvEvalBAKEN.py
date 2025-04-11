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

def encode_board_only(fen: str) -> np.ndarray:
    encoded_data = np.zeros((12, 8, 8), dtype=np.float32)

    turn_plane_index = 12

    try:
        parts = fen.split(' ')
        piece_placement = parts[0]
        turn_indicator = parts[1] # 'w' or 'b'
    except IndexError:
        print(f"Warning: Invalid FEN string format (missing parts): {fen}")
        return encoded_data
    if turn_indicator == 'b':
      piece_to_index = {
        'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
        'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11
      }
    else:
      piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
      }


    rows = piece_placement.split('/')
    if len(rows) != 8:
        print(f"Warning: FEN string piece placement does not have 8 ranks: {fen}")
        return encoded_data

    for row_idx, row in enumerate(rows):
        col_idx = 0
        for char in row:
            if col_idx >= 8:
                 print(f"Warning: Col index {col_idx} exceeded 8 in row {row_idx} for FEN {fen}. Skipping rest.")
                 break
            if char.isdigit():
                col_idx += int(char)
            else:
                if char in piece_to_index:
                    piece_index = piece_to_index[char]
                    encoded_data[piece_index, row_idx, col_idx] = 1.0
                else:
                    print(f"Warning: Unexpected character '{char}' in FEN piece placement: {fen}")
                col_idx += 1
        if col_idx != 8:
             print(f"Warning: Row {row_idx} in FEN {fen} did not sum to 8 columns (processed {col_idx} columns).")

    if turn_indicator == 'b':
        encoded_data = np.rot90(encoded_data, k=2, axes=(1, 2))

    return encoded_data

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

def label_output(value):
    if value <= -0.33:
        return -1
    elif value <= 0.33:
        return 0
    else:
        return 1

if __name__ == "__main__":
    data_loader = DataLoader("/content/drive/Shareddrives/Chess Bot BAKEN/KingBase2019-pgn")
    num_games = 1000
    game_count = 0

    # This takes forever to run, I would suggest only running it once,
    # then commenting this block of code out. It adds 2M games to a
    # csv.
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

            '''for move in all_moves[:-20]:
                board.push(move)

            last_20_moves = all_moves[-20:]'''

            for move in all_moves:
                if move in board.legal_moves:
                    turn = 1 if board.turn == chess.WHITE else 0
                    if turn == 0:
                        result = result * -1
                    else:
                      result_str = game.headers.get("Result", "*")
                      if result_str == "1-0":
                          result = 1
                      elif result_str == "0-1":
                          result = -1
                      elif result_str == "1/2-1/2":
                          result = 0

                    board.push(move)
                    pos = board.fen()
                    writer.writerow([pos, turn, result])
                else:
                    print("Illegal move encountered:", move)
                    continue

            game_count += 1
            if game_count % 1000 == 0:
                print(f"Processed {game_count} games...")

    print("Done.")

    # Create the model, loss function, and optimizer
    model = ChessNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_vals = []

    filename = "game_positions_mini.csv"
    chunksize = 256 # Essentially the batch size. Maybe play around with it.
    num_epochs = 100
    for epoch in range(num_epochs):
      print(f"Epoch {epoch}")
      batch = 0
      for chunk in pd.read_csv(filename, chunksize=chunksize):
          X, y = [], [] # Initialize lists for the current chunk
          model.train()  # Set model to training mode for each batch

          for _, row in chunk.iterrows():
                fen = row["pos"]
                result = int(row["result"])
                turn = int(row["turn"])
                board_encoded_array = encode_board_only(fen) # Returns np.ndarray shape (12, 8, 8)

                if np.any(board_encoded_array):
                    X.append(board_encoded_array)
                    y.append(result)

          X_train = torch.tensor(np.array(X), dtype=torch.float32)
          y_train = torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1)


          #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

          # Convert numpy arrays to tensors
          X_train = torch.tensor(np.array(X), dtype=torch.float32)
          y_train = torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1)

          optimizer.zero_grad()
          output = model(X_train)
          loss = criterion(output, y_train)
          loss.backward()
          optimizer.step()
          loss_vals.append(loss.item())

          if batch % 1000 == 0:
            print(f"Batch {batch}: Loss {loss}")
          batch = batch + 1
    # Plot training loss curve
    plt.plot(loss_vals)
    plt.xlabel("batchs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()

    torch.save(model.state_dict(), "ConvModv2_1.pt")

