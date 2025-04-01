import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import os

class ValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.tanh(x)
        return x

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
            with open(f"{self.data_path}/{files[self.current_file]}", 'r') as f:
                f.seek(self.current_offset)
                while len(self.cached_games) < self.cache_size:
                    game = chess.pgn.read_game(f)
                    self.current_offset = f.tell()
                    if (game == None):
                        self.current_file += 1
                        self.current_offset = 0
                        break
                    self.cached_games.append(game)
                
def encode_position(board: chess.Board):
    position_tensor = torch.zeros((8, 8, 12), dtype=torch.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            position_tensor[square // 8][square % 8][piece.piece_type - (1 if piece.color == chess.WHITE else 0)] = 1.0
    return torch.cat((position_tensor.flatten(), torch.tensor([1.0 if board.turn == chess.WHITE else -1.0], dtype=torch.float32)), 0)
            
def main():
    model = ValueNet(8*8*12+1, 256, 128, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    running_loss = 0.0
    steps_completed = 0

    data_loader = DataLoader("../KingBase2019-pgn")
    
    for i in range(10000):
        game = data_loader.get_next_game()
        if game is None:
            break

        if i % 100 == 99:
            print(f"Processing game {i+1}")

        board = game.board()

        result = game.headers["Result"]
        if result == "1-0":
            result_tensor = torch.tensor([1.0])
        elif result == "0-1":
            result_tensor = torch.tensor([-1.0])
        else:
            result_tensor = torch.tensor([0.0])

        for move in game.mainline_moves():
            board.push(move)
            position_tensor = encode_position(board)
            optimizer.zero_grad()
            output = model(position_tensor)
            loss = criterion(output, result_tensor)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            steps_completed += 1
            if steps_completed % 1000 == 0:
                print(f"\tStep {steps_completed}, Loss: {running_loss / 1000}")
                running_loss = 0.0
    
    torch.save(model.state_dict(), "ValueNet.pt")


        

if __name__ == "__main__":
    main()