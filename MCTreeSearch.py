import math
import numpy as np
import random
import time
import chess
import chess.pgn
import torch
import torch.nn as nn
from PositionEvaluation import EvalFunctions

# Copied from the notebook in order to use the model here
# Simple NN architeture
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # self.fc1 = nn.Linear(64, 128)
        self.fc1 = nn.Linear(70, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


class MCNode:

    EXPLORATION_CONSTANT = 2
    eval_model = None

    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.score = 0
        MCNode.eval_model = torch.load("PositionEvaluation/eval_model.pt", weights_only=False)

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None
    
    def ucb1(self):
        if self.visits == 0:
            return math.inf
        return (self.score / self.visits) + math.sqrt(MCNode.EXPLORATION_CONSTANT * math.log(self.parent.visits) / self.visits)

    def get_actions(self):
        # TODO this should be driven by the move prediction network
        # for now, it just brute forces every legal move
        for move in list(self.state.legal_moves):
            new_state = self.state.copy()
            new_state.push(move)
            self.children[move.uci()] = MCNode(new_state, self)
    

class MCTreeSearch:
    def __init__(self, root_state):
        self.root = MCNode(root_state)
        self.root.get_actions()

    @staticmethod
    def evaluate_state(state):
        
        inputArr = EvalFunctions.fen_to_array(state.fen())
        inputTensor = torch.tensor(np.array(inputArr), dtype=torch.float32)
        score = MCNode.eval_model(inputTensor)
        return score.item()
        
        # # TODO this will use the evalutation network
        # # for now it just uses material advantage as a heuristic
        # PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        # score = 0
        # for square, piece in state.piece_map().items():
        #     value = PIECE_VALUES[piece.piece_type]
        #     score += value if piece.color == state.turn else -value
        # return score

    @staticmethod
    def rollout(leaf_node, max_depth = 10):
        # TODO this should probably use the move selection network to guide the rollout
        # for now it just chooses random moves
        curr_state = leaf_node.state
        curr_depth = 0
        curr_state_copy = curr_state.copy()
        while not curr_state_copy.is_game_over() and curr_depth < max_depth:
            curr_state_copy.push(list(curr_state_copy.legal_moves)[random.randint(0, curr_state_copy.legal_moves.count() - 1)])
            curr_depth += 1
        return MCTreeSearch.evaluate_state(curr_state_copy)
    
    @staticmethod
    def backpropogate(leaf_node, score):
        curr_node = leaf_node
        while not curr_node.is_root():
            curr_node.visits += 1
            curr_node.score += score
            curr_node = curr_node.parent
        curr_node.visits += 1
        curr_node.score += score

    def search_iteration(self):
        current_node = self.root
        while not current_node.is_leaf():
            current_node = max(current_node.children.values(), key=lambda child: child.ucb1())
        if current_node.visits != 0:
            # expand
            current_node.get_actions()
            if (not current_node.is_leaf()):
                current_node = list(current_node.children.values())[0]
        # rollout
        score_estimate = MCTreeSearch.rollout(current_node)
        MCTreeSearch.backpropogate(current_node, score_estimate)

    def best_action(self):
        action = None
        if self.root.state.turn == chess.WHITE:
            action = max(self.root.children.keys(), key=lambda child: self.root.children[child].score /self.root.children[child].visits)
        else:
            action = min(self.root.children.keys(), key=lambda child: self.root.children[child].score /self.root.children[child].visits)
        return action
    
    def search_for_time(self, thinking_time):
        start_time = time.time()
        while time.time() - start_time < thinking_time:
            self.search_iteration()

    def search_for_iterations(self, iterations):
        for i in range(iterations):
            self.search_iteration()

    def take_action(self, action):
        self.root = self.root.children[action]

    def print_tree(self, node=None, indent=1):
        if node is None:
            print(f'Root (visits: {self.root.visits}, score: {self.root.score})')
            node = self.root
            
        for child in node.children.keys():
            str = ""
            for _ in range(indent):
                str += "    "
            str += f'{child} (visits: {node.children[child].visits}, score: {node.children[child].score})'
            print(str)
            self.print_tree(node.children[child], indent + 1)

def main():
    tree_search = MCTreeSearch(chess.Board())
    game = chess.pgn.Game()
    node = game

    tree_search.print_tree()
    for i in range(100):
        tree_search.search_iteration()
        tree_search.print_tree()
        input("Press Enter to continue...")

    # for i in range(100):
    #     tree_search.search_for_iterations(100)
    #     action = tree_search.best_action()
    #     print(action)
    #     tree_search.take_action(action)
    #     node = node.add_variation(chess.Move.from_uci(action))
    #     print(tree_search.root.state)

    # print(game, file=open("game.pgn", "w"), end="\n\n")

if __name__ == "__main__":
    main()