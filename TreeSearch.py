from __future__ import annotations
import numpy as np
import chess
from MoveSelector import MoveSelector
from Evaluator import Evaluator

class MCTreeNode:
    """
    This class represents a node in the Monte Carlo Tree.
    It contains the game state of the node, its parent, children,
    visit count, and score.
    """
    def __init__(self, state: chess.Board, exploration_constant: float, parent: MCTreeNode = None) -> None:
        """
        Initializes the MCTreeNode with a given game state and parent node.
        Args:
            state (chess.Board): The game state at this node.
            exploration_constant (float): The exploration constant for UCB1.
            parent (MCTreeNode): The parent node of this node.
        """
        self.state = state
        self.exploration_constant = exploration_constant
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.score = 0

    def is_leaf(self) -> bool:
        """
        Checks if the current node is a leaf node (no children).
        Returns:
            bool: True if the node is a leaf, False otherwise.
        """
        return len(self.children) == 0

    def is_root(self) -> bool:
        """
        Checks if the current node is the root node (no parent).
        Returns:
            bool: True if the node is the root, False otherwise.
        """
        return self.parent is None
    
    def ucb1(self) -> float:
        """
        Calculates the Upper Confidence Bound (UCB1) for the node.
        Returns:
            float: The UCB1 value for this node.
        """
        if self.visits == 0:
            return float("inf")
        return (self.score / self.visits) + np.sqrt(self.exploration_constant * np.log(self.parent.visits) / self.visits)


class MCTreeSearch:
    """
    This class implements the Monte Carlo Tree Search algorithm.
    It is used to search for the best move in a given position.
    """
    def __init__(self, root_state: chess.Board, selector_path, evaluator_path, exploration_constant) -> None:
        """
        Initializes the MCTreeSearch with a given game position and model paths.
        Args:
            root_state (chess.Board): The initial game position.
            selector_path (str): The path to the move selector model.
            evaluator_path (str): The path to the evaluator model.
            exploration_constant (float): The exploration constant for UCB1.
        """
        self.exploration_constant = exploration_constant
        self.root = MCTreeNode(root_state, exploration_constant)
        self.move_selector = MoveSelector(selector_path)
        self.evaluator = Evaluator(evaluator_path)

    def search_iteration(self) -> None:
        """
        Performs one iteration of the Monte Carlo Tree Search.
        It will modify the nodes in the tree based on the search results.
        """
        current_node = self.root
        while not current_node.is_leaf():
            current_node = max(current_node.children.values(), key=lambda child: child.ucb1())
        if current_node.visits != 0:
            # expand
            self._expand_node(current_node)
            if (not current_node.is_leaf()):
                current_node = list(current_node.children.values())[0]
        # rollout
        score_estimate = self._rollout(current_node)
        self._backpropogate(current_node, score_estimate)

    def play_move(self, move: chess.Move) -> None:
        """
        Moves the root of the tree to the child node corresponding to the given move.
        Args:
            move (chess.Move): The move to play.
        """
        self.root = self.root.children[move.uci()]
        self.root.parent = None

    def get_move_list(self) -> list[tuple[str, float]]:
        """
        Returns a list of considered moves and their evaluations according to the current state of the tree search.
        This doesn't do any searching, just parses the current state of the tree.
        The list is sorted by evaluation (best for active player first).
        Returns:
            list[tuple[str, float]]: A list of tuples (move, evaluation).
        """
        moves = []
        for move in self.root.children.keys():
            child_node = self.root.children[move]
            if child_node.visits != 0:
                moves.append((move, child_node.score / child_node.visits))
        moves.sort(key=lambda x: x[1], reverse=True)
        return moves

    def _expand_node(self, node: MCTreeNode) -> None:
        """
        Expands the given node by adding moves suggested by the MoveSelector.
        Args:
            node (MCTreeNode): The node to expand.
        """
        moves = self.move_selector.get_move_probabilities(node.state)

        if len(moves) == 0:
            return

        # only consider moves that the model suggests with a probability > 1%
        moves = list(filter(lambda move: move[1] > 0.01, moves))

        for move, _ in moves:
            new_state = node.state.copy()
            new_state.push(chess.Move.from_uci(move))
            child_node = MCTreeNode(new_state, self.exploration_constant, node)
            node.children[move] = child_node

    def _rollout(self, node: MCTreeNode) -> float:
        """
        Performs a rollout guided by the MoveSelector from the given node to estimate its value.
        Args:
            node (MCTreeNode): The node to perform the rollout from.
        Returns:
            float: The estimated score for the node.
        """
        player_sign = 1 if node.state.turn == chess.WHITE else -1
        curr_state_copy = node.state.copy()
        
        for _ in range(0):
            move = self.move_selector.get_move_probabilities(curr_state_copy)[0][0]
            curr_state_copy.push(chess.Move.from_uci(move))
            if curr_state_copy.is_game_over():
                return 1.0*player_sign if curr_state_copy.result() == "1-0" else -1.0*player_sign if curr_state_copy.result() == "0-1" else 0.0
        return self.evaluator.evaluate_position(curr_state_copy)*player_sign
        
    def _backpropogate(self, node: MCTreeNode, score: float) -> None:
        """
        Backpropagates the score from the given node up to the root node.
        Args:
            node (MCTreeNode): The node to backpropagate from.
            score (float): The score to backpropagate.
        """
        curr_node = node
        while not curr_node.is_root():
            curr_node.visits += 1
            curr_node.score += score
            curr_node = curr_node.parent
            score *= -1
        curr_node.visits += 1
        curr_node.score += score