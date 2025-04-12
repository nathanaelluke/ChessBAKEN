import sys
import time
import threading
import chess
from TreeSearch import MCTreeSearch
from MoveSelector import myFCN
from Evaluator import ChessNet

class BAKEN:
    """
    This class represents the BAKEN chess engine.
    """
    def __init__(self, selector_path: str, evaluator_path: str, exploration_constant: float) -> None:
        """
        Initializes the BAKEN engine with given model paths.
        Args:
            selector_path (str): The path to the move selector model.
            evaluator_path (str): The path to the evaluator model.
            exploration_constant (float): The exploration constant for MCTS.
        """
        self.tree_search = MCTreeSearch(chess.Board(), selector_path, evaluator_path, exploration_constant)
        
        self.search_thread = None
        self.stop_event = threading.Event()

    def load_position(self, fen: str) -> None:
        """
        Loads a given position in FEN format into the engine.
        Args:
            fen (str): The FEN string representing the position.
        """
        board = chess.Board(fen)
        self.tree_search.root_state = board

    def choose_move(self, time_limit: float) -> str:
        """
        Chooses the best move for the current position within a given time limit.
        Args:
            time_limit (float): The time limit for the search in seconds.
        Returns:
            str: The chosen move in UCI format.
        """
        moves = self.tree_search.search_for_time(time_limit)
        return moves[0][0]
    
    def run(self) -> None:
        """
        This is the main loop which complies with the UCI protocol.
        """
        while True:
            command = sys.stdin.readline().strip()
            if not command:
                continue

            if command == "quit":
                print("Bye!")
                break
            elif command == "uci":
                self._handle_uci()
            elif command == "isready":
                self._handle_isready()
            elif command.startswith("position"):
                self._handle_position(command)
            elif command.startswith("go"):
                self._handle_go(command)
            elif command == "stop":
                self._handle_stop()
            elif command == "ucinewgame":
                self._handle_ucinewgame()
            else:
                print(f"Unknown UCI command: {command}")

    def _handle_uci(self) -> None:
        """
        Handles the UCI command.
        """
        print("id name BAKEN")
        print("id author BAKEN_Team")
        print("uciok", flush=True)

    def _handle_isready(self) -> None:
        """
        Handles the isready command.
        """
        print("readyok", flush=True)
    
    def _handle_ucinewgame(self) -> None:
        """
        Handles the ucinewgame command.
        """
        self.tree_search.root_state = chess.Board()

    def _handle_position(self, command: str) -> None:
        """
        Handles the position command.
        """
        args = command.split()
        moves_start_index = args.index("moves") if "moves" in args else None

        if args[1] == "startpos":
            self.tree_search.root_state = chess.Board()
        elif args[1] == "fen":
            fen = " ".join(args[2:moves_start_index] if moves_start_index else args[2:])
            self.load_position(fen)
        
        if moves_start_index:
            for move in args[moves_start_index + 1:]:
                chess_move = chess.Move.from_uci(move)
                if chess_move in self.tree_search.root_state.legal_moves:
                    self.tree_search.play_move(chess_move)
                else:
                    print(f"Error: Illegal move {move} in position command", flush=True)

    def _handle_go(self, command: str) -> None:
        """
        Handles the go command.
        """
        # Time each player has on the clock plus the increment
        wtime = -1
        btime = -1
        winc = -1
        binc = -1

        # -1 if no fixed time is given or float("inf") if infinite time is given
        fixed_time = -1

        args = command.split()

        curr_arg = 1

        while curr_arg < len(args):
            if args[curr_arg] == "wtime":
                wtime = int(args[curr_arg + 1])
                curr_arg += 2
            elif args[curr_arg] == "btime":
                btime = int(args[curr_arg + 1])
                curr_arg += 2
            elif args[curr_arg] == "winc":
                winc = int(args[curr_arg + 1])
                curr_arg += 2
            elif args[curr_arg] == "binc":
                binc = int(args[curr_arg + 1])
                curr_arg += 2
            elif args[curr_arg] == "movetime":
                fixed_time = int(args[curr_arg + 1])
                curr_arg += 2
            elif args[curr_arg] == "infinite":
                fixed_time = float("inf")
                break
            else:
                print(f"Unsupported argument in go command: {args[curr_arg]}")
                curr_arg += 1

        if self.search_thread and self.search_thread.is_alive():
            self.stop_event.set()
            self.search_thread.join()
        
        self.stop_event.clear()
        self.search_thread = threading.Thread(target=self._search_thread, args=(fixed_time, wtime, btime, winc, binc))
        self.search_thread.start()

    def _handle_stop(self) -> None:
        """
        Handles the stop command.
        """
        if self.search_thread and self.search_thread.is_alive():
            self.stop_event.set()
            self.search_thread.join(timeout=1.0)

    def _search_thread(self, fixed_time: int, wtime: int, btime: int, winc: int, binc: int) -> None:
        """
        The search thread that performs the MCTS search.
        """

        thinking_time = 0.0

        if fixed_time != -1:
            thinking_time = fixed_time
        else:
            time_remaining = wtime if self.tree_search.root_state.turn == chess.WHITE else btime
            increment = winc if self.tree_search.root_state.turn == chess.WHITE else binc

            # base time is 1/25 of the total time on the clock
            base_time = time_remaining / 25
            # use up most of the increment time in addition to the time remaining
            thinking_time = base_time + 0.7 * increment

        start_time = time.time()
        while time.time() - start_time < thinking_time/1000:
            if self.stop_event.is_set():
                break
            self.tree_search.search_iteration()
        
        best_move = self.tree_search.get_move_list()[0][0]
        self.tree_search.play_move(chess.Move.from_uci(best_move))
        print(f"bestmove {best_move}", flush=True)



def main():
    """
    Runs the engine.
    """
    selector_path = "MoveSelectorV1.7.pt"
    evaluator_path = "ConvModv2_1.pt"
    exploration_constant = 2.0

    engine = BAKEN(selector_path, evaluator_path, exploration_constant)
    print("BAKEN engine initialized.")
    engine.run()

if __name__ == "__main__":
    main()