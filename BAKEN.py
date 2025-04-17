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
            elif command == "debug":
                debug = True
            elif command == "setoption":
                option = "idk lol"
            elif command == "register":
                print("later")


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

        # In your BAKEN class:

# Add this import if MCTreeNode is used directly for reset
# from TreeSearch import MCTreeNode

    def _handle_position(self, command: str) -> None:
        """
        Handles the position command. Sets up the board and resets the search tree.
        """
        args = command.split()
        moves_start_index = -1
        try:
            moves_start_index = args.index("moves")
        except ValueError:
            pass # No "moves" keyword found

        board = None # This will hold the final board state

        try:
            if args[1] == "startpos":
                board = chess.Board()
                current_move_index = 2 # Moves start after "startpos"
            elif args[1] == "fen":
                fen_parts = []
                fen_end_index = 2
                # Collect FEN parts (up to 6 fields) until "moves" or end of command
                # Ensure we don't accidentally consume 'moves' as part of FEN
                while fen_end_index < len(args) and args[fen_end_index] != "moves" and len(fen_parts) < 6:
                    fen_parts.append(args[fen_end_index])
                    fen_end_index += 1
                fen = " ".join(fen_parts)
                board = chess.Board(fen)
                current_move_index = fen_end_index # Moves start after the FEN string (or at "moves")
            else:
                print(f"info string ERROR: Unknown position type: {args[1]}", flush=True, file=sys.stderr)
                return # Or default to startpos if preferred

            # Apply moves if the "moves" keyword was found
            if moves_start_index != -1:
                # Ensure we start reading moves *after* the "moves" keyword
                if current_move_index <= moves_start_index:
                    current_move_index = moves_start_index + 1

                for move_uci in args[current_move_index:]:
                    try:
                        move = chess.Move.from_uci(move_uci)
                        # IMPORTANT: Check legality *before* pushing
                        if move in board.legal_moves:
                            board.push(move)
                        else:
                            print(f"info string ERROR: Illegal move '{move_uci}' in position command for FEN '{board.fen()}'", flush=True, file=sys.stderr)
                            # Stop processing further moves for this invalid position command
                            return
                    except ValueError:
                        print(f"info string ERROR: Invalid move UCI '{move_uci}' in position command.", flush=True, file=sys.stderr)
                        # Stop processing further moves
                        return

            # --- Reset the MCTS tree to the final board state ---
            # This requires a method in MCTreeSearch, e.g., reset_to_position
            # If that method doesn't exist, you might need to re-initialize
            # parts of MCTreeSearch or directly set its root node.

            if board is not None:
                # Option 1: Call a dedicated reset method (Preferred)
                self.tree_search.reset_to_position(board)

                # Option 2: Directly reset the root node (if MCTreeNode is accessible)
                # self.tree_search.root = MCTreeNode(board.copy(), self.tree_search.exploration_constant)

                # Option 3: Update root_state (Only if MCTS uses it dynamically at 'go')
                # self.tree_search.root_state = board.copy() # This might not be enough on its own

        except ValueError as e:
            print(f"info string ERROR: Invalid FEN or move in position command: {e}", flush=True, file=sys.stderr)
        except IndexError:
            print(f"info string ERROR: Malformed position command: {command}", flush=True, file=sys.stderr)

    def _handle_go(self, command: str) -> None:
        # Stop previous search if any
        if self.search_thread and self.search_thread.is_alive():
            print("info string Stopping previous search before starting new one.", flush=True, file=sys.stderr)
            self.stop_event.set()
            self.search_thread.join(timeout=1.0) # Wait a bit for it to stop

        # Parse parameters
        wtime = btime = winc = binc = -1
        movetime = -1
        infinite = False
        # Add depth, nodes, mate limits if needed
        # depth_limit = nodes_limit = mate_limit = -1

        args = command.split()
        curr_arg = 1
        while curr_arg < len(args):
            arg_name = args[curr_arg]
            if arg_name in ["wtime", "btime", "winc", "binc", "movetime"]: # Add "depth", "nodes", "mate" if supported
                if curr_arg + 1 < len(args):
                    try:
                        value = int(args[curr_arg + 1])
                        if arg_name == "wtime": wtime = value
                        elif arg_name == "btime": btime = value
                        elif arg_name == "winc": winc = value
                        elif arg_name == "binc": binc = value
                        elif arg_name == "movetime": movetime = value
                        # elif arg_name == "depth": depth_limit = value
                        # ... etc
                        curr_arg += 2
                    except ValueError:
                        print(f"info string Warning: Invalid value for {arg_name}: {args[curr_arg + 1]}", flush=True, file=sys.stderr)
                        curr_arg += 2 # Skip invalid value pair
                else:
                    print(f"info string Warning: Missing value for {arg_name}", flush=True, file=sys.stderr)
                    curr_arg += 1 # Skip arg name
            elif arg_name == "infinite":
                infinite = True
                curr_arg += 1
            else:
                print(f"info string Warning: Unsupported 'go' parameter: {arg_name}", flush=True, file=sys.stderr)
                curr_arg += 1 # Skip unknown parameter

        # Start search thread
        self.stop_event.clear()
        self.search_thread = threading.Thread(
            target=self._search_thread,
            args=(movetime, wtime, btime, winc, binc, infinite) # Pass infinite explicitly
            # Pass depth_limit etc. if adding them
        )
        self.search_thread.start()

    def _handle_stop(self) -> None:
        if self.search_thread and self.search_thread.is_alive():
            # print("info string Received stop command.", flush=True, file=sys.stderr)
            self.stop_event.set()
            self.search_thread.join(timeout=1.0)

    def _search_thread(self, movetime: int, wtime: int, btime: int, winc: int, binc: int, infinite: bool) -> None:
        """
        The search thread that performs the MCTS search.
        Times from UCI (movetime, wtime, btime, winc, binc) are in MILLISECONDS.
        """
        thinking_time = 0.0 # This will be in MILLISECONDS

        # --- Calculate thinking time ---
        if infinite:
            thinking_time = float("inf")
        elif movetime != -1:
            thinking_time = movetime
        elif wtime != -1 and btime != -1: # Standard time controls
            time_remaining = wtime if self.tree_search.root.state.turn == chess.WHITE else btime
            increment = winc if self.tree_search.root.state.turn == chess.WHITE else binc
            if increment == -1: increment = 0 # Default increment is 0 if not specified

            # Heuristic: Aim for e.g., 1/25th of remaining time + 70% of increment
            # Adjust divisor (25) and increment factor (0.7) as needed
            base_time = time_remaining / 25
            thinking_time = base_time + 0.7 * increment

            # Safety buffer: don't use more than e.g., 50% of remaining time in one go
            # and ensure minimum thinking time (e.g. 50ms)
            max_allowed = time_remaining * 0.5
            thinking_time = min(thinking_time, max_allowed)
            thinking_time = max(thinking_time, 50) # Ensure at least 50ms

        else:
            # No time control given (e.g., just "go") - default to a fixed time
            print("info string No time control specified, searching for 5 seconds.", flush=True)
            thinking_time = 5000 # 5000 ms = 5 seconds

        # --- Search Loop ---
        start_time = time.time() # time.time() is in SECONDS
        last_info_time = start_time
        nodes_searched_total = 0

        # Convert thinking_time (ms) to seconds for comparison with time.time()
        thinking_time_seconds = float("inf") if thinking_time == float("inf") else thinking_time / 1000.0

        while True:
            current_time = time.time()
            elapsed_time_seconds = current_time - start_time

            if self.stop_event.is_set():
                break

            if thinking_time_seconds != float("inf") and elapsed_time_seconds >= thinking_time_seconds:
                    break

            nodes_this_iteration = self.tree_search.search_iteration()
            if nodes_this_iteration: # Add nodes if search_iteration returns them
                nodes_searched_total += nodes_this_iteration
            else:
                    nodes_searched_total += 1 # Basic increment if nodes aren't returned


            # --- Send Info Periodically (e.g., every second) ---
            if current_time - last_info_time >= 1.0 or elapsed_time_seconds >= thinking_time_seconds:
                last_info_time = current_time
                # Get data from self.tree_search (replace with actual calls)
                pv_list = self.tree_search.get_principal_variation() # -> ['e2e4', 'c7c5']
                score_cp = self.tree_search.get_root_score_cp() # -> centipawns or None
                depth = self.tree_search.get_effective_depth() # -> integer or None
                #nodes = self.tree_search.get_total_nodes() # -> Use nodes_searched_total

                pv_string = " ".join(pv_list)
                time_ms = int(elapsed_time_seconds * 1000)
                nps = int(nodes_searched_total / elapsed_time_seconds) if elapsed_time_seconds > 0 else 0

                # Construct the info string
                info_parts = ["info"]
                if depth is not None: info_parts.append(f"depth {depth}")
                if score_cp is not None: info_parts.append(f"score cp {score_cp}")
                info_parts.append(f"nodes {nodes_searched_total}")
                if nps > 0 : info_parts.append(f"nps {nps}")
                info_parts.append(f"time {time_ms}")
                if pv_string: info_parts.append(f"pv {pv_string}")

                print(" ".join(info_parts), flush=True) # Send info string

            # Optional: Yield control briefly to allow other threads
            # time.sleep(0.001) # Can help responsiveness but slows down search

        if self.stop_event.is_set() or (thinking_time_seconds != float("inf")):
            final_move_list = self.tree_search.get_move_list()
            print(final_move_list)
            best_move = final_move_list[0][0] # Assuming format [(move_uci, score/visits), ...]
            print(f"bestmove {best_move}", flush=True)

def main():
    """
    Runs the engine.
    """
    selector_path = "MoveSelectorV1.7.pt"
    evaluator_path = "ConvModv2_2.pt"
    exploration_constant = 2.0

    engine = BAKEN(selector_path, evaluator_path, exploration_constant)
    print("BAKEN engine initialized.")
    engine.run()

if __name__ == "__main__":
    main()
