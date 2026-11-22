
import math
import random

from agents.base import Agent
from agents.heuristics import HeuristicEvaluator
from engine.game import Game, MOVES, PROB_TWO, SIZE, copy_board


class MCTSNode:
    def __init__(
        self,
        board,
        node_type,
        parent=None,
        move=None,
        probability=1.0,
    ):
        self.board = copy_board(board)
        self.node_type = node_type
        self.parent = parent
        self.move = move
        self.probability = probability
        self.visits = 0
        self.total_reward = 0.0
        self.children = []
        self.lookup = {}


class MCTSAgent(Agent):
    def __init__(
        self,
        iterations=100,
        exploration_constant=0.5,
        evaluator=None,
        rollout_depth=20,
    ):
        if (
            not isinstance(iterations, int)
            or isinstance(iterations, bool)
            or iterations <= 0
        ):
            raise ValueError("iterations must be a positive integer")

        if (
            not isinstance(rollout_depth, int)
            or isinstance(rollout_depth, bool)
            or rollout_depth <= 0
        ):
            raise ValueError("rollout_depth must be a positive integer")

        if (
            not isinstance(exploration_constant, (int, float))
            or isinstance(exploration_constant, bool)
            or not math.isfinite(exploration_constant)
            or exploration_constant <= 0
        ):
            raise ValueError("exploration_constant must be positive and finite")

        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.rollout_depth = rollout_depth
        self.evaluator = evaluator or HeuristicEvaluator.from_file("tuned")
        self.rng = random.Random()

    def reset(self, seed=None):
        self.rng = random.Random(seed)

    @staticmethod
    def board_key(board):
        return tuple(tuple(row) for row in board)

    def expand_player_node(self, node):
        """Create one chance node for each legal player move."""
        if node.children:
            return

        for move in MOVES:
            if not Game.is_legal(node.board, move):
                continue

            next_board, _ = Game.slide(node.board, move)

            child = MCTSNode(
                board=next_board,
                node_type="chance",
                parent=node,
                move=move,
            )

            node.children.append(child)

        node.lookup = {
            child.move: child for child in node.children
        }

    def sample_spawn(self, board):
        """Sample one tile spawn using the game's probabilities."""
        next_board = copy_board(board)
        empty_cells = [
            (r, c)
            for r in range(SIZE)
            for c in range(SIZE)
            if next_board[r][c] == 0
        ]

        if not empty_cells:
            return next_board, 1.0

        r, c = self.rng.choice(empty_cells)

        if self.rng.random() < PROB_TWO:
            tile = 2
            tile_probability = PROB_TWO
        else:
            tile = 4
            tile_probability = 1.0 - PROB_TWO

        next_board[r][c] = tile
        probability = tile_probability / len(empty_cells)

        return next_board, probability

    def expand_chance_node(self, node):
        """Enumerate every possible spawn outcome and its probability."""
        if node.children:
            return

        empty_cells = [
            (r, c)
            for r in range(SIZE)
            for c in range(SIZE)
            if node.board[r][c] == 0
        ]

        if not empty_cells:
            child = MCTSNode(
                board=node.board,
                node_type="player",
                parent=node,
                probability=1.0,
            )
            node.children.append(child)
            node.lookup[self.board_key(child.board)] = child
            return

        cell_probability = 1.0 / len(empty_cells)

        for r, c in empty_cells:
            for tile, tile_probability in (
                (2, PROB_TWO),
                (4, 1.0 - PROB_TWO),
            ):
                next_board = copy_board(node.board)
                next_board[r][c] = tile

                child = MCTSNode(
                    board=next_board,
                    node_type="player",
                    parent=node,
                    probability=cell_probability * tile_probability,
                )

                key = self.board_key(next_board)
                node.children.append(child)
                node.lookup[key] = child

    def select_chance_outcome(self, node):
        """Sample an actual spawn and reuse its tree node if already seen."""
        next_board, probability = self.sample_spawn(node.board)
        key = self.board_key(next_board)

        if key in node.lookup:
            return node.lookup[key]

        child = MCTSNode(
            board=next_board,
            node_type="player",
            parent=node,
            probability=probability,
        )

        node.children.append(child)
        node.lookup[key] = child

        return child

    def select_child_uct(self, node):
        """Select a player move using the UCT formula."""
        unvisited = [
            child for child in node.children
            if child.visits == 0
        ]

        if unvisited:
            return self.rng.choice(unvisited)

        log_parent_visits = math.log(max(1, node.visits))

        def uct_score(child):
            average_reward = child.total_reward / child.visits
            exploration = self.exploration_constant * math.sqrt(
                log_parent_visits / child.visits
            )
            return average_reward + exploration

        return max(node.children, key=uct_score)

    def evaluate_leaf(self, board):
        """Return a bounded heuristic reward for the current board."""
        if not any(Game.is_legal(board, move) for move in MOVES):
            return -1.0

        score = self.evaluator.evaluate(board)
        return math.tanh(score / 10.0)

    @staticmethod
    def backpropagate(node, reward):
        """Update visit counts and rewards all the way to the root."""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def choose_move(self, game):
        legal_moves = game.legal_moves()

        if not legal_moves:
            return None

        root = MCTSNode(game.board, "player")

        for _ in range(self.iterations):
            node = root

            while True:
                if node.node_type == "player":
                    if not any(
                        Game.is_legal(node.board, move)
                        for move in MOVES
                    ):
                        self.backpropagate(node, -1.0)
                        break

                    if not node.children:
                        self.expand_player_node(node)

                    if not node.children:
                        self.backpropagate(node, -1.0)
                        break

                    node = self.select_child_uct(node)

                else:
                    node = self.select_chance_outcome(node)

                    if node.visits == 0:
                        reward = self.evaluate_leaf(node.board)
                        self.backpropagate(node, reward)
                        break

        if not root.children:
            return self.rng.choice(legal_moves)

        # Prefer moves explored more often, then use average reward
        # to break ties.
        best_child = max(
            root.children,
            key=lambda child: (
                child.visits,
                child.total_reward / max(1, child.visits),
            ),
        )

        return best_child.move
