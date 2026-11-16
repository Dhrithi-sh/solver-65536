# P1 Engine — API Reference

**Snapshot:** commit `d8379a0`. Signatures below were read from the source with `inspect.signature`. Edge-case
behaviour marked "(checked)" was run on both backends (`cpp` and `ENGINE_PURE_PYTHON=1`) at this commit and
behaved the same on both.

**Origin** says where a name was introduced, not who owns it (P1 owns all of `engine/`):
*original* = in the skeleton commit `0e5e863`, part of the shared interface in `AGENTS.md`, changes need the
group's agreement; *P1* = added by P1 (Batches 1–3c), without changing any original signature.

**Imports.** `from engine import Game, Board, MOVES, MOVE_NAMES, UP, RIGHT, DOWN, LEFT` (that is all
`engine/__init__.py` exports). Everything else: `from engine.game import ...` or `from engine.batch import ...`.

Type alias: `Board = List[List[int]]` (4x4 tile values, 0 = empty).

---

## 1. `engine.game.Game`

### `Game(seed=0, board=None)` — original

```python
def __init__(self, seed: int = 0, board: Optional[Sequence[Sequence[int]]] = None)
```
Calls `self.reset(seed, board)`.
* `board=None`: empty board, then two spawns from `random.Random(seed)`.
* `board` given: the board is copied (`copy_board`), **nothing spawns**, RNG still seeded with `seed`.
* Edge cases (checked): `Game(-5)` and `Game(5)` start identically (CPython seeds with `abs(seed)`). A board of the
  wrong shape is **not** validated here and fails later with `IndexError`; use `Game.from_board` for a clean
  `ValueError`. An all-zero `board` gives a game that is immediately `done` (nothing can move and nothing spawns).
* Tests: `test_new_game_has_two_tiles`, `test_seeded_games_repeat`, `test_reset_with_board_spawns_nothing_and_copies_input`.

```python
from engine import Game
g = Game(seed=42)
late = Game(7, board=[[32768, 32768, 0, 0], [0]*4, [0]*4, [0]*4])
```

### `game.reset(seed=None, board=None) -> Board` — original

```python
def reset(self, seed: Optional[int] = None, board: Optional[Sequence[Sequence[int]]] = None) -> Board
```
Starts a new game: `seed=None` keeps `self.seed`; creates `self.rng = random.Random(self.seed)`, sets `score=0`,
`moves=[]`, the board as in the constructor, and `done = not self.legal_moves()`. Returns a copy of the board.
* Tests: `test_seeded_games_repeat_and_reset_restarts`.

### `game.step(move) -> (board, reward, done)` — original

```python
def step(self, move: int) -> Tuple[Board, int, bool]
```
* `move not in MOVES` -> `ValueError("invalid move ...")` (checked first, even on a finished game).
* Game already over -> returns `(copy, 0, True)`, nothing changes.
* Illegal move (board unchanged by the slide) -> `(copy, 0, done)`: no score, no spawn, **no RNG use**, not
  recorded.
* Legal move -> board slid, `score += gained`, `moves.append(move)`, one spawn, `done` recomputed. Returns
  `(copy of board, gained, done)`; `gained` is the merge score of this move.
* Edge case (checked, inherited from the skeleton): any value *equal* to 0..3 passes the check, so `step(True)` and
  `step(1.0)` play and are recorded as `True` / `1.0` in `moves`. Always pass plain `int`s.
* Tests: `test_step_spawns_one_tile_and_scores`, `test_illegal_step_is_noop`, `test_invalid_move_raises`,
  `test_illegal_move_is_noop`, `test_invalid_move_value_raises`, `test_done_when_no_moves`,
  `test_game_over_on_full_board_without_merges`, `test_full_board_with_merge_is_not_over`, `test_random_games_terminate`.

```python
board, reward, done = g.step(3)   # LEFT
```

### `game.legal_moves() -> List[int]` — original

```python
def legal_moves(self) -> List[int]
```
Moves that change the board, ascending (up, right, down, left). Uses `_cengine.legal_mask` when available and
supported, else `Game.is_legal` for each move. Empty list when the game is over.
* Tests: `test_illegal_moves_detected`, `test_legal_mask_matches_python`.

### `game.max_tile() -> int` — original

```python
def max_tile(self) -> int
```
Largest cell value. Tests: `test_max_tile`, `test_32768_plus_32768_makes_65536`.

### `game.empty_cells() -> List[Tuple[int, int]]` — original

```python
def empty_cells(self) -> List[Tuple[int, int]]
```
`(row, col)` of zeros in row-major order. This is the list `_spawn` indexes with `randrange`, so its order is part
of the RNG contract. Tests: `test_empty_cells_row_major_and_matches_game_method`.

### `Game.slide(board, move) -> (new_board, gained)` — original, static

```python
@staticmethod
def slide(board: Sequence[Sequence[int]], move: int) -> Tuple[Board, int]
```
Pure: returns the board after sliding in `move` and the merge score; **no spawn**, input not modified (works on
lists or tuples). Uses C++ when loaded and `type(move) is int` and the board is supported; otherwise the Python
row table; otherwise (cells that are not 0 / a power of two in 2..2^31) the original `_merge_line`.
* Move outside 0..3 -> `ValueError` (checked; numpy integer moves work via the Python path).
* Merge rules: each tile merges once per move (`[2,2,2,2]` LEFT -> `[4,4,0,0]`, gained 8); merges nearest the
  move direction first (`[2,2,2,0]` LEFT -> `[4,2,0,0]`, RIGHT -> `[0,0,2,4]`); no chain merges (`[4,2,2,0]` LEFT ->
  `[4,4,0,0]`). Score = sum of the created tiles.
* Large tiles (checked): `32768+32768 -> 65536` (gained 65,536); a pair of 2^19 tiles (beyond C++) still merges
  correctly via Python (gained 2^20). Odd values such as `[3,3,...]` merge to 6 via the original fallback.
* Tests: `test_left`, `test_right`, `test_up`, `test_down`, `test_no_double_merge_2222`, `test_no_chain_merge_4_2_2`,
  `test_merge_priority_toward_move_direction`, `test_65536_merge`, `test_slide_is_pure`,
  `test_all_four_directions_move_and_merge`, `test_score_is_sum_of_created_tiles`, `test_65536_in_every_direction_and_beyond`,
  `test_row_table_matches_reference_for_every_row_up_to_65536`, `test_non_power_of_two_boards_use_fallback_unchanged`,
  `test_slide_matches_python_all_directions_and_scores`, `test_every_row_up_to_exponent_18_matches_python`,
  `test_tuple_boards_are_accepted_and_match`, `test_results_are_fresh_lists_and_input_is_untouched`.

```python
new_board, gained = Game.slide(g.board, 0)   # look ahead UP without touching g
```

### `Game.is_legal(board, move) -> bool` — original, static

```python
@staticmethod
def is_legal(board: Sequence[Sequence[int]], move: int) -> bool
```
`True` if `Game.slide(board, move)` changes the board. Tests: used as the reference in `test_legal_mask_matches_python`.

### `Game.from_board(board, seed=0) -> Game` — P1 (Batch 2), classmethod

```python
@classmethod
def from_board(cls, board: Sequence[Sequence[int]], seed: int = 0) -> 'Game'
```
Same as `Game(seed, board)` plus a 4x4 shape check (`ValueError: board must be 4x4`). Board copied, nothing
spawns, spawns after each move reproducible from `seed`. A dead board gives `done=True`, `legal_moves()==[]` (checked).
For P3 (search from any position) and P8 (late-game benchmark).
* Tests: `test_from_board_uses_given_board_and_spawns_nothing`, `test_from_board_copies_input`,
  `test_from_board_default_seed_and_equivalence_with_constructor`, `test_from_board_is_deterministic_per_seed`,
  `test_from_board_32768_late_game_converts_to_65536`, `test_from_board_rejects_wrong_shape`,
  `test_from_board_full_dead_board_is_done`, `test_from_board_games_identical_across_backends`.

```python
g = Game.from_board(saved_32768_board, seed=100000)
```

### Attributes — original

| Attribute | Type | Meaning |
|---|---|---|
| `board` | `Board` | Current 4x4 tile values. Read it; do not mutate it (agents must not mutate the game) |
| `score` | `int` | Sum of merge rewards so far |
| `moves` | `List[int]` | Legal moves played, in order (replay with `Game(seed)` + `step`) |
| `seed` | `int` | Seed of this game |
| `done` | `bool` | No legal move left |
| `rng` | `random.Random` | The spawn RNG. Internal; never draw from it (it would change all later spawns) |

---

## 2. Module functions in `engine.game` (P1)

### `afterstate(board, move) -> Board` — P1 (Batch 2)

```python
def afterstate(board: Sequence[Sequence[int]], move: int) -> Board
```
`Game.slide(board, move)[0]`: the board after the slide, **before** the spawn. Pure. An illegal move returns an
**equal copy** (not the same object; checked), so use `Game.is_legal` to tell. Move outside 0..3 -> `ValueError`.
For P5 (TD learning on afterstates).
* Tests: `test_afterstate_matches_slide_and_all_directions`, `test_afterstate_known_values_and_no_spawn`,
  `test_afterstate_is_pure`, `test_afterstate_illegal_move_returns_equal_copy`, `test_afterstate_invalid_move_raises`,
  `test_afterstate_65536`, `test_batch2_helpers_identical_across_backends`.

```python
from engine.game import afterstate, Game
for m in g.legal_moves():
    a = afterstate(g.board, m)
```

### `empty_cells(board) -> List[Tuple[int, int]]` — P1 (Batch 2)

```python
def empty_cells(board: Sequence[Sequence[int]]) -> List[Tuple[int, int]]
```
`(row, col)` of zeros, row-major; same order as `Game.empty_cells()` but works on any board. Full board -> `[]`.
* Tests: `test_empty_cells_row_major_and_matches_game_method`, `test_empty_cells_full_and_empty_boards`,
  `test_empty_cells_matches_game_on_random_boards`.

### `spawn_outcomes(board) -> List[Tuple[Board, float]]` — P1 (Batch 2)

```python
def spawn_outcomes(board: Sequence[Sequence[int]]) -> List[Tuple[Board, float]]
```
Every board the next spawn can produce with its probability. Order: empty cells row-major; for each, the "2" board
(p = 0.9/n) then the "4" board (p = 0.1/n). Probabilities sum to 1 (up to float rounding: one empty cell gives
`0.9` and `0.09999999999999998`, checked). Full board -> `[]`. Fresh boards; input not modified. For expectimax
chance nodes (P3, P7).
* Tests: `test_spawn_outcomes_count_probabilities_and_values`, `test_spawn_outcomes_each_board_differs_by_one_new_tile`,
  `test_spawn_outcomes_full_board_is_empty_list_and_input_untouched`, `test_spawn_outcomes_agree_with_engine_spawn_frequencies`.

```python
value = sum(p * evaluate(child) for child, p in spawn_outcomes(afterstate_board))
```

### `exponents(board)` -> `numpy.ndarray` — P1 (Batch 2)

```python
def exponents(board: Sequence[Sequence[int]])
```
4x4 NumPy array, **dtype `int32`**, of log2 values (0 empty, 2 -> 1, 65,536 -> 16). Imports numpy only when called
(`ImportError` with a hint if missing). Accepts numpy integer cells. Wrong shape -> `ValueError`; a cell that is not
0 or a power of two from 2 to 2^31 -> `ValueError` (checked: 2^31 -> 31, 2^32 -> `ValueError`, 3 -> `ValueError`).
`int32` so index arithmetic like `e0 + 32*e1 + 1024*e2` cannot wrap (int8 would). Returns a new array. For P5, P6.
* Tests: `test_exponents_values_shape_dtype`, `test_exponents_roundtrip_and_131072`,
  `test_exponents_no_overflow_when_building_indexes`, `test_exponents_returns_copy_independent_of_board`,
  `test_exponents_accepts_numpy_ints_and_rejects_bad_cells`.

```python
from engine.game import exponents
x = exponents(g.board)          # array([[0, 1, ...], ...], dtype=int32)
```

### `play_batch(agent_fn, seeds, processes=1, *, deterministic=False, start_method=None) -> List[dict]` — P1 (Batch 2; keywords added in 3c)

```python
def play_batch(agent_fn, seeds: Sequence[int], processes: int = 1, *,
               deterministic: bool = False, start_method: Optional[str] = None) -> List[dict]
```
One full game per seed, `Game(seed)` from a fresh start, moves from `agent_fn(game) -> move` (call it like
`Agent.choose_move`; pass e.g. `agent.choose_move`).
* Returns, in the order of `seeds`: `{"seed", "score", "max_tile", "moves" (list, replayable with Game(seed)), "board" (final)}`.
  (No `n_moves` key; use `len(r["moves"])`.)
* An illegal move -> `ValueError` (never loops forever). `agent_fn` must not mutate the game.
* `processes <= 1` (or ≤ 1 seed): runs in this process. `processes > 1`: `multiprocessing` pool; `agent_fn` must be
  picklable (module-level function or bound method of a picklable object, not a lambda).
* `deterministic=True`: each game gets `copy.deepcopy(agent_fn)`; if `agent_fn` is a bound method of an object with
  `reset`, `reset(seed)` is called first (as `eval/run.py` does). Then a game depends only on `(agent, seed)` and
  results are identical for any `processes`; the caller's agent is not mutated. Without it, stateful agents give
  results that depend on process count and pool chunking.
* `start_method`: `"fork"`, `"spawn"` or `"forkserver"`; default is the platform's (spawn on Windows/macOS, where
  `agent_fn` must be importable and the calling script needs `if __name__ == "__main__":`).
* No `board` parameter: games always start from `Game(seed)`.
* Speed: limited by the agent and by `Game.step()`; **it does not reach 1M moves/s** and its throughput has not
  been measured.
* Tests: `test_play_batch_basic_shape_and_order`, `test_play_batch_matches_manual_game`,
  `test_play_batch_is_deterministic_and_replayable`, `test_play_batch_parallel_equals_sequential`,
  `test_play_batch_empty_seed_list`, `test_play_batch_illegal_move_raises_instead_of_hanging`,
  `test_play_batch_works_with_bound_method`, `test_play_batch_default_behaviour_unchanged`,
  `test_play_batch_deterministic_mode_is_independent_of_process_count`,
  `test_play_batch_deterministic_mode_does_not_mutate_callers_agent`,
  `test_play_batch_stateful_agents_differ_without_deterministic_flag_documented`,
  `test_play_batch_works_with_spawn_start_method`, `test_play_batch_still_rejects_illegal_moves_in_parallel`.

```python
from engine.game import play_batch
from agents.baselines import GreedyAgent

if __name__ == "__main__":
    results = play_batch(GreedyAgent().choose_move, range(100), processes=4, deterministic=True)
```

### `backend() -> str` — P1 (Batch 3b)

```python
def backend() -> str
```
`"cpp"` if `engine._cengine` is in use, else `"python"`. Decided once at import (`ENGINE_PURE_PYTHON` set to anything
other than `""`/`"0"` forces `"python"`).
* Tests: `test_backend_reports_active_engine`, `test_env_var_forces_pure_python`,
  `test_env_var_zero_or_empty_does_not_force_python`, `test_missing_extension_import_falls_back`,
  `test_extension_is_built_when_required`.

```bash
python -c "from engine import game; print(game.backend())"
```

---

## 3. `engine.batch` (P1, Batch 3c)

### `play_random_batch(seeds, board=None, threads=0, record_moves=True) -> List[dict]`

```python
def play_random_batch(seeds: Sequence[int], board: Optional[Sequence[Sequence[int]]] = None,
                      threads: int = 0, record_moves: bool = True) -> List[dict]
```
One **random-policy** game per seed (the policy is defined under `RandomLegalPolicy`). In C++ across threads with the
GIL released when the extension is built; otherwise pure Python, with identical results.
* Returns, in input order: `{"seed", "score", "max_tile", "n_moves", "moves" (list of int, or None if record_moves=False), "board"}`.
* `board`: every game starts from this position, no initial spawn (late-game use). Shape is checked
  (`ValueError`); the board is not modified (`test_start_board_is_not_modified`). Boards C++ does not support
  (odd values, tiles > 2^18, numpy arrays, …) are played in Python. An all-zero board ends at once with 0 moves (checked).
* `threads`: C++ worker count, `0` = all hardware threads. Results never depend on it. Ignored on the Python path.
* `record_moves=False`: `moves` is `None` (faster; used by the benchmark).
* Seeds: any exact `int` works (negative, and ≥ 2^64 via Python). **numpy integer seeds and float seeds raise
  `TypeError`** (checked; `engine/README.md` wrongly says Python plays them). If any one seed is not an exact `int`
  below 2^64 in absolute value, the **whole** batch runs in Python (slow).
* A game started from a supplied board that grows beyond the C++ range (e.g. sixteen 65,536 tiles chaining to 2^19)
  is replayed by Python ("bailed"), so the answer is still exact.
* `seeds=[]` -> `[]`.
* Speed: this is the only API that meets the 1M moves/sec target (see `P1_TESTING_AND_BENCHMARKS.md`).
* Tests (`tests/test_engine_batch.py`): `test_cpp_and_python_agree_on_10000_seeded_games`,
  `test_batch_matches_pure_python_engine_in_process`, `test_cpp_games_replay_on_the_real_engine`,
  `test_results_do_not_depend_on_thread_count`, `test_works_without_extension_and_is_repeatable`,
  `test_from_board_start_matches_python_and_can_make_65536`, `test_start_board_is_not_modified`,
  `test_huge_start_tiles_leave_cpp_range_and_still_match_python`, `test_unsupported_inputs_fall_back_to_python`,
  `test_bad_board_shape_raises`, `test_record_moves_false_and_empty_input`, `test_concurrent_python_threads_get_correct_results`.

```python
from engine.batch import play_random_batch
res = play_random_batch(range(10_000), threads=0, record_moves=False)
late = play_random_batch(range(500), board=saved_board)
```

### `RandomLegalPolicy()`

```python
class RandomLegalPolicy:
    def __init__(self) -> None
    def __call__(self, game: Game) -> int
```
The same random policy as an `agent_fn` for `play_batch`. On the first move of each game (`game.moves` empty) it
re-seeds `random.Random(policy_seed(game.seed))`; then returns `legal[rng.randrange(len(legal))]` with `legal =
game.legal_moves()`. Picklable (checked), needs no `reset`. The 10,000-game test's reference side is
`play_batch(RandomLegalPolicy(), range(n), processes=procs, deterministic=True)` in pure Python.
* Tests: `test_python_policy_via_play_batch_equals_batch_python_path`.

### `policy_seed(seed) -> int` and `POLICY_SEED_MASK`

```python
POLICY_SEED_MASK = 0x2545F4914F6CDD1D
def policy_seed(seed: int) -> int      # abs(seed) ^ POLICY_SEED_MASK
```
Seed of the move-choosing RNG. `policy_seed(-3) == policy_seed(3)` (checked). Must equal `p1::kPolicySeedMask` in
`engine/_core/batch.hpp`; `test_policy_mask_is_shared_between_c_and_python` checks it.

---

## 4. Constants and small helpers in `engine.game`

| Name | Value / signature | Origin |
|---|---|---|
| `UP, RIGHT, DOWN, LEFT` | `0, 1, 2, 3` | original |
| `MOVES` | `(0, 1, 2, 3)` | original (`test_moves_constant`) |
| `MOVE_NAMES` | `{0: "up", 1: "right", 2: "down", 3: "left"}` | original |
| `SIZE` | `4` | original |
| `PROB_TWO` | `0.9` | original |
| `Board` | `List[List[int]]` | original |
| `empty_board() -> Board` | 4x4 zeros | original |
| `copy_board(board: Sequence[Sequence[int]]) -> Board` | list-of-lists copy | original |

## 5. Not part of the shared API

Use only if you are working on the engine itself; these may change.

* `engine._cengine` (compiled): `slide`, `legal_mask`, `play_random`, `PyRandom`, `MAX_SUPPORTED_EXP` (18),
  `POLICY_SEED_MASK`. Each returns `None` for unsupported input.
* Batch 1 internals in `engine.game`: `pack_row(exps) -> int`, `unpack_row(key) -> List[int]`,
  `row_table_entry(key) -> Tuple[int, int]`, `CELL_BITS` (5), `CELL_MASK` (31), `MAX_EXP` (31).
* Anything starting with `_` (`_merge_line`, `_get_lines`, `_set_lines` are kept as the test oracle).
* `engine.bench`: `run(seconds, seed=0)`, `run_batch(seconds, seed=0, threads=0, chunk=2000)`, `main(argv=None)`;
  use it as the CLI `python -m engine.bench`.
* Environment variables: `ENGINE_PURE_PYTHON` (force Python), `ENGINE_REQUIRE_CPP` (tests only),
  `ENGINE_PARITY_GAMES` (tests only).
