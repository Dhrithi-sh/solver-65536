"""Self-play TD(0) training of the n-tuple network, on afterstates.

    python -m agents.ntuple.train --games 10000 --alpha 0.1 --seed 10000000 \\
        --out weights/ntuple/run1 --checkpoint-every 10000 --log-every 1000

One game (all inside numba, see ``play_and_learn``):

1. Play: at each step try every legal move, compute its afterstate (board
   after the slide, before the spawn) and reward, and pick the move with the
   highest ``reward + V(afterstate)``. Record (afterstate, reward), then spawn.
2. Learn, once the game is over, walking the recorded afterstates BACKWARDS:
       last afterstate:   target = 0                     (nothing follows it)
       afterstate t:      target = reward[t+1] + V(afterstate[t+1])
       V(afterstate t) += alpha * (target - V(afterstate t))
   The step goes through ``update_exps``, which spreads it over the 32 table
   lookups (alpha / 32 per weight). Going backwards means each target already
   uses the freshly updated value of the next afterstate, so a game's final
   outcome flows all the way back to its first move in a single pass.

Improvements (flags, so each can be compared with the plain run):
    --stages 2 --stage-tile 16384   multi-stage TD (Yeh et al. 2017): a separate
        network for boards holding >= 16,384; stage 2 is warm-started from
        stage 1 the first time any game reaches it.
    --init-value V                  optimistic initialisation (Guei et al. 2022):
        every board starts valued at V, so unvisited positions look attractive
        and get explored until real experience pulls them down.

Outputs in ``--out``:
    train_log.csv   one row per ``--log-every`` games (stats over that window)
    latest.ntw      checkpoint: weights + RNG state + counters (resumable)
    ckpt_<N>.ntw    numbered copies, only with --keep-checkpoints (268 MB each)

The training log is for learning curves only. Benchmark numbers for the report
must come from ``python -m eval.run`` (CONTRIBUTING.md).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from agents.ntuple._jit import njit
from agents.ntuple.board import is_game_over, legal_moves, make_rng, next_u32, slide, spawn_random
from agents.ntuple.network import (
    StagedNetwork,
    ensure_stage_started,
    evaluate_exps,
    evaluate_staged,
    stage_of,
    update_staged,
)

MILESTONES = (2048, 4096, 8192, 16384, 32768)
MILESTONE_EXPS = tuple(t.bit_length() - 1 for t in MILESTONES)
DEFAULT_SEED = 10_000_000  # training seeds stay clear of the shared eval seeds

# Every move spawns a tile worth >= 2 and merges never lose value, so the sum
# of tiles grows by >= 2 per move. A 4x4 board's sum is always < 2**18, hence
# no game can last 2**17 moves; this buffer can never overflow.
MAX_MOVES = 1 << 17


# -------------------------------------------------------------------- kernels
@njit(cache=True)
def choose_move(weights3: np.ndarray, stage_exps: np.ndarray, started: np.ndarray, board: np.ndarray):
    """1-ply greedy: argmax over legal moves of reward + V(afterstate).

    Each afterstate is scored by its own stage's network. If an afterstate is
    the first ever to reach a new stage, that stage is warm-started from the
    previous one (training only; the agent passes ``started`` = all ones).
    Returns (move, afterstate, reward); move = -1 if no move is legal.
    Ties go to the lowest move number, so play is deterministic.
    """
    best_move = -1
    best_value = 0.0
    best_after = board
    best_reward = 0
    for m in range(4):
        after, reward, changed = slide(board, m)
        if not changed:
            continue
        s = stage_of(after, stage_exps)
        if started[s] == 0:
            ensure_stage_started(weights3, started, s)
        value = reward + evaluate_exps(weights3[s], after)
        if best_move == -1 or value > best_value:
            best_move = m
            best_value = value
            best_after = after
            best_reward = reward
    return best_move, best_after, best_reward


@njit(cache=True)
def learn_backward(
    weights3: np.ndarray, stage_exps: np.ndarray, afterstates: np.ndarray, rewards: np.ndarray, n: int, alpha: float
) -> None:
    """Backward TD(0) pass over the n recorded (afterstate, reward) pairs.

    rewards[t] is the reward of the move that produced afterstates[t]; the
    target for afterstate t uses the reward of the NEXT move, rewards[t+1].
    Every value is read and written in that afterstate's own stage, so at a
    stage boundary the last stage-1 afterstate learns from V_stage2(next).
    """
    if n == 0:
        return
    # The final afterstate was followed by a spawn that ended the game.
    last = afterstates[n - 1]
    update_staged(weights3, stage_exps, last, alpha * (0.0 - evaluate_staged(weights3, stage_exps, last)))
    for t in range(n - 2, -1, -1):
        target = rewards[t + 1] + evaluate_staged(weights3, stage_exps, afterstates[t + 1])
        s = afterstates[t]
        update_staged(weights3, stage_exps, s, alpha * (target - evaluate_staged(weights3, stage_exps, s)))


@njit(cache=True)
def play_and_learn(weights3, stage_exps, started, rng_state, alpha, afterstates, rewards):
    """Play one self-play game, then learn from it. Returns (score, max_exp, moves).

    With alpha = 0 nothing is learned (useful as a baseline).
    """
    board = np.zeros(16, dtype=np.uint8)
    spawn_random(board, rng_state)
    spawn_random(board, rng_state)
    return _play_and_learn_from(board, weights3, stage_exps, started, rng_state, alpha, afterstates, rewards)


@njit(cache=True)
def pick_start(pos_exps, game_offsets, rng_state):
    """Uniform source game first, then a uniform position within it.

    game_offsets[g]:game_offsets[g+1] are game g's rows in pos_exps. Returns the
    chosen row index. Two draws from the training RNG (game, then position), so a
    run stays reproducible from its seed. Modulo bias is < count / 2**32.
    """
    n_games = game_offsets.shape[0] - 1
    g = next_u32(rng_state) % n_games
    lo = game_offsets[g]
    return lo + next_u32(rng_state) % (game_offsets[g + 1] - lo)


@njit(cache=True)
def play_and_learn_from_positions(weights3, stage_exps, started, rng_state, alpha, afterstates, rewards,
                                  pos_exps, game_offsets):
    """Like ``play_and_learn`` but start from a sampled saved position (no initial spawns).

    Returns (score, max_exp, moves, start_row); score counts only rewards earned here.
    """
    row = pick_start(pos_exps, game_offsets, rng_state)
    board = pos_exps[row].copy()
    score, max_exp, n = _play_and_learn_from(board, weights3, stage_exps, started, rng_state, alpha,
                                             afterstates, rewards)
    return score, max_exp, n, row


@njit(cache=True)
def _play_and_learn_from(board, weights3, stage_exps, started, rng_state, alpha, afterstates, rewards):
    """Play greedily from ``board`` (a state, before a move) to the end, then learn."""
    n = 0
    score = 0
    while True:
        move, after, reward = choose_move(weights3, stage_exps, started, board)
        if move == -1:
            break
        afterstates[n, :] = after
        rewards[n] = reward
        n += 1
        score += reward
        board = after.copy()
        spawn_random(board, rng_state)
    max_exp = 0
    for k in range(16):
        if board[k] > max_exp:
            max_exp = board[k]
    if alpha != 0.0:
        learn_backward(weights3, stage_exps, afterstates, rewards, n, alpha)
    return score, max_exp, n


@njit(cache=True)
def random_play_scores(n_games: int, rng_state: np.ndarray) -> np.ndarray:
    """Scores of n_games played with uniformly random legal moves (baseline)."""
    scores = np.zeros(n_games, dtype=np.int64)
    for g in range(n_games):
        board = np.zeros(16, dtype=np.uint8)
        spawn_random(board, rng_state)
        spawn_random(board, rng_state)
        while not is_game_over(board):
            mask = legal_moves(board)
            n_legal = 0
            for m in range(4):
                if mask[m]:
                    n_legal += 1
            pick = next_u32(rng_state) % n_legal
            for m in range(4):
                if mask[m]:
                    if pick == 0:
                        board, reward, _ = slide(board, m)
                        scores[g] += reward
                        break
                    pick -= 1
            spawn_random(board, rng_state)
    return scores


# ------------------------------------------------------------------- logging
CSV_FIELDS = (
    ["games", "window", "mean_score", "max_score", "mean_moves"]
    + [f"n_{t}" for t in MILESTONES]
    + [f"reach_{t}" for t in MILESTONES]
    + ["elapsed_s", "games_per_s"]
)


def window_row(games: int, scores: List[int], max_exps: List[int], moves: List[int], elapsed: float, window_s: float) -> Dict:
    """Stats over one window of games. n_T = games whose max tile reached >= T."""
    k = len(scores)
    row = {
        "games": games,
        "window": k,
        "mean_score": round(float(np.mean(scores)), 1),
        "max_score": int(max(scores)),
        "mean_moves": round(float(np.mean(moves)), 1),
    }
    for t, e in zip(MILESTONES, MILESTONE_EXPS):
        n = sum(1 for m in max_exps if m >= e)
        row[f"n_{t}"] = n
        row[f"reach_{t}"] = round(n / k, 4)
    row["elapsed_s"] = round(elapsed, 1)
    row["games_per_s"] = round(k / window_s, 2) if window_s > 0 else 0.0
    return row


def _fmt_time(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


def _progress_line(row: Dict, total: int, elapsed: float, games_done_this_run: int, run_elapsed: float) -> str:
    reach = " ".join(f"{t // 1024}k:{100 * row[f'reach_{t}']:5.1f}%" for t in MILESTONES)
    rate = games_done_this_run / run_elapsed if run_elapsed > 0 else 0.0
    eta = (total - row["games"]) / rate if rate > 0 else 0.0
    return (
        f"[{row['games']:>9,}/{total:,}] {_fmt_time(elapsed)} | mean {row['mean_score']:>9,.0f} "
        f"max {row['max_score']:>7,} | {reach} | {row['games_per_s']:.1f} g/s | ETA {_fmt_time(eta)}"
    )


# --------------------------------------------------------------- checkpoints
def save_checkpoint(net: StagedNetwork, out: Path, state: Dict, keep: bool) -> Path:
    """Write latest.ntw atomically (temp file + rename) so a crash mid-save
    never leaves a broken checkpoint. ``state`` goes into the file's metadata."""
    net.meta.update(state)
    tmp = out / "latest.ntw.tmp"
    net.save(tmp)
    latest = out / "latest.ntw"
    os.replace(tmp, latest)
    if keep:
        numbered = out / f"ckpt_{state['games']:09d}.ntw"
        net.save(numbered)
    return latest


# ------------------------------------------------------------ start positions
def load_start_positions(path: str | Path):
    """Read a positions JSONL (scripts/extract_late_positions.py) for --start-positions.

    Returns (pos_exps uint8[N, 16], game_offsets int64[G+1], rows, info). Rows are
    grouped by source game (``game_id`` if present, else ``source_file``, else
    ``source_seed``) in first-seen order; ``rows[i]`` is the original JSON object of
    pos_exps[i]. ``info`` (path, sha256, counts) goes into the checkpoint metadata.
    """
    import hashlib
    import json

    from agents.ntuple.board import to_exponents

    path = Path(path)
    data = path.read_bytes()
    groups: Dict[object, List[dict]] = {}
    for line in data.decode("utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            key = r.get("game_id", r.get("source_file", r.get("source_seed")))
            groups.setdefault(key, []).append(r)
    rows = [r for g in groups.values() for r in g]
    if not rows:
        raise ValueError(f"{path}: no positions")
    pos_exps = np.stack([to_exponents(r["board"]) for r in rows]).astype(np.uint8)
    game_offsets = np.zeros(len(groups) + 1, dtype=np.int64)
    game_offsets[1:] = np.cumsum([len(g) for g in groups.values()])
    for i, r in enumerate(rows):
        if not np.any(legal_moves(pos_exps[i])):
            raise ValueError(f"{path}: position {i} has no legal move")
    info = {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "n_positions": len(rows),
        "n_games": len(groups),
        "source_seeds": sorted({int(r["source_seed"]) for r in rows if "source_seed" in r}),
    }
    return pos_exps, game_offsets, rows, info


# ---------------------------------------------------------------------- main
def train(
    games: int,
    alpha: float = 0.1,
    seed: int = DEFAULT_SEED,
    out: str | Path = "weights/ntuple/run1",
    checkpoint_every: int = 10_000,
    log_every: int = 1_000,
    resume: Optional[str | Path] = None,
    init_value: float = 0.0,
    keep_checkpoints: bool = False,
    verbose: bool = True,
    stages: int = 1,
    stage_tiles: Optional[Sequence[int]] = None,
    init_from: Optional[str | Path] = None,
    start_positions: Optional[str | Path] = None,
) -> Dict:
    """Train until ``games`` games in total have been played (counting resumed ones).

    stages / stage_tiles: multi-stage TD. With stages=2 and stage_tiles=[16384],
        boards whose max tile is below 16,384 use network 1, the rest network 2.
    init_value: optimistic initialisation; every board's value starts here.
    init_from: start a NEW run from this weights file (its stages, stage tiles,
        init value and ``stages_started``), with fresh counters and RNG from ``seed``.
        Unlike ``resume`` it ignores the file's training counters.
    start_positions: JSONL of saved boards; every game starts from one of them
        (uniform over source games, then over that game's positions, drawn from the
        training RNG). The chosen row of each game is written to ``starts.csv``.
    Returns {"net", "rows", "games"}; rows are the CSV rows written this run.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if stage_tiles is None:
        stage_tiles = [16384] if stages == 2 else []
    if resume and init_from:
        raise ValueError("use either resume or init_from, not both")

    starts = None
    if start_positions:
        pos_exps, game_offsets, pos_rows, pos_info = load_start_positions(start_positions)
        starts = (pos_exps, game_offsets)

    if resume:
        net = StagedNetwork.load(resume)
        if verbose and (net.n_stages != stages or net.stage_tiles != list(stage_tiles)):
            print(f"note: using the checkpoint's {net.n_stages} stage(s) {net.stage_tiles}, "
                  f"not --stages {stages} {list(stage_tiles)}", flush=True)
        m = net.meta
        started = np.array(m.get("stages_started", [1] * net.n_stages), dtype=np.uint8)
        done = int(m["games"])
        rng_state = np.array([int(m["rng_state"])], dtype=np.int64)
        elapsed0 = float(m["elapsed_s"])
        init_value = float(m.get("init_value", 0.0))
        win_scores = list(m.get("window_scores", []))
        win_exps = list(m.get("window_max_exps", []))
        win_moves = list(m.get("window_moves", []))
        seed = int(m.get("seed", seed))
        if verbose:
            print(f"resumed {resume}: {done:,} games, {_fmt_time(elapsed0)} trained", flush=True)
        prev = m.get("start_positions")
        prev_sha = prev.get("sha256") if prev else None
        if prev_sha != (pos_info["sha256"] if starts else None):
            raise ValueError(f"resume: --start-positions does not match the checkpoint's ({prev})")
    elif init_from:
        net = StagedNetwork.load(init_from)
        if verbose and (net.n_stages != stages or net.stage_tiles != list(stage_tiles)):
            print(f"note: using {init_from}'s {net.n_stages} stage(s) {net.stage_tiles}", flush=True)
        src_meta = net.meta
        started = np.array(src_meta.get("stages_started", [1] * net.n_stages), dtype=np.uint8)
        init_value = float(src_meta.get("init_value", 0.0))
        net.meta = {k: v for k, v in src_meta.items() if k.startswith("promoted_")}
        net.meta["init_from"] = Path(init_from).as_posix()
        done, elapsed0 = 0, 0.0
        rng_state = make_rng(seed)
        win_scores, win_exps, win_moves = [], [], []
        if verbose:
            print(f"init from {init_from}: stages={net.n_stages} {net.stage_tiles} "
                  f"started={started.tolist()}", flush=True)
    else:
        net = StagedNetwork(stages, stage_tiles, init_value=init_value)
        started = np.zeros(stages, dtype=np.uint8)
        started[0] = 1  # later stages are warm-started when first reached
        done, elapsed0 = 0, 0.0
        rng_state = make_rng(seed)
        win_scores, win_exps, win_moves = [], [], []
    # On resume, --alpha from the command line wins (so it can be lowered later
    # in training); the seed, RNG state and init value come from the checkpoint.
    net.meta.update({"alpha": alpha, "seed": seed, "init_value": init_value})
    if starts is not None:
        net.meta["start_positions"] = pos_info
        if verbose:
            print(f"start positions: {pos_info['n_positions']:,} boards from {pos_info['n_games']} source games "
                  f"({pos_info['path']}, sha256 {pos_info['sha256'][:12]})", flush=True)

    afterstates = np.zeros((MAX_MOVES, 16), dtype=np.uint8)
    rewards = np.zeros(MAX_MOVES, dtype=np.int64)

    log_path = out / "train_log.csv"
    new_log = not log_path.exists()
    rows: List[Dict] = []
    t_start = time.perf_counter()
    t_window = t_start
    start_done = done

    def state() -> Dict:
        return {
            "games": done,
            "rng_state": int(rng_state[0]),
            "elapsed_s": elapsed0 + (time.perf_counter() - t_start),
            "window_scores": win_scores,
            "window_max_exps": win_exps,
            "window_moves": win_moves,
            "stages_started": [int(x) for x in started],
        }

    if verbose:
        staging = f" stages={net.n_stages} {net.stage_tiles}" if net.n_stages > 1 else ""
        print(f"training to {games:,} games | alpha={alpha} seed={seed} init={init_value}{staging} out={out}",
              flush=True)
    starts_f = None
    if starts is not None:
        starts_path = out / "starts.csv"
        new_starts = not starts_path.exists()
        starts_f = starts_path.open("a", newline="", encoding="utf-8")
        starts_w = csv.writer(starts_f)
        if new_starts:
            starts_w.writerow(["game", "row", "source_seed", "move_index", "game_id", "score", "max_tile", "moves"])
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_log:
            writer.writeheader()
        try:
            while done < games:
                if starts is None:
                    score, max_exp, n = play_and_learn(
                        net.weights, net.stage_exps, started, rng_state, alpha, afterstates, rewards
                    )
                else:
                    score, max_exp, n, row_i = play_and_learn_from_positions(
                        net.weights, net.stage_exps, started, rng_state, alpha, afterstates, rewards,
                        starts[0], starts[1]
                    )
                    r = pos_rows[row_i]
                    starts_w.writerow([done + 1, int(row_i), r.get("source_seed"), r.get("move_index"),
                                       r.get("game_id"), int(score), 1 << int(max_exp), int(n)])
                done += 1
                win_scores.append(int(score))
                win_exps.append(int(max_exp))
                win_moves.append(int(n))
                if done % log_every == 0:
                    now = time.perf_counter()
                    elapsed = elapsed0 + (now - t_start)
                    row = window_row(done, win_scores, win_exps, win_moves, elapsed, now - t_window)
                    writer.writerow(row)
                    f.flush()
                    rows.append(row)
                    if verbose:
                        print(_progress_line(row, games, elapsed, done - start_done, now - t_start), flush=True)
                    win_scores, win_exps, win_moves = [], [], []
                    t_window = now
                if done % checkpoint_every == 0:
                    path = save_checkpoint(net, out, state(), keep_checkpoints)
                    if verbose:
                        print(f"  checkpoint -> {path}", flush=True)
        except KeyboardInterrupt:
            if verbose:
                print(f"\ninterrupted at {done:,} games; saving checkpoint...", flush=True)
            save_checkpoint(net, out, state(), keep_checkpoints)
            raise
        finally:
            if starts_f is not None:
                starts_f.close()
    if done % checkpoint_every != 0 or done == start_done:
        path = save_checkpoint(net, out, state(), keep_checkpoints)
        if verbose:
            print(f"  final checkpoint -> {path}", flush=True)
    return {"net": net, "rows": rows, "games": done}


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, required=True, help="total games to reach (resumed games count)")
    p.add_argument("--alpha", type=float, default=0.1, help="learning rate (spread over 32 lookups)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", default="weights/ntuple/run1")
    p.add_argument("--checkpoint-every", type=int, default=10_000)
    p.add_argument("--log-every", type=int, default=1_000)
    p.add_argument("--resume", default=None, help="checkpoint .ntw to continue from")
    p.add_argument("--init-value", type=float, default=0.0,
                   help="optimistic init: every board's value starts here (each weight = V/32)")
    p.add_argument("--stages", type=int, default=1, help="number of stage networks (multi-stage TD)")
    p.add_argument("--stage-tile", type=int, nargs="+", default=None,
                   help="tile(s) that start the next stage; default 16384 for --stages 2")
    p.add_argument("--keep-checkpoints", action="store_true", help="also keep numbered ckpt_<N>.ntw files")
    p.add_argument("--init-from", default=None,
                   help="start a new run from this .ntw (e.g. a promoted 2-stage file); fresh counters/RNG")
    p.add_argument("--start-positions", default=None,
                   help="JSONL of boards; each game starts from one (uniform over source games, then positions)")
    a = p.parse_args(argv)
    if a.stages < 1:
        p.error("--stages must be >= 1")
    if a.stages > 2 and a.stage_tile is None:
        p.error(f"--stages {a.stages} needs {a.stages - 1} --stage-tile values")
    try:
        train(
            a.games, a.alpha, a.seed, a.out, a.checkpoint_every, a.log_every,
            a.resume, a.init_value, a.keep_checkpoints, stages=a.stages, stage_tiles=a.stage_tile,
            init_from=a.init_from, start_positions=a.start_positions,
        )
    except ValueError as e:  # e.g. bad stage tiles
        p.error(str(e))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
