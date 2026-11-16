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
from typing import Dict, List, Optional

import numpy as np

from agents.ntuple._jit import njit
from agents.ntuple.board import is_game_over, legal_moves, make_rng, next_u32, slide, spawn_random
from agents.ntuple.network import NTupleNetwork, evaluate_exps, update_exps

MILESTONES = (2048, 4096, 8192, 16384, 32768)
MILESTONE_EXPS = tuple(t.bit_length() - 1 for t in MILESTONES)
DEFAULT_SEED = 10_000_000  # training seeds stay clear of the shared eval seeds

# Every move spawns a tile worth >= 2 and merges never lose value, so the sum
# of tiles grows by >= 2 per move. A 4x4 board's sum is always < 2**18, hence
# no game can last 2**17 moves; this buffer can never overflow.
MAX_MOVES = 1 << 17


# -------------------------------------------------------------------- kernels
@njit(cache=True)
def choose_move(weights: np.ndarray, board: np.ndarray):
    """1-ply greedy: argmax over legal moves of reward + V(afterstate).

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
        value = reward + evaluate_exps(weights, after)
        if best_move == -1 or value > best_value:
            best_move = m
            best_value = value
            best_after = after
            best_reward = reward
    return best_move, best_after, best_reward


@njit(cache=True)
def learn_backward(weights: np.ndarray, afterstates: np.ndarray, rewards: np.ndarray, n: int, alpha: float) -> None:
    """Backward TD(0) pass over the n recorded (afterstate, reward) pairs.

    rewards[t] is the reward of the move that produced afterstates[t]; the
    target for afterstate t uses the reward of the NEXT move, rewards[t+1].
    """
    if n == 0:
        return
    # The final afterstate was followed by a spawn that ended the game.
    last = afterstates[n - 1]
    update_exps(weights, last, alpha * (0.0 - evaluate_exps(weights, last)))
    for t in range(n - 2, -1, -1):
        target = rewards[t + 1] + evaluate_exps(weights, afterstates[t + 1])
        s = afterstates[t]
        update_exps(weights, s, alpha * (target - evaluate_exps(weights, s)))


@njit(cache=True)
def play_and_learn(weights, rng_state, alpha, afterstates, rewards):
    """Play one self-play game, then learn from it. Returns (score, max_exp, moves).

    With alpha = 0 nothing is learned (useful as a baseline).
    """
    board = np.zeros(16, dtype=np.uint8)
    spawn_random(board, rng_state)
    spawn_random(board, rng_state)
    n = 0
    score = 0
    while True:
        move, after, reward = choose_move(weights, board)
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
        learn_backward(weights, afterstates, rewards, n, alpha)
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
def save_checkpoint(net: NTupleNetwork, out: Path, state: Dict, keep: bool) -> Path:
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
) -> Dict:
    """Train until ``games`` games in total have been played (counting resumed ones).

    Returns {"net", "rows", "games"}; rows are the CSV rows written this run.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    if resume:
        net = NTupleNetwork.load(resume)
        m = net.meta
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
    else:
        net = NTupleNetwork(init_value=init_value)
        done, elapsed0 = 0, 0.0
        rng_state = make_rng(seed)
        win_scores, win_exps, win_moves = [], [], []
    # On resume, --alpha from the command line wins (so it can be lowered later
    # in training); the seed, RNG state and init value come from the checkpoint.
    net.meta.update({"alpha": alpha, "seed": seed, "init_value": init_value})

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
        }

    if verbose:
        print(f"training to {games:,} games | alpha={alpha} seed={seed} out={out}", flush=True)
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_log:
            writer.writeheader()
        try:
            while done < games:
                score, max_exp, n = play_and_learn(net.weights, rng_state, alpha, afterstates, rewards)
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
    p.add_argument("--init-value", type=float, default=0.0, help="initial value of every weight")
    p.add_argument("--keep-checkpoints", action="store_true", help="also keep numbered ckpt_<N>.ntw files")
    a = p.parse_args(argv)
    try:
        train(
            a.games, a.alpha, a.seed, a.out, a.checkpoint_every, a.log_every,
            a.resume, a.init_value, a.keep_checkpoints,
        )
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
