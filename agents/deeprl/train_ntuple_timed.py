"""Time-budget adapter for P5's existing trainer; run from the repository root."""
import argparse
import json
import time
from pathlib import Path
from agents.ntuple import train as trainer
from agents.ntuple._jit import HAVE_NUMBA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minutes', type=float, default=5)
    parser.add_argument('--out', default='weights/compare5min/ntuple')
    args = parser.parse_args()
    if args.minutes <= 0:
        parser.error('--minutes must be positive')
    if not HAVE_NUMBA:
        parser.error('Numba is required for this comparison')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error('Output folder is not empty; use a new --out folder')
    original = trainer.play_and_learn
    start = None
    elapsed = 0.0
    games = moves = 0
    timed_out = False

    def timed_game(*arguments):
        nonlocal start, elapsed, games, moves, timed_out
        if start is None:
            start = time.perf_counter()
        elapsed = time.perf_counter() - start
        if elapsed >= args.minutes * 60:
            timed_out = True
            # P5's trainer catches this, saves latest.ntw, then re-raises.
            raise KeyboardInterrupt
        result = original(*arguments)
        games += 1
        moves += int(result[2])
        return result

    trainer.play_and_learn = timed_game
    print('Starting timed n-tuple training. Compilation and periodic saves count toward the budget.', flush=True)
    try:
        trainer.train(games=1_000_000_000, seed=10_000_000, out=out,
                      checkpoint_every=10000, log_every=1000)
    except KeyboardInterrupt:
        if not timed_out:
            elapsed = time.perf_counter() - start if start is not None else 0.0
            print('Stopped early by user; this is not a completed five-minute run.')
    finally:
        trainer.play_and_learn = original
    summary = {'agent': 'ntuple', 'requested_seconds': args.minutes * 60,
               'training_loop_seconds': elapsed, 'completed_budget': timed_out,
               'completed_games': games, 'moves_in_completed_games': moves,
               'seed': 10_000_000, 'alpha': 0.1, 'stages': 1,
               'timing': 'Starts at first game; includes JIT compilation, logging and periodic saves; excludes setup and final save. Stops between games.'}
    (out / 'budget.json').write_text(json.dumps(summary, indent=2))
    print(f'Saved {out / "latest.ntw"}; games={games}, moves={moves}, timed seconds={elapsed:.2f}')
    print('The n-tuple checkpoint is approximately 268 MB. Evaluate through eval.run.')


if __name__ == '__main__':
    main()
