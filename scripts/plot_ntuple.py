"""Plot n-tuple learning curves from training CSVs (agents/ntuple/train.py logs).

    python scripts/plot_ntuple.py                                  # all runs under weights/ntuple/compare
    python scripts/plot_ntuple.py weights/ntuple/compare/plain weights/ntuple/run1/train_log.csv
    python scripts/plot_ntuple.py --label plain=Plain --out results/plots

Each argument is a run directory (containing train_log.csv) or a CSV file; the
line label is the directory name unless overridden with --label NAME=TEXT.

Writes PNGs to --out (default results/plots, git-ignored):
    ntuple_mean_score.png   mean score per log window vs games trained, one line per run
    ntuple_reach.png        2048 / 8192 / 16384 reach rate vs games trained (one panel each)

These are TRAINING curves (self-play with the training RNG). Benchmark numbers
for the report come from `python -m eval.run` on the shared seeds.
Needs matplotlib (pip install matplotlib); it is not in requirements.txt so CI stays light.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

DEFAULT_GLOB_ROOT = Path("weights/ntuple/compare")
REACH_TILES = (2048, 8192, 16384)
Run = Tuple[str, List[Dict[str, float]]]


def read_log(path: Path) -> List[Dict[str, float]]:
    """Rows of a train_log.csv with every value converted to float."""
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def find_runs(args: Sequence[str], labels: Dict[str, str]) -> List[Run]:
    """Resolve CLI arguments (dirs or CSVs) to (label, rows); default: compare runs."""
    if args:
        paths = [Path(a) for a in args]
    else:
        paths = sorted(p.parent for p in DEFAULT_GLOB_ROOT.glob("*/train_log.csv"))
    runs: List[Run] = []
    for p in paths:
        csv_path = p / "train_log.csv" if p.is_dir() else p
        if not csv_path.is_file():
            sys.exit(f"error: no training log at {csv_path}")
        name = csv_path.parent.name
        rows = read_log(csv_path)
        if rows:
            runs.append((labels.get(name, name), rows))
    if not runs:
        sys.exit(f"error: no training logs found (looked in {DEFAULT_GLOB_ROOT}/*/train_log.csv)")
    return runs


def plot(runs: List[Run], out: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    written = []

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, rows in runs:
        ax.plot([r["games"] for r in rows], [r["mean_score"] for r in rows], label=label, linewidth=1.6)
    ax.set_xlabel("games trained")
    ax.set_ylabel("mean score (per log window)")
    ax.set_title("N-tuple TD(0): mean score during training")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out / "ntuple_mean_score.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(1, len(REACH_TILES), figsize=(4.2 * len(REACH_TILES), 4.2), sharey=True)
    for ax, tile in zip(axes, REACH_TILES):
        for label, rows in runs:
            ax.plot([r["games"] for r in rows], [100 * r[f"reach_{tile}"] for r in rows], label=label, linewidth=1.6)
        ax.set_title(f"reach {tile:,}")
        ax.set_xlabel("games trained")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("% of games in window")
    axes[0].set_ylim(0, 100)
    axes[-1].legend()
    fig.suptitle("N-tuple TD(0): tile reach rate during training")
    fig.tight_layout()
    path = out / "ntuple_reach.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written


def main(argv: List[str] | None = None) -> List[Path]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="run directories or train_log.csv files")
    p.add_argument("--out", default="results/plots", help="output directory for PNGs")
    p.add_argument("--label", action="append", default=[], metavar="NAME=TEXT", help="rename a run in the legend")
    a = p.parse_args(argv)
    labels = dict(item.split("=", 1) for item in a.label)
    written = plot(find_runs(a.runs, labels), Path(a.out))
    for path in written:
        print(f"wrote {path}")
    return written


if __name__ == "__main__":
    main()
