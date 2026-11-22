"""Generate comparison plots from benchmark JSONL results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


MILESTONES = (2048, 4096, 8192, 16384, 32768, 65536)


def load_results(results_dir: Path) -> dict:
    """Load valid benchmark JSONL files, including nested run folders."""
    runs = {}

    for path in sorted(results_dir.rglob("*.jsonl")):
        records = []

        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON: {path}:{line_number}")
                    continue

                required = {"seed", "score", "max_tile", "ms_per_move"}

                if not isinstance(record, dict) or not required.issubset(record):
                    continue

                records.append(record)

        if records:
            label = path.relative_to(results_dir).with_suffix("").as_posix()
            runs[label] = records

    if not runs:
        raise SystemExit(f"No valid benchmark results found in {results_dir}")

    return runs


def create_summary(runs: dict, out_dir: Path) -> None:
    """Create a CSV summary containing the main comparison metrics."""
    output = out_dir / "benchmark_summary.csv"

    columns = [
        "agent",
        "games",
        "mean_score",
        "median_score",
        "mean_ms_per_move",
    ]

    columns += [f"reach_{tile}" for tile in MILESTONES]

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for name, records in runs.items():
            scores = [r["score"] for r in records]

            row = {
                "agent": name,
                "games": len(records),
                "mean_score": round(statistics.mean(scores), 2),
                "median_score": round(statistics.median(scores), 2),
                "mean_ms_per_move": round(
                    statistics.mean(r["ms_per_move"] for r in records), 4
                ),
            }

            for tile in MILESTONES:
                row[f"reach_{tile}"] = round(
                    100 * sum(r["max_tile"] >= tile for r in records)
                    / len(records),
                    2,
                )

            writer.writerow(row)

    print("Saved:", output)


def create_plots(runs: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(runs)

    # 1. Average score comparison
    ordered = sorted(
        names,
        key=lambda name: statistics.mean(
            r["score"] for r in runs[name]
        ),
    )

    scores = [
        statistics.mean(r["score"] for r in runs[name])
        for name in ordered
    ]

    fig, ax = plt.subplots(figsize=(11, max(5, len(names) * 0.35)))
    ax.barh(ordered, scores)
    ax.set_title("Average Score Comparison")
    ax.set_xlabel("Mean game score")
    ax.set_ylabel("Agent")
    fig.tight_layout()
    fig.savefig(out_dir / "score_comparison.png", dpi=160)
    plt.close(fig)

    # 2. Score distribution
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.55), 6))

    ax.boxplot(
        [[r["score"] for r in runs[name]] for name in names],
        tick_labels=names,
        showfliers=False,
    )

    ax.set_title("Score Distribution")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Game score")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "score_distribution.png", dpi=160)
    plt.close(fig)

    # 3. Maximum tile reach rates
    fig, ax = plt.subplots(figsize=(11, 6))

    for name, records in runs.items():
        rates = [
            100 * sum(r["max_tile"] >= tile for r in records)
            / len(records)
            for tile in MILESTONES
        ]

        ax.plot(MILESTONES, rates, marker="o", label=name)

    ax.set_xscale("log", base=2)
    ax.set_xticks(MILESTONES)
    ax.set_xticklabels([str(t) for t in MILESTONES])
    ax.set_ylim(0, 105)
    ax.set_title("Maximum Tile Reach Rates")
    ax.set_xlabel("Target tile")
    ax.set_ylabel("Games reaching target (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "max_tile_comparison.png", dpi=160)
    plt.close(fig)

    # 4. Average decision time
    ordered = sorted(
        names,
        key=lambda name: statistics.mean(
            r["ms_per_move"] for r in runs[name]
        ),
    )

    times = [
        statistics.mean(r["ms_per_move"] for r in runs[name])
        for name in ordered
    ]

    fig, ax = plt.subplots(figsize=(11, max(5, len(names) * 0.35)))
    ax.barh(ordered, times)
    ax.set_title("Average Agent Decision Time")
    ax.set_xlabel("Milliseconds per move")
    ax.set_ylabel("Agent")
    fig.tight_layout()
    fig.savefig(out_dir / "decision_time.png", dpi=160)
    plt.close(fig)

    print("\nCreated 4 comparison plots in:", out_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results/plots")

    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    runs = load_results(results_dir)

    print("Loaded benchmark runs:", len(runs))

    for name, records in runs.items():
        print(f"  {name}: {len(records)} games")

    create_plots(runs, out_dir)
    create_summary(runs, out_dir)

    return runs


if __name__ == "__main__":
    main()