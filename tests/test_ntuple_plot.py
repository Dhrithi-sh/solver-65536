"""scripts/plot_ntuple.py writes both PNGs from training CSVs (skipped without matplotlib)."""

import importlib.util
from pathlib import Path

import pytest

from agents.ntuple.train import train

pytest.importorskip("matplotlib")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_ntuple.py"


def load_script():
    spec = importlib.util.spec_from_file_location("plot_ntuple", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plots_written_for_two_runs(tmp_path):
    for name, seed in (("a", 1), ("b", 2)):
        train(20, seed=seed, out=tmp_path / name, log_every=10, checkpoint_every=1_000, verbose=False)
    out = tmp_path / "plots"
    written = load_script().main([str(tmp_path / "a"), str(tmp_path / "b" / "train_log.csv"),
                                  "--out", str(out), "--label", "a=first"])
    assert sorted(p.name for p in written) == ["ntuple_mean_score.png", "ntuple_reach.png"]
    for p in written:
        assert p.stat().st_size > 1000
