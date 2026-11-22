"""Shared pytest setup."""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def ntuple_test_weights(tmp_path_factory):
    """Point $NTUPLE_WEIGHTS at an all-zero n-tuple weights file for the test run.

    The registered "ntuple" agent refuses to start without weights, and trained
    weights (268 MB) are not in git or CI. With zero weights the agent plays
    greedy-on-reward, which is enough for the shared legality/reproducibility
    tests. Any NTUPLE_WEIGHTS set in your shell is restored afterwards.
    """
    from agents.ntuple.network import NTupleNetwork

    path = tmp_path_factory.mktemp("ntuple") / "zero.ntw"
    NTupleNetwork().save(path)
    old = os.environ.get("NTUPLE_WEIGHTS")
    os.environ["NTUPLE_WEIGHTS"] = str(path)
    yield path
    if old is None:
        os.environ.pop("NTUPLE_WEIGHTS", None)
    else:
        os.environ["NTUPLE_WEIGHTS"] = old
