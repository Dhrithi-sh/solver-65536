"""Promote a 1-stage n-tuple weights file to a 2-stage file for late-game fine-tuning.

    python scripts/ntuple_promote.py weights/ntuple/long/ntuple-plain-1m.ntw \\
        weights/ntuple/late/promoted-2stage.ntw --stage-tile 16384

Output (format version 2, read by ``StagedNetwork.load``, which is also how P7's
hybrid agent loads weights before handing them to its C++ search):
    shape        [2, 4, 16**6] little-endian float32, 64-byte aligned
    stage_tiles  [16384]  (TILE values on disk; loaders turn them into exponent 14)
    stage 0      the source weights, unchanged
    stage 1      an exact copy of stage 0
    meta         stages_started = [1, 1] (so training never re-copies stage 0 over
                 stage 1), plus where the weights came from. No training counters
                 (games / rng_state), so the file starts a fresh run via
                 ``train.py --init-from``; it is not a ``--resume`` checkpoint.

After writing, the script reloads both files and checks that, on --check-boards random
boards whose max tile is below the stage tile, the new file's value is EXACTLY
(bit-for-bit) the source file's value, and that stage 1 is byte-identical to stage 0.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.ntuple.network import NTupleNetwork, StagedNetwork  # noqa: E402


def promote(src: NTupleNetwork, stage_tile: int, source_name: str = "") -> StagedNetwork:
    """Two-stage network: stage 0 = ``src``, stage 1 = a copy of it."""
    w = np.empty((2,) + src.weights.shape, dtype=np.float32)
    w[0] = src.weights
    w[1] = src.weights
    meta = {
        "init_value": float(src.meta.get("init_value", 0.0)),
        "stages_started": [1, 1],
        "promoted_from": source_name,
        "promoted_from_games": src.meta.get("games"),
        "promoted_from_seed": src.meta.get("seed"),
        "promoted_from_sha256": None,  # filled in by main() when it knows the file
    }
    return StagedNetwork(2, [stage_tile], weights=w, meta=meta)


def random_boards_below(n: int, max_exp_exclusive: int, seed: int) -> np.ndarray:
    """n random boards (uint8[16] exponents), each with max exponent < max_exp_exclusive.

    Each cell is empty with probability 0.3, else a uniform exponent 1..max-1, so
    the set covers sparse and crowded boards and every tile value below the stage tile.
    """
    rng = np.random.default_rng(seed)
    exps = rng.integers(1, max_exp_exclusive, size=(n, 16), dtype=np.int64)
    exps[rng.random((n, 16)) < 0.3] = 0
    return exps.astype(np.uint8)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(src_path: Path, out_path: Path, n_boards: int, seed: int) -> None:
    """Raise AssertionError unless the promoted file matches the source as promised."""
    src = NTupleNetwork.load(src_path)
    dst = StagedNetwork.load(out_path)
    assert dst.n_stages == 2, dst.n_stages
    assert list(dst.stage_exps) == [dst.stage_tiles[0].bit_length() - 1]
    assert dst.meta.get("stages_started") == [1, 1], dst.meta.get("stages_started")
    assert np.array_equal(dst.weights[0].view(np.uint32), src.weights.view(np.uint32)), "stage 0 != source"
    assert np.array_equal(dst.weights[1].view(np.uint32), dst.weights[0].view(np.uint32)), "stage 1 != stage 0"
    boards = random_boards_below(n_boards, int(dst.stage_exps[0]), seed)
    mismatches = 0
    for b in boards:
        assert dst.stage_of(b) == 0
        if dst.evaluate(b) != src.evaluate(b):  # exact float equality, on purpose
            mismatches += 1
    assert mismatches == 0, f"{mismatches}/{n_boards} boards evaluate differently"
    print(f"verify OK: {n_boards} random boards below {dst.stage_tiles[0]} evaluate exactly as the "
          f"source; stage 0 == source and stage 1 == stage 0 (bitwise)")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path, help="1-stage .ntw file")
    p.add_argument("out", type=Path, help="2-stage .ntw file to write")
    p.add_argument("--stage-tile", type=int, default=16384)
    p.add_argument("--check-boards", type=int, default=1000)
    p.add_argument("--check-seed", type=int, default=0)
    a = p.parse_args(argv)
    src = NTupleNetwork.load(a.src)
    net = promote(src, a.stage_tile, source_name=a.src.as_posix())
    net.meta["promoted_from_sha256"] = sha256(a.src)
    net.save(a.out)
    print(f"wrote {a.out}: stages=2 stage_tiles={net.stage_tiles} (exponents {[int(e) for e in net.stage_exps]}) "
          f"stages_started={net.meta['stages_started']}")
    verify(a.src, a.out, a.check_boards, a.check_seed)


if __name__ == "__main__":
    main()
