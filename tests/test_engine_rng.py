"""P1 Batch 3c: the C++ generator must be CPython's random.Random, bit for bit.

Everything is compared against the real ``random.Random``: seeding (the whole
624-word internal state), getrandbits, random(), randrange (rejection loop and
how many outputs it consumes), and the state after long mixed sequences.
Skipped when the compiled extension is not built.
"""

import random
import unittest

from engine import batch as B
from engine import game as G

CPP = G._cengine


def need_cpp():
    if CPP is None or not hasattr(CPP, "PyRandom"):
        raise unittest.SkipTest("compiled extension engine._cengine not available")


SEEDS = [0, 1, 2, 5, -5, 12345, 2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**32 + 1, 2**53 + 7,
         2**63 - 1, 2**63, 2**64 - 1, 2**64, 2**100 + 12345, -(2**40), -(2**64)]


def state_of(py):
    return py.getstate()[1]  # (624 words..., index)


def test_seeding_matches_full_state_for_edge_case_seeds():
    need_cpp()
    for seed in SEEDS + list(range(100, 140)):
        assert CPP.PyRandom(seed).getstate() == state_of(random.Random(seed)), seed


def test_negative_seed_equals_positive_seed():
    need_cpp()
    assert CPP.PyRandom(-77).getstate() == CPP.PyRandom(77).getstate()


def test_getrandbits_every_width():
    need_cpp()
    for seed in (0, 3, 99):
        c, p = CPP.PyRandom(seed), random.Random(seed)
        for k in list(range(1, 33)) * 20:
            assert c.getrandbits(k) == p.getrandbits(k)
        assert c.getstate() == state_of(p)


def test_random_float_is_bit_identical():
    need_cpp()
    for seed in (0, 1, 2**40, 424242):
        c, p = CPP.PyRandom(seed), random.Random(seed)
        for _ in range(3000):  # crosses several 624-word twists
            assert c.random() == p.random()
        assert c.getstate() == state_of(p)


def test_randbelow_matches_randrange_and_output_consumption():
    need_cpp()
    # n = 1 still consumes outputs in CPython; the final state check catches any difference.
    for seed in (0, 7, 2**33):
        c, p = CPP.PyRandom(seed), random.Random(seed)
        for n in list(range(1, 17)) * 150 + [100, 255, 256, 257, 65535, 65537, 2**31 - 1, 2**31 + 1, 2**32 - 1]:
            assert c.randbelow(n) == p.randrange(n), (seed, n)
        assert c.getstate() == state_of(p)


def test_spawn_style_sequence_matches_engine_rng_calls():
    """The exact call pattern of Game._spawn: randrange(n_empty) then random()."""
    need_cpp()
    rng = random.Random(2024)
    for seed in range(40):
        c, p = CPP.PyRandom(seed), random.Random(seed)
        for _ in range(500):
            n = rng.randint(1, 16)
            assert c.randbelow(n) == p.randrange(n)
            assert (2 if c.random() < G.PROB_TWO else 4) == (2 if p.random() < G.PROB_TWO else 4)
        assert c.getstate() == state_of(p)


def test_policy_mask_is_shared_between_c_and_python():
    need_cpp()
    assert CPP.POLICY_SEED_MASK == B.POLICY_SEED_MASK
