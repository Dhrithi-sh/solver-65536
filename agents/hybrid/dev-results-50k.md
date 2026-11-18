# P7 development results with P5's 50k checkpoint

## Board collection

Hybrid-d3 played 20 games on development seeds 100–119 using P5's
ntuple-v0-50k checkpoint. The mean score was 150,103.8. Of the 20 games,
19 reached 8,192, none reached 16,384, and none reached 32,768. These runs
produced no starting boards for P8's late-game benchmark, so the
32,768-to-65,536 conversion rate remains unmeasured.

## Integration tests

All 27 hybrid tests passed locally with Numba active and the C++ extension
built. All four hybrid configurations completed a full game through the
locally integrated evaluation runner using P5's ntuple-v0-50k weights.

## Thread benchmark

Mean milliseconds per move across 20 positions from one game.

| Depth | 1 thread | 4 threads | Speed-up |
|---|---:|---:|---:|
| 2 | 0.907 ms | 0.747 ms | 1.21× |
| 3 | 13.763 ms | 4.867 ms | 2.83× |

## Development sweep

Depth 2, base cutoff 0.0001, development seeds 0–9.

| Late-game trigger | Mean score | ms/move | Reached 8,192 | Reached 16,384 |
|---|---:|---:|---:|---:|
| Off | 108,430.8 | 0.637 | 6/10 | 0/10 |
| 8,192 | 125,602.0 | 2.707 | 6/10 | 0/10 |
| 16,384 | 108,430.8 | 0.652 | 6/10 | 0/10 |

The 8,192 trigger increased mean score by 15.8% on these ten seeds and
used 4.25× as much time per move. No game reached the registered 16,384
trigger, so these runs did not measure its late-game effect.
