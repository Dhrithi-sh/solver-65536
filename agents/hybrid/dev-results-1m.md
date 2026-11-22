# P7 development results with P5's 1M checkpoint

Agent: hybrid-d3. Checkpoint: ntuple-v1-1m.
Both checkpoints were evaluated on the same 20 development seeds (100–119).

| Checkpoint | Mean score | Reached 8,192 | Reached 16,384 | Reached 32,768 |
|---|---:|---:|---:|---:|
| 50k | 150,103.8 | 19/20 | 0/20 | 0/20 |
| 1M | 251,562.8 | 20/20 | 11/20 | 0/20 |

With the 1M checkpoint, 5/10 games reached 16,384 on seeds 100–109
and 6/10 on seeds 110–119. The combined development reach rate was 55%.
Held-out evaluation is pending. These runs produced no 32,768 starting
boards, so late-game conversion remains unmeasured.
