# agents/ntuple — P5 (N-tuple network)

**Owner:** P5 · **Review buddy:** P6 · **Label:** `P5-ntuple`

N-tuple network value function (`ValueFunction.evaluate`) trained with TD learning
on afterstates. Weights go in `weights/` (git-ignored; share through a release or drive link).

## TODO
- [ ] 4 x 6-tuple network with all 8 board symmetries (rotations + reflections)
- [ ] TD(0) afterstate learning
- [ ] Multi-stage weights (separate tables per game stage, e.g. after reaching 16,384 / 32,768)
- [ ] Optimistic initialisation
- [ ] Learning curves (score and reach rates vs training games)
- [ ] Save/load weights; register a greedy-on-value agent in `eval/run.py`
