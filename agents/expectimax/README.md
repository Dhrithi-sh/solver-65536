# agents/expectimax — P3 (Expectimax)

**Owner:** P3 · **Review buddy:** P4 · **Label:** `P3-expectimax`

Expectimax search: max nodes for the player, chance nodes for the tile spawn
(2 with p=0.9, 4 with p=0.1, uniform over empty cells). Takes any `ValueFunction`
for its leaves (P2 heuristics now, P5's n-tuple later through P7).

## TODO
- [ ] Expectimax with chance nodes, using `Game.slide` on copies (never mutate the game)
- [ ] Transposition table, keyed by board + depth
- [ ] Probability cutoff (prune chance branches below a cumulative probability)
- [ ] Adaptive depth (deeper when few empty cells / late game)
- [ ] Plot depth vs strength vs time per move (data from `eval/run.py`)
- [ ] Register in `eval/run.py` → `AGENTS`
