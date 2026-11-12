# agents/heuristics — P2 (Heuristics)

**Owner:** P2 · **Review buddy:** P1 · **Label:** `P2-heuristics`

Hand-crafted evaluation functions, exposed as `ValueFunction.evaluate(board) -> float`
(see `agents/base.py`) so that P3's expectimax and P7's hybrid can reuse them.

## TODO
- [ ] Eval terms: empty cells, monotonicity, smoothness, corner / snake weighting
- [ ] Combine the terms as a weighted sum and add a 1-ply greedy agent that uses it
- [ ] Tune the weights with CMA-ES or a GA (fixed training seeds, separate from eval seeds)
- [ ] Per-phase weights (early / mid / late game, e.g. split by max tile)
- [ ] Ablation table: drop one term at a time and report it with `eval/run.py` on shared seeds
- [ ] Register the agent(s) in `eval/run.py` → `AGENTS`
