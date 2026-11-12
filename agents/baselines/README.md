# agents/baselines — P4 (Baselines)

**Owner:** P4 · **Review buddy:** P3 · **Label:** `P4-baselines`

Simple reference agents that every other approach is compared against.
Already here: `RandomAgent` (`random`) and `GreedyAgent` (`greedy`, best immediate score).

## TODO
- [ ] Corner strategy (fixed move priority that keeps the max tile in a corner)
- [ ] Minimax baseline (adversarial spawn: worst-case tile placement)
- [ ] MCTS baseline with random rollouts
- [ ] Stretch: MCTS with n-tuple rollouts / leaf values (with P5)
- [ ] Register each in `eval/run.py` → `AGENTS`
