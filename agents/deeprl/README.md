# agents/deeprl — P6 (Deep RL)

**Owner:** P6 · **Review buddy:** P5 · **Label:** `P6-deeprl`

Deep reinforcement learning agents. Torch or JAX is fine, but keep it out of the
top-level `requirements.txt`; add a `requirements-deeprl.txt` here instead so CI stays light.

## TODO
- [ ] Gym-style env wrapper around `engine.Game` (`reset`, `step`, legal-move mask)
- [ ] One-hot 16-channel board encoding (channel k = tile 2**k; channel 0 = empty)
- [ ] CNN DQN or PPO
- [ ] Symmetry augmentation (8 symmetries)
- [ ] Fair compute comparison vs the n-tuple network (same wall-clock / sample budget)
- [ ] Register in `eval/run.py` → `AGENTS`
