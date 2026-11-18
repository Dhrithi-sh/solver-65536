# P6: CNN + Double DQN

A small deep reinforcement-learning agent that uses the existing engine unchanged.
The shared interface is `choose_move(game)`, not `choose_move(board)`.

## Run from the repository root

```bash
python -m pip install -r requirements.txt
python -m pip install -r agents/deeprl/requirements-deeprl.txt
python -m pytest tests agents/deeprl/test_deeprl.py
```

Quick check that training runs (this does NOT produce a strong agent):

```bash
python -m agents.deeprl.train --steps 2000 --warmup 128 --epsilon-steps 1600 --checkpoint weights/dqn_smoke.pt --log logs/dqn_smoke.csv
```

A first real training run on CPU:

```bash
python -m agents.deeprl.train --steps 100000 --checkpoint weights/dqn.pt
```

Or use a time budget; the run stops at whichever limit comes first:

```bash
python -m agents.deeprl.train --steps 1000000 --minutes 30 --checkpoint weights/dqn.pt
```

If your PyTorch installation supports an NVIDIA GPU, add `--device cuda`.
For this tiny network CPU may be competitive: measure rather than assuming.
Training saves every 5,000 steps and on normal completion or Ctrl+C.
Repeated commands start fresh: optimizer/replay resumption is not implemented.
Default seeds start at 1,000,000; confirm that P8 excludes this range from testing.

## P8 registration (two additions)

Only `agents/deeprl/` was changed, in accordance with folder ownership.
Ask P8 to add the following in `eval/run.py`, after the existing imports and registry:

```python
from agents.deeprl import DQNAgent
AGENTS["dqn"] = DQNAgent
```

The model defaults to `weights/dqn.pt`, or the path in `DQN_CHECKPOINT`.
Then run the actual shared harness:

```bash
python -m eval.run --agent dqn --games 10 --seed0 0
python -m eval.replay results/dqn.jsonl --index 0
```

Those seeds are a development example, not an agreed final test set. Use P8's final
seed ranges when set. Give the commit, machine, checkpoint and command with results.
DQN deliberately fails if its checkpoint is missing instead of silently benchmarking
an untrained random network. Consequently, P8 should only register DQN in environments
with PyTorch and a checkpoint; unconditional registration also affects the generic CI tests.

For an interactive integration check without editing anyone else's folder:

```python
from agents.deeprl import DQNAgent
from engine.game import Game
agent = DQNAgent("weights/dqn.pt")
game = Game(seed=0)
move = agent.choose_move(game)
print(move, agent.move_scores(game))
```

## What the implementation does

- Gym-style `reset` and `step` return one-hot observations and an action mask.
  This is a lightweight wrapper, not a registered Gymnasium environment.
- Input shape is **18 x 4 x 4**: empty, exponents 1 through 16, then an overflow
  channel for 131,072 and above. The starter TODO's 16 channels could not represent
  65,536 separately. Exponent 16 has its own channel here.
- Two convolution layers (32 and 64 filters), then 128 hidden units and four Q values.
- Epsilon-greedy exploration selects only legal moves. Evaluation is greedy and masked.
- Replay buffer, a separate target network, Double DQN targets, Huber loss and gradient clipping.
- Raw merge reward divided by 128; gamma 0.99. This is a discounted-score objective.
  No heuristic reward bonuses or terminal penalties are added.
- Replay augmentation uniformly samples from all eight square symmetries and transforms
  both boards, the action, and the next-state legal mask together.
- Terminal transitions have zero bootstrap value. Models train every four game moves;
  target parameters copy every 250 gradient updates.
- `move_scores(game)` exposes masked Q estimates for P8's viewer. These are predictions,
  not success probabilities. Neither agent method mutates the supplied game.

## Fair comparison and limits

Train an augmented and an unaugmented model from scratch with the same seed and step
budget (`--no-augmentation` disables it). Store them under different checkpoint/log names.
Compare them through P8's harness on the same held-out seeds.

For P5 comparison, agree on either equal environment interactions or equal wall-clock
budget on the same hardware; these are distinct experiments. Checkpoint metadata records
steps, updates, elapsed seconds and configuration. Training CSV scores include exploration
and must not be used as the reported evaluation results.

Same seeds reproduce a game's spawns given the SAME sequence of moves. Different agents
usually produce different boards and empty-cell sets, so same-seed games do not guarantee
identical spawned coordinates. They provide paired seeded evaluation, not identical boards.

Late-game boards can be supplied via `GameEnv.reset(seed, board)` or `Game(seed, board)`.
P8 still needs to provide the official saved boards and late-game harness. Success at 65,536
has not been established by this implementation or its smoke test. Reaching that tile is
not guaranteed; a short DQN run should be presented as an initial deep RL baseline.

## Files

- `model.py`: encoding, CNN, board/action symmetries.
- `env.py`: wrapper around the shared engine.
- `train.py`: self-play and learning.
- `agent.py`: checkpoint loading and shared agent interface.
- `test_deeprl.py`: large tiles, all symmetry/action mappings, wrapper and agent contracts.

No engine, evaluator, top-level requirements or teammate-owned files were modified.
