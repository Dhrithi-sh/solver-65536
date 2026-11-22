# P6: Deep reinforcement learning results

## Method

For P6, I implemented a CNN-based Double DQN agent using the shared game engine.

The board is represented using 18 one-hot channels: an empty-cell channel, 16 channels for tiles from 2 to 65,536, and an overflow channel. Two convolutional layers extract spatial features, and fully connected layers produce four Q-values, one for each direction. Illegal moves are masked.

Training uses epsilon-greedy exploration, a replay buffer and minibatches of 64 transitions. Rotations and reflections provide eight board symmetries, with actions and legal-move masks transformed consistently.

The online network selects the next legal action, and the target network evaluates it. The learning target combines the observed merge reward divided by 128 with a discounted future estimate, using a discount factor of 0.99. For terminal transitions, the target contains only the reward.

Smooth L1 loss, backpropagation and Adam update the online network. Its weights are periodically copied to the target network. During evaluation, weights stay fixed and the agent chooses the highest-Q legal action without exploration.

## Longer training run

- Completed training moves: 302,560.
- Learning updates: 75,390.
- Completed training games: 2,279.
- Epsilon decreased from 1 to 0.05 over 300,000 moves.
- Original checkpoint name: dqn_500k.pt.
- Handoff checkpoint name: models/dqn_302560_steps.pt.

Although the original filename says 500k, training stopped at 302,560 moves. Both filenames refer to the same saved model.

The training CSV records completed games and ends at move 302,319. The checkpoint was saved partway through the next game. Completed-game seeds were 1,000,000 to 1,002,278.

The log recorded approximately 14.58 hours of elapsed time. This may include pauses or laptop suspension, so I do not use it as a verified active-compute budget.

## Baseline evaluation

All three agents were evaluated through the shared harness on 100 games using seeds 100 to 199.

| Agent | Mean score | Median score | Mean ms/move | Highest tile | Reached 2048 |
|---|---:|---:|---:|---:|---:|
| Random | 1025.4 | 1008 | 0.0630 | 256 | 0% |
| Greedy | 3003.2 | 2702 | 0.0935 | 512 | 0% |
| DQN | 2593.8 | 2418 | 0.5383 | 512 | 0% |

My DQN scored better than random but did not beat greedy. Its average score was approximately 2.53 times random's score and 13.6% below greedy's. It also took longer to choose each move.

DQN's highest tile was 512. None of these agents reached 2048 in this evaluation. These results describe this trained model and evaluation set, rather than every possible DQN implementation.

## Five-minute comparison with P5

For a separate experiment, I trained DQN and P5's n-tuple agent from scratch for approximately five minutes each on the same CPU laptop.

DQN completed 91,516 moves and 734 games. N-tuple completed 72,132,253 moves and 43,401 games, with a recorded training timer of 300.01 seconds.

This compares approximately equal training time, not equal training interactions or operations. Model setup and the final checkpoint save were outside the training timers.

Both agents were evaluated through the shared harness on 100 games using seeds 100000 to 100099.

| Result | DQN | N-tuple |
|---|---:|---:|
| Mean score | 1289.4 | 47972.3 |
| Median score | 1184 | 45434 |
| Mean ms/move | 0.5183 | 0.0213 |
| Highest tile | 256 | 8192 |
| Reached 2048 | 0% | 82% |
| Reached 4096 | 0% | 46% |
| Reached 8192 | 0% | 2% |

N-tuple performed substantially better and chose moves faster. Its implementation processed far more training moves within the time budget.

The five-minute DQN is a different model from the longer training run. Each experiment used one training run per learned model, so repeated runs would be needed to assess consistency.

## Hardware and source code

- Windows laptop.
- AMD Ryzen 5 7535HS processor.
- RAM reported by Windows: 7.2 GB.
- Both five-minute training runs used the CPU.
- Software versions recorded at handoff are in software_versions.txt.
- P6 source commit: 65afb11.
- Branch: p6-deeprl.
- PR: https://github.com/Dhrithi-sh/solver-65536/pull/15
- Source snapshot: P6_source_65afb11.zip.

The comparison workspace used P5 commit 1db1e03 with P6 integration changes. P5's implementation is an additional dependency for reproducing the n-tuple comparison.

## Reproducing the baseline evaluation

Extract P6_source_65afb11.zip. Run these commands from the extracted project root using a compatible Python environment:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r agents/deeprl/requirements-deeprl.txt
```

Set the model path below to its actual location on your machine:

```powershell
$env:DQN_CHECKPOINT = "C:\Users\HP\Downloads\P6_Final_Handoff\models\dqn_302560_steps.pt"
python -m eval.run --agent dqn --games 100 --seed0 100 --out-dir results/reproduced_baselines
python -m eval.run --agent greedy --games 100 --seed0 100 --out-dir results/reproduced_baselines
python -m eval.run --agent random --games 100 --seed0 100 --out-dir results/reproduced_baselines
```

Move timings can differ across machines and runs.

## Testing

The local test run passed 42 tests. Tests cover implementation behaviour, including encoding, symmetry consistency, legal actions and reproducibility. Passing tests does not establish strong playing performance.

## Limitations

Using the same seeds makes evaluation reproducible, but different actions produce different boards and may produce different tile-spawn positions.

Training scores include exploration and changing weights. The training graph is a diagnostic, not a held-out evaluation learning curve.

Symmetry augmentation is implemented, but its benefit was not isolated through a matching run without augmentation.

The project-wide late-game evaluation requires saved boards and P8's evaluation setup. The final team protocol still needs confirmation.

My DQN has not demonstrated reaching 65,536. Models, logs and results are supplied separately from the source-code PR.

## Package contents

- agent_comparison.png: longer-run DQN, random and greedy evaluation.
- training_curve.png: longer-run training scores and exploration.
- five_minute_comparison.png: separate timed DQN versus n-tuple comparison.
- inputs/: longer-run training CSV and baseline evaluation JSONL files.
- inputs/compare5min/: timed evaluation results and n-tuple budget record.
- logs/dqn_5min.csv: five-minute DQN training log.
- models/dqn_302560_steps.pt: longer-run DQN checkpoint.
- models/dqn_5min.pt: five-minute DQN checkpoint.
- models/ntuple_5min.ntw: five-minute n-tuple checkpoint.
- make_results.py: regenerates the graphs.
- software_versions.txt: recorded environment versions.
- HANDOFF_CHECKLIST.md: source, artifact and delivery information.
- P6_source_65afb11.zip: tracked source snapshot.

To regenerate graphs, run from this handoff folder:

```powershell
python -m pip install numpy matplotlib
python make_results.py inputs
```
