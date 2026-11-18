# \# P6: Deep reinforcement learning

# 

# For my part, I used a CNN with Double DQN to learn how to play 2048. It uses the shared game engine and the `choose\_move(game)` interface.

# 

# \## How it works

# 

# The board is encoded using 18 channels: one for empty cells, 16 for tiles from 2 to 65,536, and one for larger tiles.

# 

# The CNN gives a value to each of the four moves. During evaluation, the agent picks the legal move with the highest value.

# 

# Training uses a replay buffer and a target network. I also used all eight rotations and reflections of the board for augmentation, with the actions and legal-move masks transformed along with them. The reward is the merge score divided by 128, and the discount factor is 0.99.

# 

# The environment has Gym-style reset and step methods, but it is not registered with Gymnasium.

# 

# \## Setup

# 

# Run these commands from the project root, using your Python environment:

# 

# ```powershell

# python -m pip install -r requirements.txt

# python -m pip install -r agents/deeprl/requirements-deeprl.txt

# ```

# 

# \## Training

# 

# For a new 100,000-move training run:

# 

# ```powershell

# python -m agents.deeprl.train --steps 100000 --checkpoint weights/dqn.pt --log logs/dqn\_training.csv

# ```

# 

# You can also add `--minutes 5` to set a time limit. Training stops when either the step limit or time limit is reached.

# 

# The model is saved every 5,000 steps, at the end, and when training is stopped with Ctrl+C. Running the command again starts a new model; it does not resume the previous training.

# 

# \## Evaluation

# 

# DQN is registered in the shared evaluation harness. Set the checkpoint path before running it:

# 

# ```powershell

# $env:DQN\_CHECKPOINT = "weights/dqn\_500k.pt"

# python -m eval.run --agent dqn --games 100 --seed0 100 --out-dir results/check100

# ```

# 

# The checkpoint must be downloaded separately. If no path is set, the agent looks for `weights/dqn.pt`. It gives an error if the file is missing.

# 

# These are the seeds used for my development evaluation. P8 will confirm the final team evaluation setup.

# 

# \## Results

# 

# My longer run completed 302,560 training moves. Its checkpoint is called `dqn\_500k.pt`, but it did not finish 500,000 moves.

# 

# On 100 games with seeds 100–199:

# 

# | Agent | Mean score |

# |---|---:|

# | Random | 1025.4 |

# | Greedy | 3003.2 |

# | DQN | 2593.8 |

# 

# DQN beat random but did not beat greedy. Its highest tile was 512.

# 

# I also trained DQN and P5's n-tuple agent from scratch for approximately five minutes each on the same laptop. On 100 evaluation games with seeds 100000–100099:

# 

# | Agent | Mean score | Highest tile |

# |---|---:|---:|

# | DQN | 1289.4 | 256 |

# | N-tuple | 47972.3 | 8192 |

# 

# N-tuple performed much better in this experiment. This compares the implementations under the same time budget, rather than the same number of training moves. Each result is from one training run.

# 

# `train\_ntuple\_timed.py` runs the timed P5 training. It requires P5's n-tuple implementation and its dependencies, which are not included in this P6 change.

# 

# \## Tests

# 

# ```powershell

# python -m pytest tests agents/deeprl/test\_deeprl.py

# ```

# 

# All 42 tests passed locally. The shared DQN tests create a temporary model, so they do not need my trained checkpoint. Tests requiring PyTorch are skipped when it is unavailable.

# 

# \## Files

# 

# \- `model.py`: CNN, board encoding and symmetry transformations.

# \- `env.py`: wrapper around the shared game engine.

# \- `train.py`: DQN training.

# \- `agent.py`: loads a checkpoint and chooses moves.

# \- `train\_ntuple\_timed.py`: timed training wrapper for P5.

# \- `test\_deeprl.py`: tests for the P6 implementation.

# 

# The trained models, logs, graphs and detailed report are shared separately.

# 

# My DQN has not reached 65,536. The late-game evaluation still needs P8's saved boards and harness. Symmetry augmentation is implemented, but I have not separately measured its effect against training without augmentation.

