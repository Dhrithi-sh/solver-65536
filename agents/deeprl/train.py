"""Train a small Double DQN using legal actions and replay symmetry augmentation."""
import argparse
from collections import deque
import csv
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .env import GameEnv
from .model import QNetwork, action_map, transform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--minutes", type=float, default=0, help="Optional wall-clock training limit")
    parser.add_argument("--seed", type=int, default=1000000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--epsilon-steps", type=int, default=80000)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--checkpoint", default="weights/dqn.pt")
    parser.add_argument("--log", default="logs/dqn_training.csv")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.buffer_size, args.epsilon_steps) <= 0:
        parser.error("Steps, batch size, buffer size and epsilon steps must be positive")
    if args.buffer_size < args.batch_size or args.warmup < 0 or args.minutes < 0:
        parser.error("Invalid replay size, warmup or time limit")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)  # Small networks are often slower with many CPU threads.
    torch.use_deterministic_algorithms(True)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    online, target = QNetwork().to(device), QNetwork().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
    replay = deque(maxlen=args.buffer_size)
    env = GameEnv(args.seed)
    state, info = env.reset(args.seed)
    episode = updates = steps = 0
    last_loss = float("nan")
    start = time.perf_counter()
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        torch.save({"model": online.state_dict(), "steps": steps, "updates": updates,
                    "episodes": episode, "elapsed_seconds": time.perf_counter() - start,
                    "config": vars(args), "reward_scale": 128.0}, checkpoint)

    with log.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "episode", "train_seed", "score", "max_tile", "epsilon", "loss", "seconds"])
        try:
            for steps in range(1, args.steps + 1):
                epsilon = max(0.05, 1 - 0.95 * (steps - 1) / args.epsilon_steps)
                legal = np.flatnonzero(info["action_mask"]).tolist()
                if rng.random() < epsilon:
                    action = rng.choice(legal)
                else:
                    with torch.no_grad():
                        q = online(torch.from_numpy(state).unsqueeze(0).to(device))[0]
                        mask = torch.tensor(info["action_mask"], device=device)
                        action = int(q.masked_fill(~mask, -torch.inf).argmax())
                next_state, reward, done, _, next_info = env.step(action)
                # A constant scale preserves the raw-score objective.
                replay.append((state, action, reward / 128.0, next_state,
                               done, next_info["action_mask"]))
                state, info = next_state, next_info
                if len(replay) >= args.batch_size and steps >= args.warmup and steps % 4 == 0:
                    batch = rng.sample(list(replay), args.batch_size)
                    states, actions, rewards, next_states, dones, masks = [], [], [], [], [], []
                    for s, a, r, ns, terminal, mask in batch:
                        sym = 0 if args.no_augmentation else rng.randrange(8)
                        mapping = action_map(sym)
                        new_mask = np.zeros(4, dtype=bool)
                        new_mask[mapping] = mask
                        states.append(transform(s, sym))
                        next_states.append(transform(ns, sym))
                        actions.append(mapping[a])
                        rewards.append(r)
                        dones.append(terminal)
                        masks.append(new_mask)
                    s = torch.tensor(np.stack(states), device=device)
                    ns = torch.tensor(np.stack(next_states), device=device)
                    a = torch.tensor(actions, device=device).long()
                    r = torch.tensor(rewards, device=device)
                    terminal = torch.tensor(dones, device=device)
                    mask = torch.tensor(np.stack(masks), device=device)
                    predicted = online(s).gather(1, a[:, None]).squeeze(1)
                    with torch.no_grad():
                        next_actions = online(ns).masked_fill(~mask, -torch.inf).argmax(1)
                        next_q = target(ns).gather(1, next_actions[:, None]).squeeze(1)
                        next_q = next_q.masked_fill(terminal | ~mask.any(1), 0)
                        expected = r + 0.99 * next_q
                    loss = F.smooth_l1_loss(predicted, expected)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(online.parameters(), 10)
                    optimizer.step()
                    updates += 1
                    last_loss = float(loss.detach())
                    if updates % 250 == 0:
                        target.load_state_dict(online.state_dict())
                if done:
                    writer.writerow([steps, episode, args.seed + episode, info["score"],
                                     info["max_tile"], epsilon, last_loss, time.perf_counter() - start])
                    file.flush()
                    episode += 1
                    state, info = env.reset(args.seed + episode)
                if steps % 5000 == 0:
                    save()
                    print(f"steps={steps} episodes={episode} epsilon={epsilon:.3f} loss={last_loss:.4f}", flush=True)
                if args.minutes and time.perf_counter() - start >= args.minutes * 60:
                    break
        except KeyboardInterrupt:
            print("Interrupted; saving current model.")
        finally:
            save()
    print(f"Saved {checkpoint}; steps={steps}, updates={updates}, episodes={episode}")
    print("Training logs are diagnostics, not benchmark results. Evaluate through P8's harness.")


if __name__ == "__main__":
    main()
