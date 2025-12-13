# scripts/evaluate.py
import argparse
import os
import time
import numpy as np
import torch

from cppo.cppo import CPPO
from env.safe_hover_env import SafeHoverAviary

def evaluate(model_path, episodes=5, gui=True, obstacle_positions=None, safe_radius=0.4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = SafeHoverAviary(gui=gui, obstacle_positions=obstacle_positions, safe_radius=safe_radius)

    # sample obs/action dims
    obs = env.reset()
    obs = np.array(obs, dtype=np.float32)
    obs_dim = obs.shape[0]
    act_dim = env.action_space.shape[0]

    agent = CPPO(obs_dim, act_dim, device=device, steps_per_epoch=1024, epochs=1)
    agent.load(model_path, map_location=device)

    rewards = []
    costs = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_r = 0.0
        ep_cost = 0.0
        step = 0
        while True:
            a, _, _, _ = agent.select_action(obs)  # deterministic policy could be obtained by mean; here we sample
            next_obs, reward, terminated, truncated, info = env.step(a)
            ep_r += reward
            ep_cost += info.get("cost", 0.0)
            step += 1
            obs = np.array(next_obs, dtype=np.float32)
            if terminated or truncated or step > 1000:
                break

        rewards.append(ep_r)
        costs.append(ep_cost)
        print(f"Episode {ep+1}: reward={ep_r:.2f}, cost={ep_cost:.2f}")

    print("=== Evaluation summary ===")
    print(f"Avg reward: {np.mean(rewards):.2f}  Std: {np.std(rewards):.2f}")
    print(f"Avg cost:   {np.mean(costs):.3f}  Std: {np.std(costs):.3f}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to checkpoint .pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    evaluate(args.model, episodes=args.episodes, gui=args.gui)
