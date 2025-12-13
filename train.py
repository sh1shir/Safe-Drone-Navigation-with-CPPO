# train.py
import time
import os
import csv
import numpy as np
import torch

from cppo.cppo import CPPO
from env.safe_hover_env import SafeHoverAviary

def make_env(gui=False, obstacles=None, safe_radius=0.4):
    return SafeHoverAviary(gui=gui, obstacle_positions=obstacles, safe_radius=safe_radius)

def run_training(total_epochs=100, steps_per_epoch=4096,
                 save_path="results/models/cppo_drone1.pt",
                 log_csv="results/logs/training1.csv"):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_csv), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(gui=False, obstacles=[[1.0, 0.0, 1.0]], safe_radius=0.4)
    obs, info = env.reset()
    obs = np.array(obs, dtype=np.float32).squeeze()
    obs_dim = obs.shape[0]
    act_dim = int(np.prod(env.action_space.shape))
    agent = CPPO(obs_dim, act_dim, device=device, steps_per_epoch=steps_per_epoch, epochs=total_epochs)
    
    fieldnames = ["epoch", "total_reward", "avg_reward_per_step", "num_episodes", 
                  "avg_cost", "lambda", "pi_loss", "loss_vr", "loss_vc"]
    
    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    start = time.time()
    
    for epoch in range(total_epochs):
        obs, info = env.reset()
        obs = np.array(obs, dtype=np.float32).squeeze()
        
        ep_reward_total = 0.0
        ep_cost_total = 0.0
        num_episodes = 0
        
        agent.start_epoch()
        steps = 0
        
        while steps < steps_per_epoch:
            a, logp, v_r, v_c = agent.select_action(obs)
            a_reshaped = a.reshape(1, -1)
            next_obs, reward, terminated, truncated, info = env.step(a_reshaped)
            done = terminated or truncated
            cost = float(info.get("cost", 0.0))
            
            agent.store_step(obs, a, reward, cost, logp, done, v_r, v_c)
            obs = np.array(next_obs, dtype=np.float32).squeeze()
            
            ep_reward_total += reward
            ep_cost_total += cost
            steps += 1
            
            if done:
                num_episodes += 1
                obs, info = env.reset()
                obs = np.array(obs, dtype=np.float32).squeeze()
        
        # bootstrap final values
        last_v_r, last_v_c = agent.compute_values(obs)
        agent.finish_epoch(last_v_r, last_v_c)
        stats = agent.update()
        
        # Calculate metrics
        avg_reward_per_step = ep_reward_total / steps
        
        # Write CSV with more detailed metrics
        with open(log_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({
                "epoch": epoch+1,
                "total_reward": float(ep_reward_total),
                "avg_reward_per_step": float(avg_reward_per_step),
                "num_episodes": num_episodes,
                "avg_cost": float(stats.get("avg_cost", 0.0)),
                "lambda": float(stats.get("lambda", 0.0)),
                "pi_loss": float(stats.get("pi_loss", 0.0)),
                "loss_vr": float(stats.get("loss_vr", 0.0)),
                "loss_vc": float(stats.get("loss_vc", 0.0)),
            })
        
        print(f"[Epoch {epoch+1}/{total_epochs}] "
              f"total_reward={ep_reward_total:.1f} "
              f"avg_reward={avg_reward_per_step:.3f} "
              f"episodes={num_episodes} "
              f"cost={ep_cost_total:.1f} "
              f"lambda={stats['lambda']:.3f}")
        
        if (epoch + 1) % 10 == 0:
            agent.save(save_path)
    
    print("Training finished in {:.1f}s".format(time.time() - start))
    env.close()

if __name__ == "__main__":
    run_training(total_epochs=100, steps_per_epoch=4096)
