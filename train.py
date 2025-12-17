# train.py
import time
import os
import csv
import numpy as np
import torch

from cppo.cppo import CPPO
from cppo.ppo import PPO
from env.safe_hover_env import SafeHoverAviary

GOAL_THRESH = 0.1
GUI_LAST_EPOCHS = 10

def make_env(gui=False, obstacles=None, safe_radius=0.4):
    return SafeHoverAviary(gui=gui, obstacle_positions=obstacles, safe_radius=safe_radius)

def run_training(
    algorithm="cppo",  # "cppo" or "ppo"
    total_epochs=100,
    steps_per_epoch=4096,
    save_path=None,
    log_csv=None
):
    # Set default paths based on algorithm
    if save_path is None:
        save_path = f"results/models/{algorithm}_drone.pt"
    if log_csv is None:
        log_csv = f"results/logs/{algorithm}_training.csv"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_csv), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create environment
    env = make_env(gui=False, obstacles=[[1.0, 0.0, 1.0]], safe_radius=0.4)

    obs, info = env.reset()
    obs = np.array(obs, dtype=np.float32).squeeze()
    obs_dim = obs.shape[0]
    act_dim = int(np.prod(env.action_space.shape))

    # Create agent based on algorithm choice
    use_cppo = (algorithm.lower() == "cppo")
    
    if use_cppo:
        print(f"[INFO] Training with CPPO (Constrained PPO)")
        agent = CPPO(
            obs_dim,
            act_dim,
            device=device,
            steps_per_epoch=steps_per_epoch,
            epochs=total_epochs
        )
    else:
        print(f"[INFO] Training with PPO (Standard)")
        agent = PPO(
            obs_dim,
            act_dim,
            device=device,
            steps_per_epoch=steps_per_epoch,
            epochs=total_epochs
        )
    
    # CSV fieldnames (include all possible fields)
    fieldnames = [
        "epoch",
        "total_reward",
        "avg_reward_per_step",
        "num_episodes",
        "goal_reached",
        "goal_reach_rate",
        "total_cost",
        "avg_cost",
        "lambda",
        "pi_loss",
        "loss_v",
    ]
    
    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    start = time.time()
    
    for epoch in range(total_epochs):
        # Switch to GUI for final epochs
        if epoch == total_epochs - GUI_LAST_EPOCHS:
            print("\n[INFO] Switching to GUI mode for final epochs\n")
            env.close()
            env = make_env(gui=True, obstacles=[[1.0, 0.0, 1.0]], safe_radius=0.4)

        obs, info = env.reset()
        obs = np.array(obs, dtype=np.float32).squeeze()

        ep_reward_total = 0.0
        ep_cost_total = 0.0
        num_episodes = 0
        goal_reached_count = 0
        goal_reached_this_episode = False
        
        agent.start_epoch()
        steps = 0
        
        while steps < steps_per_epoch:
            # Select action (different return signatures for PPO vs CPPO)
            if use_cppo:
                a, logp, v_r, v_c = agent.select_action(obs)
            else:
                a, logp, v_r = agent.select_action(obs)
            
            a_reshaped = a.reshape(1, -1)
            next_obs, reward, terminated, truncated, info = env.step(a_reshaped)
            done = terminated or truncated

            # Track cost (for both PPO and CPPO)
            cost = float(info.get("cost", 0.0))
            drone_pos = np.array(info.get("drone_pos", [np.inf, np.inf, np.inf]))
            goal_pos = env.target_pos
            
            # Goal reached check
            if (not goal_reached_this_episode and
                np.linalg.norm(drone_pos - goal_pos) < GOAL_THRESH):
                goal_reached_count += 1
                goal_reached_this_episode = True

            # Store step (different signatures)
            if use_cppo:
                agent.store_step(obs, a, reward, cost, logp, done, v_r, v_c)
            else:
                agent.store_step(obs, a, reward, logp, done, v_r)
            
            obs = np.array(next_obs, dtype=np.float32).squeeze()
            ep_reward_total += reward
            ep_cost_total += cost
            steps += 1

            # Slow down GUI for visualization
            if epoch >= total_epochs - GUI_LAST_EPOCHS:
                time.sleep(0.01)
            
            if done:
                num_episodes += 1
                goal_reached_this_episode = False
                obs, info = env.reset()
                obs = np.array(obs, dtype=np.float32).squeeze()
        
        # Bootstrap values
        if use_cppo:
            last_v_r, last_v_c = agent.compute_values(obs)
            agent.finish_epoch(last_v_r, last_v_c)
        else:
            last_v_r = agent.compute_values(obs)
            agent.finish_epoch(last_v_r)
        
        stats = agent.update()
        
        # Calculate metrics
        avg_reward_per_step = ep_reward_total / steps
        avg_cost = ep_cost_total / steps
        goal_reach_rate = goal_reached_count / max(1, num_episodes)

        # Write to CSV (PPO will have 0 for lambda, avg_cost in stats)
        with open(log_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({
                "epoch": epoch + 1,
                "total_reward": float(ep_reward_total),
                "avg_reward_per_step": float(avg_reward_per_step),
                "num_episodes": num_episodes,
                "goal_reached": goal_reached_count,
                "goal_reach_rate": float(goal_reach_rate),
                "total_cost": float(ep_cost_total),
                "avg_cost": float(avg_cost),
                "lambda": float(stats.get("lambda", 0.0)),
                "pi_loss": float(stats.get("pi_loss", 0.0)),
                "loss_v": float(stats.get("loss_v", stats.get("loss_vr", 0.0))),
            })
        
        print(
            f"[Epoch {epoch+1}/{total_epochs}] "
            f"reward={ep_reward_total:.1f} "
            f"avg_reward={avg_reward_per_step:.3f} "
            f"episodes={num_episodes} "
            f"goals={goal_reached_count} "
            f"goal_rate={goal_reach_rate:.2f} "
            f"cost={ep_cost_total:.1f} "
            f"avg_cost={avg_cost:.3f} "
            f"lambda={stats.get('lambda', 0.0):.3f}"
        )
        
        if (epoch + 1) % 10 == 0:
            agent.save(save_path)
    
    print(f"\n[INFO] Training finished in {time.time() - start:.1f}s")
    print(f"[INFO] Model saved to: {save_path}")
    print(f"[INFO] Log saved to: {log_csv}")
    env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train drone with PPO or CPPO")
    parser.add_argument("--algorithm", type=str, default="cppo", 
                        choices=["ppo", "cppo"],
                        help="Algorithm to use: ppo or cppo")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=4096,
                        help="Steps per epoch")
    
    args = parser.parse_args()
    
    run_training(
        algorithm=args.algorithm,
        total_epochs=args.epochs,
        steps_per_epoch=args.steps
    )
