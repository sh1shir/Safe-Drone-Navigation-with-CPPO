# utils/compare_algorithms.py
import csv
import matplotlib.pyplot as plt
import numpy as np
import os

def read_csv(log_path):
    rows = []
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def moving_average(data, window=10):
    """Compute moving average for smoothing."""
    if len(data) < window:
        window = len(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')

def compare_algorithms(ppo_csv, cppo_csv, out_dir="results/comparison"):
    os.makedirs(out_dir, exist_ok=True)
    
    # Read data
    ppo_data = read_csv(ppo_csv)
    cppo_data = read_csv(cppo_csv)
    
    if len(ppo_data) == 0 or len(cppo_data) == 0:
        print("[WARNING] One or both CSV files are empty")
        return
    
    # Extract PPO data
    ppo_epochs = [int(r["epoch"]) for r in ppo_data]
    ppo_avg_rewards = [float(r["avg_reward_per_step"]) for r in ppo_data]
    ppo_avg_costs = [float(r["avg_cost"]) for r in ppo_data]
    ppo_goal_rates = [float(r["goal_reach_rate"]) for r in ppo_data]
    
    # Extract CPPO data
    cppo_epochs = [int(r["epoch"]) for r in cppo_data]
    cppo_avg_rewards = [float(r["avg_reward_per_step"]) for r in cppo_data]
    cppo_avg_costs = [float(r["avg_cost"]) for r in cppo_data]
    cppo_goal_rates = [float(r["goal_reach_rate"]) for r in cppo_data]
    cppo_lambdas = [float(r["lambda"]) for r in cppo_data]
    
    # Create comprehensive comparison plot (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    smooth_window = 10
    
    # 1. Average Reward per Step
    ax = axes[0, 0]
    ax.plot(ppo_epochs, ppo_avg_rewards, alpha=0.3, color='blue', label='PPO (raw)')
    ax.plot(cppo_epochs, cppo_avg_rewards, alpha=0.3, color='red', label='CPPO (raw)')
    if len(ppo_avg_rewards) >= smooth_window:
        smoothed = moving_average(ppo_avg_rewards, smooth_window)
        ax.plot(ppo_epochs[smooth_window-1:], smoothed, linewidth=2, color='blue', label='PPO (smoothed)')
    if len(cppo_avg_rewards) >= smooth_window:
        smoothed = moving_average(cppo_avg_rewards, smooth_window)
        ax.plot(cppo_epochs[smooth_window-1:], smoothed, linewidth=2, color='red', label='CPPO (smoothed)')
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Avg Reward per Step", fontsize=11)
    ax.set_title("Reward Performance", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Average Cost per Step (Safety Violations)
    ax = axes[0, 1]
    ax.plot(ppo_epochs, ppo_avg_costs, alpha=0.3, color='blue', label='PPO (raw)')
    ax.plot(cppo_epochs, cppo_avg_costs, alpha=0.3, color='red', label='CPPO (raw)')
    if len(ppo_avg_costs) >= smooth_window:
        smoothed = moving_average(ppo_avg_costs, smooth_window)
        ax.plot(ppo_epochs[smooth_window-1:], smoothed, linewidth=2, color='blue', label='PPO (smoothed)')
    if len(cppo_avg_costs) >= smooth_window:
        smoothed = moving_average(cppo_avg_costs, smooth_window)
        ax.plot(cppo_epochs[smooth_window-1:], smoothed, linewidth=2, color='red', label='CPPO (smoothed)')
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Avg Cost per Step", fontsize=11)
    ax.set_title("Safety Violations Rate", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Goal Reach Rate
    ax = axes[1, 0]
    ax.plot(ppo_epochs, ppo_goal_rates, alpha=0.3, color='blue', label='PPO (raw)')
    ax.plot(cppo_epochs, cppo_goal_rates, alpha=0.3, color='red', label='CPPO (raw)')
    if len(ppo_goal_rates) >= smooth_window:
        smoothed = moving_average(ppo_goal_rates, smooth_window)
        ax.plot(ppo_epochs[smooth_window-1:], smoothed, linewidth=2, color='blue', label='PPO (smoothed)')
    if len(cppo_goal_rates) >= smooth_window:
        smoothed = moving_average(cppo_goal_rates, smooth_window)
        ax.plot(cppo_epochs[smooth_window-1:], smoothed, linewidth=2, color='red', label='CPPO (smoothed)')
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Goal Reach Rate", fontsize=11)
    ax.set_title("Task Success Rate", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # 4. Lambda (CPPO only)
    ax = axes[1, 1]
    ax.plot(cppo_epochs, cppo_lambdas, linewidth=2, color='orange', label='CPPO Lambda')
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Lambda (Lagrange Multiplier)", fontsize=11)
    ax.set_title("CPPO Safety Enforcement", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("PPO vs CPPO: Performance Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ppo_vs_cppo_comparison.png"), dpi=150)
    plt.close()
    
    # Calculate final performance (last 20 epochs) for summary
    ppo_final_reward = np.mean(ppo_avg_rewards[-20:]) if len(ppo_avg_rewards) >= 20 else np.mean(ppo_avg_rewards)
    cppo_final_reward = np.mean(cppo_avg_rewards[-20:]) if len(cppo_avg_rewards) >= 20 else np.mean(cppo_avg_rewards)
    
    ppo_final_cost = np.mean(ppo_avg_costs[-20:]) if len(ppo_avg_costs) >= 20 else np.mean(ppo_avg_costs)
    cppo_final_cost = np.mean(cppo_avg_costs[-20:]) if len(cppo_avg_costs) >= 20 else np.mean(cppo_avg_costs)
    
    ppo_final_goal = np.mean(ppo_goal_rates[-20:]) if len(ppo_goal_rates) >= 20 else np.mean(ppo_goal_rates)
    cppo_final_goal = np.mean(cppo_goal_rates[-20:]) if len(cppo_goal_rates) >= 20 else np.mean(cppo_goal_rates)
    
    cost_reduction = ((ppo_final_cost - cppo_final_cost) / ppo_final_cost * 100) if ppo_final_cost > 0 else 0
    
    print(f"\n[OK] Comparison plot saved to: {out_dir}/ppo_vs_cppo_comparison.png")
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    print(f"PPO  - Final Reward: {ppo_final_reward:.3f}, Cost: {ppo_final_cost:.4f}, Goal Rate: {ppo_final_goal:.2%}")
    print(f"CPPO - Final Reward: {cppo_final_reward:.3f}, Cost: {cppo_final_cost:.4f}, Goal Rate: {cppo_final_goal:.2%}")
    print(f"\nCost Reduction: {cost_reduction:.1f}%")
    print("="*50)

if __name__ == "__main__":
    PPO_CSV = "results/logs/ppo_training.csv"
    CPPO_CSV = "results/logs/cppo_training.csv"
    
    if not os.path.exists(PPO_CSV):
        print(f"[ERROR] PPO CSV not found: {PPO_CSV}")
    elif not os.path.exists(CPPO_CSV):
        print(f"[ERROR] CPPO CSV not found: {CPPO_CSV}")
    else:
        compare_algorithms(PPO_CSV, CPPO_CSV)
