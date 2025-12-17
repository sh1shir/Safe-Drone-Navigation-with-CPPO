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

def plot_training(log_path, out_dir="results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    rows = read_csv(log_path)

    if len(rows) == 0:
        print(f"[WARNING] No rows found in CSV: {log_path}")
        return

    epochs = [int(r["epoch"]) for r in rows]
    rewards = [float(r["avg_reward_per_step"]) for r in rows]
    costs = [float(r["avg_cost"]) for r in rows]
    lambdas = [float(r["lambda"]) for r in rows]

    goals = [int(r.get("goal_reached", 0)) for r in rows]
    goal_rates = [float(r.get("goal_reach_rate", 0.0)) for r in rows]

    # --- Reward ---
    plt.figure(figsize=(7,4))
    plt.plot(epochs, rewards, linewidth=2, color='b')
    plt.xlabel("Epoch")
    plt.ylabel("Average reward per step")
    plt.title("Reward per epoch")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "reward1.png"))
    plt.close()

    # --- Cost ---
    plt.figure(figsize=(7,4))
    plt.plot(epochs, costs, linewidth=2, color='r')
    plt.xlabel("Epoch")
    plt.ylabel("Average cost")
    plt.title("Average cost per epoch")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "cost1.png"))
    plt.close()

    # --- Lambda ---
    plt.figure(figsize=(7,4))
    plt.plot(epochs, lambdas, linewidth=2, color='r')
    plt.xlabel("Epoch")
    plt.ylabel("Lambda (Lagrange multiplier)")
    plt.title("Lambda over training")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "lambda1.png"))
    plt.close()

    # --- Goals reached ---
    plt.figure(figsize=(7,4))
    plt.plot(epochs, goals, linewidth=2, color='m')
    plt.xlabel("Epoch")
    plt.ylabel("Goals reached")
    plt.title("Number of successful episodes per epoch")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "goals_reached1.png"))
    plt.close()

    # --- Goal reach rate ---
    plt.figure(figsize=(7,4))
    plt.plot(epochs, goal_rates, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Goal reach rate")
    plt.title("Goal reach rate per epoch")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "goal_rate1.png"))
    plt.close()

    print(f"[OK] Saved plots to: {out_dir}")

# -------------------------------------------------------
# MAIN ENTRY POINT
# -------------------------------------------------------
if __name__ == "__main__":
    DEFAULT_CSV = "results/logs/training1.csv"

    if not os.path.exists(DEFAULT_CSV):
        print(f"[ERROR] CSV not found: {DEFAULT_CSV}")
    else:
        plot_training(DEFAULT_CSV)
