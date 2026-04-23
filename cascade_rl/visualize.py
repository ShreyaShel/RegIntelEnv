import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots(csv_path="cascade_metrics.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run training first.")
        return

    df = pd.read_csv(csv_path)
    
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)

    # 1. Reward Curve
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['avg_reward'])
    plt.title("Reward Curve")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.savefig("plots/reward_curve.png")
    plt.close()

    # 2. Coalition Rate
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['coalitions'])
    plt.title("Coalition Count per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Coalitions Formed")
    plt.savefig("plots/coalition_rate.png")
    plt.close()

    # 3. Drift Adaptation rate (cumulative or per ep)
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['drift_adaptations'])
    plt.title("Drift Adaptations per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Successful Adaptations")
    plt.savefig("plots/drift_adaptation.png")
    plt.close()

    # 4. Meta-Learner Spawn Timeline
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['total_agents'])
    plt.title("Meta-Learner Spawn Timeline")
    plt.xlabel("Episodes")
    plt.ylabel("Total Agents in Ecosystem")
    plt.savefig("plots/spawn_timeline.png")
    plt.close()

    print("\n[SUCCESS] All plots saved to 'plots/' directory.")

if __name__ == "__main__":
    generate_plots()
