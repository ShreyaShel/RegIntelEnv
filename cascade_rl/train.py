import logging
import json
import random
import argparse
import pandas as pd
from typing import List, Dict, Any
from cascade_rl.agents import AuditorAgent, LawyerAgent, EngineerAgent, ComplianceOfficerAgent
from cascade_rl.environment import CascadeEnvironment
from cascade_rl.coalition import CoalitionEngine
from cascade_rl.meta_learner import MetaLearner

# Suppress noisy logs, keep achievements
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def run_cascade_train(num_episodes: int = 100, seed: int = 42, drift_chance: float = 0.2, demo_mode: bool = False):
    random.seed(seed)
    
    env = CascadeEnvironment(seed=seed)
    coalition_engine = CoalitionEngine()
    
    agents = [
        AuditorAgent("auditor_alpha", seed=seed),
        LawyerAgent("lawyer_alpha", seed=seed),
        EngineerAgent("engineer_alpha", seed=seed),
        ComplianceOfficerAgent("coordinator_alpha", seed=seed)
    ]
    
    meta_learner = MetaLearner(agents)
    metrics_history = []
    
    if not demo_mode:
        print(f"Initializing CascadeRL Training | Episodes: {num_episodes} | Seed: {seed}")

    # Auto-Snapshot Before Training
    snapshot = {
        "num_episodes": num_episodes,
        "seed": seed,
        "drift_chance": drift_chance,
        "initial_agents": [a.agent_id for a in agents]
    }
    with open("cascade_snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2)

    global_total_coalitions = 0
    global_total_drift = 0
    global_total_adapted = 0

    for ep in range(num_episodes):
        ep_seed = seed + ep
        obs_dict = env.reset(difficulty="medium", seed=ep_seed)
        
        done = False
        step_log = []
        ep_coalition_count = 0
        ep_drift_count = 0
        total_drift_adapted = 0
        ep_rewards_sum = 0
        
        while not done:
            step_idx = env._state.step_count
            agent_roles = {a.agent_id: meta_learner._get_agent_role(a.agent_id) for a in agents}
            actions = {}
            
            for agent in agents:
                obs = obs_dict.get(agent.agent_id, obs_dict) if isinstance(obs_dict, dict) else obs_dict
                act = agent.act(obs)
                actions[agent.agent_id] = act
                
                others = [a.agent_id for a in agents if a.agent_id != agent.agent_id]
                if others:
                    target = sorted(others)[ep % len(others)]
                    coalition_engine.propose_coalition(agent.agent_id, [target])

            resolved_coalitions = coalition_engine.resolve_coalitions(coalition_engine.proposals)
            if demo_mode and resolved_coalitions:
                for c in resolved_coalitions:
                    print(f"[STEP {step_idx}] Coalition formed: {' + '.join(sorted(list(c)))}")
            
            modifiers = coalition_engine.merge_and_grade(actions, agent_roles)
            ep_coalition_count += len(resolved_coalitions)

            # Drift
            if env._rng.random() < drift_chance:
                env.inject_drift(step_idx)
                ep_drift_count += 1
                if demo_mode:
                    print(f"[STEP {step_idx}] Drift detected: {env.drift_history[-1]['type'].upper()}")
            
            was_drift = env.drift_active

            obs_dict, raw_rewards, done, info = env.step(actions)
            
            step_total = 0
            for aid, r in raw_rewards.items():
                final_r = r * modifiers.get(aid, 1.0)
                step_total += final_r
                for a in agents:
                    if a.agent_id == aid:
                        a.update(final_r)
                        if was_drift and final_r > r:
                            total_drift_adapted += 1
                            if demo_mode:
                                print(f"[STEP {step_idx}] Adaptation success: +50% reward for {aid}")
            
            ep_rewards_sum += step_total / len(agents)
            step_log.append({"rewards": {aid: r * modifiers.get(aid, 1.0) for aid, r in raw_rewards.items()}})

        # Post-episode
        meta_learner.analyze_episode({"steps": step_log})
        spawned = meta_learner.update()
        if demo_mode and spawned:
            for s in spawned:
                print(f"[EPISODE END] Meta-learner spawned: {s.agent_id}")
        
        metrics_history.append({
            "episode": ep,
            "avg_reward": ep_rewards_sum / (env._state.step_count or 1),
            "coalitions": ep_coalition_count,
            "drift_adaptations": total_drift_adapted,
            "total_agents": len(agents)
        })

        global_total_coalitions += ep_coalition_count
        global_total_drift += ep_drift_count
        global_total_adapted += total_drift_adapted

        if not demo_mode and (ep + 1) % 10 == 0:
            avg_r = metrics_history[-1]["avg_reward"]
            print(f"Ep {ep+1:03d} | Reward: {avg_r:.3f} | Agents: {len(agents)} | Coalitions: {ep_coalition_count}")

    # Export
    with open("cascade_metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)
    pd.DataFrame(metrics_history).to_csv("cascade_metrics.csv", index=False)
    
    if not demo_mode:
        print("\nTraining Complete. Metrics exported to 'cascade_metrics.json' and '.csv'")

    # FINAL SUMMARY
    avg_final_reward = sum(m["avg_reward"] for m in metrics_history) / len(metrics_history)
    coalition_rate = (global_total_coalitions / (len(metrics_history) * 3)) * 100 # Approx steps per ep
    drift_success = (global_total_adapted / max(1, global_total_drift)) * 100

    print("\n=== FINAL SUMMARY ===")
    print(f"Average Reward: {avg_final_reward:.4f}")
    print(f"Coalition Success Rate: {coalition_rate:.1f}% (Normalized)")
    print(f"Drift Adaptation Rate: {drift_success:.1f}%")
    print(f"Specialists Spawned: {len(agents) - 4}")
    print("=====================\n")

    return metrics_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo_mode", action="store_true")
    args = parser.parse_args()
    run_cascade_train(num_episodes=args.episodes, seed=args.seed, demo_mode=args.demo_mode)
