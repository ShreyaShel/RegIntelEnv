import json
import os
import argparse
import logging
from cascade_rl.train import run_cascade_train

def run_readable_demo(safe_mode=False, explain=False):
    print("\n" + "="*50)
    print("      CASCADERL: MULTI-AGENT COMPLIANCE DEMO")
    print("="*50 + "\n")
    
    if safe_mode:
        replay_precomputed(explain)
        print_pitch()
        return

    try:
        # Run live simulation
        # Note: We assume run_cascade_train has been updated to handle its own exceptions or we catch here
        run_cascade_train(num_episodes=2, seed=42, drift_chance=0.8, demo_mode=True)
    except Exception as e:
        print(f"\n[SAFE MODE] Replaying precomputed demo output (Live run failed: {e})")
        replay_precomputed(explain)
    
    print_pitch()

def replay_precomputed(explain):
    fallback_path = os.path.join(os.path.dirname(__file__), "demo_fallback.json")
    if not os.path.exists(fallback_path):
        print("Fatal Error: demo_fallback.json not found!")
        return

    with open(fallback_path, "r") as f:
        data = json.load(f)
    
    for episode in data:
        for event in episode["events"]:
            print(f"[STEP {event['step']}] {event['msg']}")
            if explain and "explain" in event:
                print(f"  -> \"{event['explain']}\"")
        
        s = episode["summary"]
        print("\n=== FINAL SUMMARY ===")
        print(f"Average Reward: {s['avg_reward']:.4f}")
        print(f"Coalition Success Rate: {s['coalition_rate']:.1f}%")
        print(f"Drift Adaptation Rate: {s['drift_adaptation']:.1f}%")
        print(f"Specialists Spawned: {s['spawned']}")
        print("=====================\n")

def print_pitch():
    print("\n=== WHAT YOU JUST SAW ===")
    print("* Agents collaborated (coalitions)")
    print("* System adapted to changing rules (drift)")
    print("* New specialists were created (meta-learning)")
    print("\nThis demonstrates:")
    print("A system that improves itself in a dynamic environment.")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe_mode", action="store_true", help="Replay precomputed demo data")
    parser.add_argument("--explain", action="store_true", help="Show explanations for demo events")
    args = parser.parse_args()
    
    # Mute noisy logs for demo
    logging.getLogger('cascade_rl').setLevel(logging.ERROR)
    
    run_readable_demo(safe_mode=args.safe_mode, explain=args.explain)
