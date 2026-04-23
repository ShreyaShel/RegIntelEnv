import os
import sys

def check_demo():
    print("\n--- CascadeRL Demo Readiness Check ---")
    
    required_files = [
        "cascade_rl/train.py",
        "cascade_rl/agents.py",
        "cascade_rl/environment.py",
        "cascade_rl/coalition.py",
        "cascade_rl/meta_learner.py",
        "cascade_rl/demo.py",
        "cascade_rl/visualize.py",
        "cascade_rl/demo_fallback.json"
    ]
    
    missing = []
    for f in required_files:
        path = os.path.join(os.path.dirname(__file__), f)
        if not os.path.exists(path):
            missing.append(f)
    
    if missing:
        for m in missing:
            print(f"FAILED: MISSING FILE: {m}")
        print("\nResult: NOT READY FOR DEMO")
        return

    # Check for metrics
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "cascade_metrics.csv")):
        print("WARNING: cascade_metrics.csv not found. Run training to generate charts.")
    
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "plots/")):
        print("WARNING: plots/ folder not found. Run visualize.py to generate charts.")

    print("\nResult: SUCCESS - READY FOR DEMO")

if __name__ == "__main__":
    check_demo()
