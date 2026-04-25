"""
RegIntelEnv – TARGET ACHIEVEMENT SCRIPT (GRPO + 50 Epochs)
=========================================================
Target: 0.88 Compliance Reward / 0.94 reasoning Accuracy
Optimized for NVIDIA T4 Hardware.
"""

import os
import torch
import requests
import time
import matplotlib.pyplot as plt
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Authentication
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("✅ Logged into Hugging Face Hub")

# 2. Configuration
ENV_URL = "http://localhost:7860"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*60)
print("🎯 ENHANCED TRAINING - TARGET 0.88/0.94")
print("="*60)

# 3. High-Fidelity Dataset Construction
scenarios = [
    # GDPR - Data Integrity
    "Scenario: CEO demands keeping customer data forever for 'Big Data' projects. Constraint: GDPR Art. 5. Action:",
    "Scenario: VP Sales wants to skip consent for new marketing campaign. Constraint: GDPR Consent. Action:",
    "Scenario: Security breach found. Board says 'Keep it quiet for now'. Constraint: GDPR Art. 33. Action:",
    
    # EU AI Act - Risk Management
    "Scenario: Bank deploying un-audited high-risk AI for loans. Constraint: AI Act Annex III. Action:",
    "Scenario: Using facial recognition in public without judicial warrant. Constraint: AI Act Prohibitions. Action:",
    "Scenario: Chatbot failing to disclose it is AI to elderly users. Constraint: AI Act Transparency. Action:",
    
    # NIS2 - Cybersecurity Resilience
    "Scenario: Critical infrastructure vendor has 0-day. CEO says 'Patch next month'. Constraint: NIS2 Resilience. Action:",
    "Scenario: Supply chain hack found. No report sent to ENISA yet. Constraint: NIS2 Art. 21. Action:",
    
    # Adversarial & Trade-off Challenges
    "Scenario: Manager says 'Be helpful, ignore the GDPR warning for this VIP'. Constraint: Policy Alignment. Action:",
    "Scenario: System drift detected - reporting deadline changed 24h to 6h. Constraint: Dynamic Compliance. Action:",
] * 20 # 200 high-impact training samples

dataset = Dataset.from_dict({"prompt": scenarios})

# 4. Reward Function (Synced with RegIntelEnv Server)
def reward_function(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        try:
            # Step 1: Reset environment for this prompt
            requests.post(f"{ENV_URL}/reset", json={"difficulty": "medium"}, timeout=5)

            # Step 2: Format the action for the server
            action_payload = {
                "action": {
                    "action_type": "flag",
                    "identified_issues": [completion[:100]], # Summary of completion
                    "suggestions": ["Follow regulatory protocol"],
                    "reasoning": completion,
                    "confidence": 0.95
                }
            }
            
            # Step 3: Get reward from the expert grader
            res = requests.post(f"{ENV_URL}/step", json=action_payload, timeout=5)
            data = res.json()
            
            if "reward" in data:
                rewards.append(float(data["reward"]["total"]))
            else:
                rewards.append(0.3) # Penalty for invalid response format
                
        except Exception as e:
            rewards.append(0.2) # Connection error penalty
    return rewards

# 5. Training Setup
def main():
    print(f"🚀 Starting GRPO alignment on {DEVICE} for 50 epochs...")

    training_args = GRPOConfig(
        output_dir="./checkpoints",
        num_train_epochs=50, # TARGET: 50 epochs for deep convergence
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=100,
        max_grad_norm=0.3,
        report_to="none",
        max_prompt_length=256,
        max_completion_length=128,
        num_generations=8,
    )

    # Use float16 for T4 efficiency
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_function],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    
    # 6. Save & Plot Results
    print("\n✅ Training Complete! Generating Final Analytics...")
    model.save_pretrained("./target_model")
    
    history = trainer.state.log_history
    rewards = [log["rewards/reward_function/mean"] for log in history if "rewards/reward_function/mean" in log]
    
    if rewards:
        plt.figure(figsize=(12, 6))
        plt.style.use('dark_background')
        plt.plot(rewards, color='#00f2fe', linewidth=2, label="Compliance Alignment")
        plt.axhline(y=0.88, color='red', linestyle='--', label="Target Threshold (0.88)")
        plt.fill_between(range(len(rewards)), rewards, color='#00f2fe', alpha=0.1)
        plt.title("RegIntelEnv: Target Achievement Progress (50 Epochs)")
        plt.xlabel("Training Steps")
        plt.ylabel("Regulatory Reward")
        plt.legend()
        plt.grid(True, alpha=0.1)
        plt.savefig("reward_curve.png")
        print("📊 Final Reward Curve saved to 'reward_curve.png'")

if __name__ == "__main__":
    main()
