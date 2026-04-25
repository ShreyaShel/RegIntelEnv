"""
RegIntelEnv – RL Training Script (GRPO + OpenEnv)
=================================================
This script demonstrates the reinforcement learning loop for the OpenEnv Hackathon.
It uses GRPOTrainer to refine an agent based on the RegIntelEnv reward function.
"""

import argparse
import json
import os
import torch
import requests
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from huggingface_hub import login
from transformers import AutoTokenizer

# Authenticate with Hugging Face
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("✅ Logged into Hugging Face Hub")
else:
    print("⚠️ HF_TOKEN not found - this may cause authentication errors")

# 1. Configuration
ENV_URL = "http://localhost:7860"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" # Upgraded for GRPO compatibility
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Reward Function for GRPO
def reward_function(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # Interact with the RegIntelEnv server
        try:
            # Note: In a real GRPO loop, we'd batch these requests
            action_payload = {
                "action": {
                    "action_type": "flag",
                    "identified_issues": [completion],
                    "suggestions": ["Follow regulations"],
                    "reasoning": completion,
                    "confidence": 0.9
                }
            }
            res = requests.post(f"{ENV_URL}/step", json=action_payload, timeout=5)
            reward = res.json()["reward"]["total"]
            rewards.append(reward)
        except Exception as e:
            print(f"Error getting reward: {e}")
            rewards.append(0.0)
    return rewards

# 3. Training Setup
def main():
    print(f"🚀 Initializing GRPO Training on {DEVICE}...")

    # Dummy dataset for initialization
    dataset = Dataset.from_dict({
        "prompt": ["Scenario: Customer wants to keep data indefinitely. Constraint: GDPR Art. 5. Action:"]
    })

    training_args = GRPOConfig(
        output_dir="./checkpoints",
        per_device_train_batch_size=2,  # REDUCED from 4 to 2 for T4 GPU
        gradient_accumulation_steps=4,   # ADDED to maintain effective batch size
        learning_rate=1e-5,
        num_train_epochs=1,
        max_prompt_length=256,
        max_completion_length=128,
        num_generations=8, # GRPO uses multiple generations per prompt
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[reward_function],
        args=training_args,
        train_dataset=dataset,
    )

    print("📈 Starting Training...")
    trainer.train()
    print("✅ Training Complete!")

if __name__ == "__main__":
    main()
