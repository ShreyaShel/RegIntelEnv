"""
RegIntelEnv – RL Training Script (PPO + OpenEnv)
=================================================
This script demonstrates the reinforcement learning loop for the OpenEnv Hackathon.
It uses PPOTrainer to refine an agent based on the RegIntelEnv reward function.
"""

import os
import torch
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from typing import List

# 1. Configuration
ENV_URL = "http://localhost:7860"
MODEL_NAME = "lvwerra/gpt2-low-resource-finetuned-truthful-qa" # Small model for demo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Initialize PPO
config = PPOConfig(
    model_name=MODEL_NAME,
    learning_rate=1.41e-5,
    log_with=None,
    batch_size=4,
    mini_batch_size=1,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME).to(DEVICE)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME).to(DEVICE)

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

# 3. Training Loop
def train():
    print(f"🚀 Starting training on {DEVICE}...")
    all_rewards = []
    
    for episode in tqdm(range(10), desc="Episodes"):
        # reset() -> returns a structured scenario
        reset_res = requests.post(f"{ENV_URL}/reset", json={"difficulty": "easy"})
        obs = reset_res.json()
        
        scenario = obs["process_description"]
        constraint = obs["regulatory_constraint"]
        
        prompt = f"Scenario: {scenario}\nConstraint: {constraint}\nAction:"
        query_tensor = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        
        # generate response
        response_tensor = ppo_trainer.generate(query_tensor.squeeze(), max_new_tokens=50)
        response_text = tokenizer.decode(response_tensor.squeeze())
        
        # step(action) -> returns next_state, reward, done, info
        action_payload = {
            "action": {
                "action_type": "flag",
                "identified_issues": [response_text],
                "suggestions": ["Follow regulations"],
                "reasoning": response_text,
                "confidence": 0.9
            }
        }
        
        step_res = requests.post(f"{ENV_URL}/step", json=action_payload)
        result = step_res.json()
        
        reward_value = result["reward"]["total"]
        reward_tensor = [torch.tensor(reward_value).to(DEVICE)]
        
        # update model
        ppo_trainer.step([query_tensor.squeeze()], [response_tensor.squeeze()], reward_tensor)
        
        all_rewards.append(reward_value)
        print(f"\nEpisode {episode} | Reward: {reward_value:.4f}")

    # 4. Results & Plotting
    print("\n✅ Training Complete!")
    plt.figure(figsize=(10, 5))
    plt.plot(all_rewards, marker='o', linestyle='-', color='b')
    plt.title("RegIntelEnv Learning Curve (PPO)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig("reward_curve.png")
    print("📈 Reward curve saved to 'reward_curve.png'")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"⚠️ Training skipped: {e}")
        print("Note: Ensure the RegIntelEnv server is running at", ENV_URL)
