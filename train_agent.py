"""
RegIntelEnv – RL Training Script (GRPO + Unsloth)
=================================================
This specialized script demonstrates how to refine a base LLM into a 
'Regulatory Specialist' using Reinforcement Learning. 

Key Techniques:
1. Unsloth Optimization: Enables high-performance 4-bit training on minimal compute.
2. GRPOTrainer: Implements Group Relative Policy Optimization for rule-based verification.
3. OpenEnv Integration: Directly connects the training loop to the RegIntelEnv API.
"""

import os
import torch
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel, PatchFastRL
from models import RegAction, StepResult
import requests

# Patching for RL efficiency
PatchFastRL("GRPO", FastLanguageModel)

# 1. Configuration
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
ENV_URL = "http://localhost:7860"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 16

# 2. Load Model & Tokenizer (Unsloth optimized)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=LORA_RANK,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

import re
import json

# 3. Define Reward Function (OpenEnv Bridge)
def reg_intel_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Connects the model's output to the RegIntelEnv environment and returns the reward.
    Uses robust JSON extraction to parse agent actions.
    """
    rewards = []
    
    for prompt, completion in zip(prompts, completions):
        try:
            # 1. Start a fresh episode
            reset_res = requests.post(f"{ENV_URL}/reset", json={"difficulty": "medium"})
            obs = reset_res.json()
            
            # 2. Robust Action Parsing (JSON Extraction)
            # We look for a JSON block in the model's output
            json_match = re.search(r"\{.*\}", completion, re.DOTALL)
            if json_match:
                try:
                    action_data = json.loads(json_match.group())
                except:
                    action_data = {"action_type": "analyze", "reasoning": completion}
            else:
                # Fallback: Treat raw text as reasoning
                action_data = {
                    "action_type": "flag",
                    "identified_issues": ["Unstructured Output"],
                    "suggestions": ["Follow JSON protocol"],
                    "reasoning": completion,
                    "confidence": 0.5
                }
            
            # 3. Step the environment
            step_res = requests.post(f"{ENV_URL}/step", json={"action": action_data})
            result = step_res.json()
            
            # 4. Extract the reward (Scale it for RL optimization)
            reward_score = result["reward"]["total"]
            
            # Apply format penalty if it didn't follow JSON (Rule 7)
            if not json_match:
                reward_score *= 0.5
                
            rewards.append(float(reward_score))
            print(f"Step Reward: {reward_score:.4f} | Action: {action_data.get('action_type')}")
            
        except Exception as e:
            print(f"⚠️ Reward Loop Error: {e}")
            rewards.append(0.0)
            
    return rewards

# 4. Training Loop Setup (GRPO)
training_args = GRPOConfig(
    output_dir="./outputs",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    logging_steps=5,
    bf16=True,
    report_to="tensorboard",
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reg_intel_reward_func],
    args=training_args,
    train_dataset=None, # Load your 'Law-Policy' prompt pairs here
    tokenizer=tokenizer,
)

if __name__ == "__main__":
    print("\n" + "═"*50)
    print("🚀 REG-INTEL-ENV RL TRAINER INITIALIZED")
    print(f"📍 Target: {ENV_URL}")
    print(f"🧠 Base Model: {MODEL_NAME}")
    print("═"*50 + "\n")
    
    print("Note: To begin training, provide a dataset of (Question/Law/Policy) prompts.")
    print("Example: Load from Hugging Face Datasets or a local JSONL file.")
    # trainer.train()
