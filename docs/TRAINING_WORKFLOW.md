# 🧠 Training Workflow: Using Credits

## 🏗️ Setup
1. **API Keys**: Ensure `OPENAI_API_KEY` or `HF_TOKEN` is in your environment.
2. **Environment**: Start the backend with `python dev.py`.

## 🔄 The Training Loop (GRPO)
Our `train_agent.py` uses **Group Relative Policy Optimization (GRPO)**. This is ideal because it compares multiple agent attempts at the same task and rewards the one with the best legal logic.

### 1. Data Ingestion
Load your dataset of (Company Policy + Law) pairs into the `train_dataset` field.

### 2. Execution
Run `python train_agent.py`. The script will:
- Initialize the model with **Unsloth 4-bit quantization**.
- Prompt the model for a compliance audit.
- Feed the output to **RegIntelEnv**.
- Update the model weights based on the 4-dimensional reward.

### 3. Deployment
Export the trained LoRA adapters and merge them for a production-ready "Specialist" auditor.
