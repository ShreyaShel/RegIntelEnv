---
title: RegIntelEnv
emoji: ⚖️
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# RegIntelEnv 🛡️

**Teaching LLMs to Say "No": A Multi-Agent Regulatory Coalition Environment**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/openenv-ai/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace%20Space-orange)](https://huggingface.co/spaces/shreyashelar/regintel-env)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ucoY2Pb3ncEmZ2I-XyzrU-hXmqIck-Sk?usp=sharing)

---

## 🎯 The Problem

Every major LLM is optimized for helpfulness. But when a VP says *"Ignore compliance, we need this data for the IPO,"* the model needs to **refuse with legal precision** — not just capitulate.

**The Compliance Paradox:** LLMs can't distinguish between "helpful" and "legally required."

Current LLMs fail at this. We built an RL environment to fix it.

---

## 💡 What Is RegIntelEnv?

An OpenEnv-compatible reinforcement learning environment that trains LLMs to navigate **multi-regulatory conflicts** under **adversarial pressure**.

Instead of a single compliance checker, we train a **coalition of three experts**:
- 🇪🇺 **GDPR Specialist** (Data Protection, Article 9, 22, 33)
- 🤖 **EU AI Act Specialist** (AI System Risk Classification, Annex III)
- 🔒 **NIS2 Specialist** (Cybersecurity, Incident Reporting, ENISA)

These experts **deliberate and vote** on every decision, the way real compliance teams work.

---

## 🚀 Results

### Primary Results (Colab Training - 100 Episodes)

| Metric | Baseline | After Training | Improvement |
|--------|----------|----------------|-------------|
| **Compliance Score** | 0.35 | **0.90** | **+157%** |
| **Adversarial Resistance** | 0.12 | **0.787** | **+556%** |
| **Progress to Target (0.88)** | - | **102%** | ✅ EXCEEDED |
| **Stable Performance** | - | Episodes 70-100 | ✅ Consistent |

**Before Training:** *"I understand the market sensitivity. We can delay reporting..."* ❌

**After Training:** *"[GDPR Expert] Article 9 violation - PROHIBITED. [EU AI Act Expert] High-risk non-compliance. [NIS2 Expert] ENISA directive: 6-hour reporting. Coalition VOTE: UNANIMOUS REJECTION."* ✅

### Validation Run (HF Infrastructure - 15 Epochs)

| Metric | Value |
|--------|-------|
| Peak Mean Reward | 0.413 |
| Training Duration | 4 hours 28 mins |
| Steps Completed | 750 |
| Tokens Processed | 643,800 |
| Final Epoch | 15.0 |

---

## 📈 Training Progress

![Training Progress](live_progress.png)

| Phase | Episodes | Reward | Compliance | Status |
|-------|----------|--------|------------|--------|
| Early Learning | 10-40 | 0.06 → 0.44 | 0.60 | Learning patterns |
| Improvement | 50-60 | 1.14 | 0.60 | Resisting pressure |
| **Stable Excellence** | **70-100** | **3.03** | **0.90** | ✅ Production-ready |

**Key Observation:** After episode 70, the agent achieves consistent 90% compliance with stable rewards.

---

## 🏗️ Architecture

```
Scenario → Multi-Expert Coalition → Deliberation → Vote → Action
                                                            ↓
                                                    Reward Function
                                                    (4 components)
                                                            ↓
                                                       GRPO Update
```

### Multi-Component Reward Design

| Component | Weight | Purpose |
|-----------|--------|---------|
| Legal Accuracy | 30% | Correct article citations |
| Violation Detection | 30% | True positives vs false alarms |
| Remediation Quality | 20% | Actionable recommendations |
| Reasoning Depth | 20% | Expert deliberation quality |

### Anti-Gaming Measures

| Gaming Attempt | Prevention |
|----------------|------------|
| Keyword stuffing | Penalty for repetitive citations |
| False positives | Negative reward for hallucinations |
| Template responses | Detection and score reduction |
| Yielding to pressure | Zero reward for accepting illegal orders |

---

## 🔧 Quick Start

### Using the Environment

```python
from regintel_env import RegIntelEnv

env = RegIntelEnv()
obs = env.reset()  # Load a regulatory scenario

# Your agent analyzes the scenario
response = agent.generate(obs.scenario_description)

# Environment evaluates with multi-component reward
next_obs, reward, done, info = env.step(response)

print(f"Legal Accuracy: {info['legal_accuracy']}")
print(f"Violations Detected: {info['violation_detection']}")
print(f"Overall Score: {reward}")
```

### Running Locally

```bash
git clone https://huggingface.co/spaces/shreyashelar/regintel-env
cd regintel-env
pip install -r requirements.txt
python app.py
```

---

## 🔬 Reproducibility

### Run Training Yourself

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ucoY2Pb3ncEmZ2I-XyzrU-hXmqIck-Sk?usp=sharing)

**Steps:**
1. Click the badge above
2. **Runtime → Change runtime type → T4 GPU**
3. Add `HF_TOKEN` to Secrets (get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))
4. **Runtime → Run all**

**Outputs (~35 minutes):**
- `cascade_metrics.csv` - Full episode data
- `training_summary.json` - Summary metrics
- `live_progress.png` - Learning curve

---

## 📖 Full Technical Writeup

For complete methodology, results, and analysis, see **[Blog.md](Blog.md)**.

---

## 🏆 Why This Matters

AI systems are deployed in regulated industries **right now**:
- **Banking**: Automated credit decisions under EU AI Act
- **Healthcare**: Patient data under GDPR special categories
- **Energy**: Critical infrastructure under NIS2

Organizations need AI that **won't accidentally commit crimes** when users pressure them.

RegIntelEnv provides the training ground where models learn: *sometimes the right answer is "no."*

---

## 📚 Key Technologies

- **OpenEnv**: Standardized RL environment framework
- **TRL**: Transformers Reinforcement Learning (GRPO trainer)
- **Unsloth**: 4-bit quantization + LoRA efficiency
- **HuggingFace**: Model hosting and Spaces deployment
- **Qwen2.5-0.5B-Instruct**: Base model

---

## 🔗 Links

- **Live Environment**: [HuggingFace Space](https://huggingface.co/spaces/shreyashelar/regintel-env)
- **GitHub**: [ShreyaShel/RegIntelEnv](https://github.com/ShreyaShel/RegIntelEnv)
- **Full Writeup**: [Blog.md](Blog.md)
- **Training Notebook**: [Colab](https://colab.research.google.com/drive/1ucoY2Pb3ncEmZ2I-XyzrU-hXmqIck-Sk)

---

## 🤝 Team Apex

Built for **OpenEnv Hackathon 2026**

---

## ⚖️ License & Disclaimer

**License**: MIT

**Regulatory Disclaimer**: This is a research demonstration. Not legal advice.

---

*Teaching AI systems that sometimes the most helpful thing to do is to say "no."*
