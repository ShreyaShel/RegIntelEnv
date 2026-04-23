# 📜 RegIntelEnv: Master Technical Guide

## 📡 Architecture Overview
RegIntelEnv is a hybrid intelligence ecosystem designed to bridge the gap between static legal text and autonomous agent action.

### 1. The Environment (OpenEnv)
- **Standardized API**: Implements `/reset`, `/step`, and `/state` for universal compatibility.
- **Dynamic Scenarios**: Features **Regulatory Drift** where laws update in real-time mid-audit.
- **Audit Files**: Real-world company process descriptions served as observations.

### 2. The Multi-Reward System (The Grader)
We verify agents using a 4-dimensional scoring matrix:
1. **Issue ID**: Precise matching against ground-truth violations.
2. **Remediation**: Actionability of proposed fixes.
3. **Legal Accuracy**: Validation of specific regulatory citations (e.g., GDPR Art. 13).
4. **Logic Depth**: Semantic evaluation of the reasoning chain.

### 3. The LLM Judge
- **Semantic Evaluation**: Uses LLM API credits to act as a senior legal counsel.
- **Expert Judge Toggle**: Enable/Disable AI grading via the dashboard settings.
- **Anti-Cheat**: Penalizes keyword stuffing and format violations.

### 4. Coalition Engine
- **Multi-Agent Interaction**: Supports multiple agents working on the same episode.
- **Context Sharing**: Agents can build upon each other's findings (e.g., a Legal Agent flagging a gap and a Tech Agent suggesting a fix).
