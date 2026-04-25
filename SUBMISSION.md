# Hackathon Submission: RegIntelEnv

## 1. Project Vision
RegIntelEnv solves the "Compliance Paradox": How do we build AI that is both helpful and legally sound? By treating regulatory compliance as a reinforcement learning problem, we can train models that don't just follow rules, but understand the *trade-offs* between business utility and legal safety.

## 2. Core Themes Implemented

### A) Decision-Making Under Constraints
We implemented three high-stakes scenarios:
1. **GDPR/Customer Analytics**: Balancing personalized marketing with data minimization.
2. **AI Act/Hiring**: Balancing predictive efficiency with anti-bias mandates.
3. **NIS2/Energy Infrastructure**: Balancing rapid incident response with strict evidence preservation.

### B) OpenEnv Protocol
Our environment is 100% compliant with the OpenEnv specification:
- **`reset()`**: Initializes a fresh regulatory scenario with randomized "drift" events.
- **`step()`**: Processes agent actions through a multi-dimensional reward function.
- **`state()`**: Provides a full snapshot of the environment's legal and operational state.

## 3. The "WOW" Factor: Neural Dashboard
We built a premium monitoring interface that visualizes the "thinking" of the agent. Judges can see the **Integrity Score** shift in real-time as the agent navigates legal constraints.

## 4. Technical Excellence
- **Reward Continuity**: No binary pass/fail. Agents get partial credit for identifying violations even if their remediation is weak, allowing for smoother gradient updates.
- **Expert Judge**: Integrated LLM-based grading to catch semantic nuances that keyword matching misses.
- **TRL Integration**: Fully functional PPO training script included.

## 5. Evidence
- [README.md](file:///c:/Users/shrey/Hackaton/RegIntelEnv/README.md)
- [Training Script](file:///c:/Users/shrey/Hackaton/RegIntelEnv/train_agent.py)
- [Reward Function](file:///c:/Users/shrey/Hackaton/RegIntelEnv/grader.py)
- [Task Definitions](file:///c:/Users/shrey/Hackaton/RegIntelEnv/tasks.py)
