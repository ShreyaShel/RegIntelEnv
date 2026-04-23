import random
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

# We're pulling in the base environment and models to extend them 
# with more advanced features like multi-agent support and drift.
from server.reg_intel_environment import RegIntelEnvironment
from models import RegAction, ActionType, DifficultyLevel, ComplianceStatus

logger = logging.getLogger(__name__)

class CascadeEnvironment(RegIntelEnvironment):
    """
    CascadeEnvironment is where we add the "real world" complexity.
    In a hackathon or production setting, things aren't static. 
    This class handles multi-agent coordination and simulates "Regulatory Drift" 
    to see if agents can keep up when the rules change mid-game.
    """
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.set_seed(seed)
        self.drift_active = False
        self.drift_history = []

    def set_seed(self, seed: Optional[int]):
        """Keep things deterministic so we can reproduce our results."""
        self.seed = seed
        self._rng = random.Random(seed)

    def inject_drift(self, step: int):
        """
        This is the "curveball" function. 
        It simulates a regulatory update or a sudden corporate change.
        If an agent is just following a script, they'll fail here. 
        A good agent will notice the text change and adapt.
        """
        drift_types = ["amendment", "new_regulation", "scope_change"]
        dtype = self._rng.choice(drift_types)
        self.drift_active = True
        
        # We physically update the task description or regulation name.
        # It's not just a flag; the text the agent reads actually changes.
        if dtype == "amendment":
            self._task.process_description += f"\n[URGENT AMENDMENT S{step}] All identity data must be purged after 30 days."
        elif dtype == "new_regulation":
            self._task.regulation_name += " [REVISED 2026]"
            self._task.regulation_summary += "\nMandatory: AI usage must be explicitly disclosed in UI."
        else:
            self._task.company_name += " (M&A Group)"
            self._task.process_description += "\n[NEW SCOPE] Analysis must include acquisition targets."

        # We log this so we can see when the "rules" changed in our logs.
        self._state.history.append({"drift_event": dtype, "step": step})
        self.drift_history.append({"step": step, "type": dtype})
        logger.info(f"DANGER: Regulatory Drift '{dtype}' injected at step {step}")

    def reset(self, **kwargs) -> Any:
        """Start fresh. Clear any drift history from the previous run."""
        seed = kwargs.get("seed")
        if seed is not None:
            self.set_seed(seed)
        self.drift_active = False
        self.drift_history = []
        return super().reset(**kwargs)

    def step(self, actions_dict: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        """
        The core of the multi-agent logic. 
        We take actions from multiple agents, validate them, merge them into 
         a single consensus, and then see how the environment reacts.
        """
        if not isinstance(actions_dict, dict):
            actions_dict = {}

        # 1. First, make sure every agent sent something sensible. 
        #    If they didn't, we give them a safe "continue" move so the simulation doesn't crash.
        validated = self._safe_validate(actions_dict)
        
        # 2. We merge all agent outputs. If one agent flags an issue, 
        #    it's added to the collective finding.
        merged_action = self._merge_multi_agent(validated)
        
        # 3. Pass the merged action to the underlying environment.
        try:
            base_result = super().step(merged_action)
        except Exception as e:
            logger.error(f"FATAL ENV STEP ERROR: {e}")
            return {}, {}, True, {"error": str(e)}

        # 4. Map the global reward back to each agent.
        rewards = {aid: base_result.reward.total for aid in validated}
        
        # 5. Reward/Penalty for adaptation:
        #    If drift happened in the last step, agents MUST change their output.
        #    If they just repeat what they said before, they get a heavy penalty.
        if self.drift_active:
            for aid, act in validated.items():
                findings = act.get("findings", [])
                suggestions = act.get("suggestions", [])
                if not findings and not suggestions:
                    # They ignored the change. Not good for compliance!
                    rewards[aid] *= 0.7 
                else:
                    # They adapted quickly. This is what we want.
                    rewards[aid] *= 1.5 
            self.drift_active = False # Clear the flag once we've checked their response

        # 6. Build the observations for each agent.
        observations = {}
        for aid in validated:
            obs_data = base_result.observation.model_dump()
            # We let the agents know if a drift happened so they have a chance to react.
            obs_data["metadata"]["drift_occurred"] = (len(self.drift_history) > 0)
            obs_data["metadata"]["last_drift_step"] = self.drift_history[-1]["step"] if self.drift_history else -1
            observations[aid] = obs_data

        return observations, rewards, base_result.done, base_result.info

    def _safe_validate(self, actions_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """A simple safety layer to prevent crashes if an agent sends garbage."""
        valid = {}
        default = {"findings": [], "suggestions": [], "references": [], "action": "continue"}
        for aid, act in actions_dict.items():
            if not isinstance(act, dict):
                valid[aid] = default
            else:
                valid[aid] = {k: act.get(k, default[k]) for k in default}
        return valid

    def _merge_multi_agent(self, actions_dict: Dict[str, Dict[str, Any]]) -> RegAction:
        """
        Consolidates the wisdom of the crowd. 
        We take all unique findings and suggestions from every agent to 
        form a single 'Consensus Action' against the environment.
        """
        sum_issues = []
        sum_suggestions = []
        sum_refs = []
        finalize = False
        
        for data in actions_dict.values():
            sum_issues.extend(data.get("findings", []))
            sum_suggestions.extend(data.get("suggestions", []))
            sum_refs.extend(data.get("references", []))
            if data.get("action") == "finalize":
                finalize = True
        
        # We return a single RegAction that the base grader can understand.
        return RegAction(
            action_type=ActionType.CONCLUDE if finalize else ActionType.FLAG,
            identified_issues=list(set(sum_issues)),
            suggestions=list(set(sum_suggestions)),
            regulation_references=list(set(sum_refs)),
            reasoning="Unified Multi-Agent Consensus",
            confidence=0.8
        )

