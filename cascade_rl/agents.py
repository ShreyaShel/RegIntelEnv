import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

class BaseAgent:
    """Base class for all agents in CascadeRL with production-level safeguards."""
    def __init__(self, agent_id: str, seed: Optional[int] = None):
        self.agent_id = agent_id
        self.memory: List[Dict[str, Any]] = []
        self.seed = seed
        self._rng = random.Random(seed)
        self.policy: Dict[str, Any] = {"performance": 0.0, "specialization": "none"}
        self.role = "base"

    def observe(self, observation: Any) -> Dict[str, Any]:
        """Process observation and store in memory safely."""
        try:
            obs = observation if isinstance(observation, dict) else observation.model_dump()
        except AttributeError:
            obs = {"error": "Invalid observation format"}
        
        self.memory.append({"observation": obs})
        return obs

    def act(self, observation: Any) -> Dict[str, Any]:
        """Must be implemented by subclasses."""
        raise NotImplementedError

    def update(self, reward: float):
        """Update policy based on reward safely."""
        safe_reward = max(0.0, float(reward)) if reward is not None else 0.0
        if self.memory:
            self.memory[-1]["reward"] = safe_reward
        self.policy["performance"] = self.policy.get("performance", 0.0) + safe_reward * 0.1

    def _get_fallback_action(self) -> Dict[str, Any]:
        """Standard fallback to prevent system crash."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "findings": [],
            "suggestions": [],
            "references": [],
            "action": "continue",
            "confidence": 0.1
        }

class AuditorAgent(BaseAgent):
    """Detects compliance issues. Heuristics: Keyword matching and pattern detection."""
    def __init__(self, agent_id: str, seed: Optional[int] = None):
        super().__init__(agent_id, seed)
        self.role = "auditor"

    def act(self, observation: Any) -> Dict[str, Any]:
        try:
            obs = observation if isinstance(observation, dict) else observation.model_dump()
            desc = obs.get("process_description", "").lower()
            
            # Simple heuristic detection
            findings = []
            if "retention" in desc or "storage" in desc:
                findings.append("Potential Data Retention Violation")
            if "not notify" in desc or "no notification" in desc:
                findings.append("Transparency/Notification Gap")
            
            return {
                "agent_id": self.agent_id,
                "role": self.role,
                "findings": findings,
                "confidence": 0.9 if findings else 0.5
            }
        except Exception:
            return self._get_fallback_action()

class LawyerAgent(BaseAgent):
    """Maps issues to regulations. Heuristics: Contextual mapping to known articles."""
    def __init__(self, agent_id: str, seed: Optional[int] = None):
        super().__init__(agent_id, seed)
        self.role = "lawyer"

    def act(self, observation: Any) -> Dict[str, Any]:
        try:
            obs = observation if isinstance(observation, dict) else observation.model_dump()
            findings = obs.get("issues_found_so_far", [])
            
            refs = []
            for f in findings:
                if "Retention" in f: refs.append("GDPR Art.5")
                elif "Notification" in f: refs.append("GDPR Art.13")
                else: refs.append(f"Reg Art.{self._rng.randint(1,100)}")
            
            return {
                "agent_id": self.agent_id,
                "role": self.role,
                "references": list(set(refs)),
                "confidence": 0.85
            }
        except Exception:
            return self._get_fallback_action()

class EngineerAgent(BaseAgent):
    """Proposes technical fixes. Heuristics: Standard remediation templates."""
    def __init__(self, agent_id: str, seed: Optional[int] = None):
        super().__init__(agent_id, seed)
        self.role = "engineer"

    def act(self, observation: Any) -> Dict[str, Any]:
        try:
            obs = observation if isinstance(observation, dict) else observation.model_dump()
            findings = obs.get("issues_found_so_far", [])
            
            suggestions = []
            for f in findings:
                suggestions.append(f"Automated remediation for: {f}")
            
            return {
                "agent_id": self.agent_id,
                "role": self.role,
                "suggestions": suggestions,
                "confidence": 0.75
            }
        except Exception:
            return self._get_fallback_action()

class ComplianceOfficerAgent(BaseAgent):
    """Coordinates decisions. Heuristics: Checks progress vs max_steps."""
    def __init__(self, agent_id: str, seed: Optional[int] = None):
        super().__init__(agent_id, seed)
        self.role = "coordinator"

    def act(self, observation: Any) -> Dict[str, Any]:
        try:
            obs = observation if isinstance(observation, dict) else observation.model_dump()
            step_num = obs.get("step_number", 0)
            max_steps = obs.get("max_steps", 3)
            
            # Decide to conclude or flag
            action_type = "finalize" if step_num >= max_steps - 1 else "continue"
            
            return {
                "agent_id": self.agent_id,
                "role": self.role,
                "action": action_type,
                "confidence": 1.0
            }
        except Exception:
            return self._get_fallback_action()
