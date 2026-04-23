from typing import Any, Dict, List, Optional, Set
import logging
from uuid import uuid4

logger = logging.getLogger(__name__)

class MetaLearner:
    """Production MetaLearner with capability gap analysis and specialist weight inheritance."""
    
    def __init__(self, initial_agents: List[Any], failure_threshold: int = 3, min_reward: float = 0.25):
        self.agents = initial_agents
        self.failure_threshold = failure_threshold
        self.min_reward = min_reward
        self.performance_tracker: Dict[str, List[float]] = {}
        self.active_specializations: Set[tuple] = set() # (role, spec_name)

    def analyze_episode(self, history: Dict[str, Any]):
        """Track per-agent reward history."""
        steps = history.get("steps", [])
        for step in steps:
            rewards = step.get("rewards", {})
            for aid, r in rewards.items():
                if aid not in self.performance_tracker:
                    self.performance_tracker[aid] = []
                self.performance_tracker[aid].append(max(0.0, float(r)))

    def detect_capability_gaps(self) -> List[Dict[str, str]]:
        """Identify critical failures repeated >= threshold."""
        gaps = []
        for aid, rewards in self.performance_tracker.items():
            if len(rewards) >= self.failure_threshold:
                # Calculate windowed average
                window = rewards[-self.failure_threshold:]
                avg = sum(window) / len(window)
                if avg < self.min_reward:
                    role = self._get_agent_role(aid)
                    gaps.append({"role": role, "agent_id": aid, "current_avg": avg})
        return gaps

    def spawn_specialist(self, role_name: str, specialization_type: str) -> Optional[Any]:
        """Spawn and initialize specialized agent from closest parent."""
        spec_key = (role_name, specialization_type)
        if spec_key in self.active_specializations:
            logger.debug(f"Cap: Already have a specialist for {spec_key}")
            return None

        from cascade_rl.agents import AuditorAgent, LawyerAgent, EngineerAgent, ComplianceOfficerAgent
        class_map = {
            "auditor": AuditorAgent,
            "lawyer": LawyerAgent,
            "engineer": EngineerAgent,
            "coordinator": ComplianceOfficerAgent
        }
        
        if role_name not in class_map:
            return None
        
        new_id = f"specialist_{role_name}_{specialization_type}_{str(uuid4())[:4]}"
        agent_cls = class_map[role_name]
        new_agent = agent_cls(agent_id=new_id)
        
        # INHERITANCE: Find an existing agent of the same role to source 'knowledge'
        parent = next((a for a in self.agents if self._get_agent_role(a.agent_id) == role_name), None)
        if parent:
            new_agent.policy = parent.policy.copy()
            logger.info(f"Specialist {new_id} inherited policy from {parent.agent_id}")
            
        new_agent.policy["specialization"] = specialization_type
        self.active_specializations.add(spec_key)
        return new_agent

    def update(self) -> List[Any]:
        """Run Eco-system evolution cycle."""
        spawned = []
        gaps = self.detect_capability_gaps()
        for gap in gaps:
            # Create a focused role for the detected failure
            focus = "AdaptationExpert" if gap["current_avg"] < 0.1 else "RecoveryExpert"
            new_agent = self.spawn_specialist(gap["role"], focus)
            if new_agent:
                spawned.append(new_agent)
                self.agents.append(new_agent)
                # Reset tracker to clear trigger
                self.performance_tracker[gap["agent_id"]] = [] 
                logger.info(f"ACHIEVEMENT: Meta-Learner spawned {new_agent.agent_id} to fix {gap['role']} gap.")
        return spawned

    def _get_agent_role(self, agent_id: str) -> str:
        for agent in self.agents:
            if agent.agent_id == agent_id:
                name = agent.__class__.__name__.lower()
                if "auditor" in name: return "auditor"
                if "lawyer" in name: return "lawyer"
                if "engineer" in name: return "engineer"
                if "compliance" in name or "coordinator" in name: return "coordinator"
        return "unknown"
