from typing import Any, Dict, List, Set, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class CoalitionEngine:
    """Production-grade Coalition Engine with deterministic tie-breaking and conflict resolution."""
    
    COMPLEMENTARY_PAIRS = {
        ("auditor", "lawyer"),
        ("lawyer", "engineer"),
        ("auditor", "engineer"),
        ("coordinator", "auditor"),
        ("coordinator", "lawyer"),
        ("coordinator", "engineer")
    }

    def __init__(self):
        self.proposals: Dict[str, Set[str]] = {}

    def propose_coalition(self, agent_id: str, target_agents: List[str]):
        """Safely register a coalition proposal."""
        if not isinstance(agent_id, str) or not isinstance(target_agents, list):
            logger.warning(f"Malformed proposal from {agent_id}")
            return
        self.proposals[agent_id] = set(target_agents)

    def resolve_coalitions(self, proposals: Dict[str, Set[str]]) -> List[Set[str]]:
        """
        DETERMINISTIC RESOLUTION:
        1. Sort agents alphabetically to ensure same output for same input.
        2. Priority given to agents appearing earlier in sorted list.
        3. Agent is removed from market once they join a coalition.
        """
        final_coalitions = []
        processed = set()
        
        sorted_proposers = sorted(proposals.keys())
        
        for proposer in sorted_proposers:
            if proposer in processed:
                continue
            
            # Mutual agreement logic
            targets = proposals[proposer]
            current_coalition = {proposer}
            
            # Sort targets to ensure deterministic selection if multiple mutuals exist
            sorted_targets = sorted([t for t in targets if t != proposer])
            for target in sorted_targets:
                if target in processed:
                    continue
                
                # Bi-directional mutual agreement check
                if target in proposals and proposer in proposals[target]:
                    current_coalition.add(target)
            
            if len(current_coalition) > 1:
                final_coalitions.append(current_coalition)
                processed.update(current_coalition)
        
        return final_coalitions

    def merge_and_grade(self, 
                       agent_actions: Dict[str, Dict[str, Any]], 
                       agent_roles: Dict[str, str]) -> Dict[str, float]:
        """
        Merge actions and calculate reward modifiers safely.
        Returns: Dict[agent_id, modifier_float]
        """
        # Resolve coalitions deterministically
        resolved = self.resolve_coalitions(self.proposals)
        reward_modifiers = {aid: 1.0 for aid in agent_actions}
        
        # 1. Redundancy Processing (Deterministic findings scan)
        finding_registry: Dict[str, List[str]] = {} # finding_text -> list[agent_id]
        
        for aid in sorted(agent_actions.keys()):
            act = agent_actions[aid]
            findings = act.get("findings", [])
            if isinstance(findings, list):
                for f in findings:
                    if f not in finding_registry:
                        finding_registry[f] = []
                    finding_registry[f].append(aid)
        
        # Apply -20% penalty for redundant findings
        for f, discovery_agents in finding_registry.items():
            if len(discovery_agents) > 1:
                for aid in discovery_agents:
                    reward_modifiers[aid] *= 0.8
                    logger.debug(f"Redundancy Penalty: {aid} (shared '{f}')")

        # 2. Coalition Processing (+30% bonus)
        for coalition in resolved:
            roles = [agent_roles.get(aid, "unknown") for aid in coalition]
            
            # Check if any pair in coalition is complementary
            has_synergy = False
            for i in range(len(roles)):
                for j in range(i + 1, len(roles)):
                    if (roles[i], roles[j]) in self.COMPLEMENTARY_PAIRS or \
                       (roles[j], roles[i]) in self.COMPLEMENTARY_PAIRS:
                        has_synergy = True
                        break
            
            if has_synergy:
                for aid in coalition:
                    if aid in reward_modifiers:
                        reward_modifiers[aid] *= 1.3
                        logger.debug(f"Coalition Bonus: {aid} (+30%)")
        
        # Clear state for next step
        self.proposals = {}
        return reward_modifiers
