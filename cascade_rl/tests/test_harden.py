import unittest
import random
from cascade_rl.agents import AuditorAgent, LawyerAgent, EngineerAgent
from cascade_rl.environment import CascadeEnvironment
from cascade_rl.coalition import CoalitionEngine
from cascade_rl.meta_learner import MetaLearner

class TestHardenCoalition(unittest.TestCase):
    def setUp(self):
        self.engine = CoalitionEngine()
        self.roles = {"A": "auditor", "B": "lawyer", "C": "engineer"}

    def test_overlapping_proposals(self):
        # A wants B, C
        self.engine.propose_coalition("A", ["B", "C"])
        # B wants A
        self.engine.propose_coalition("B", ["A"])
        # C wants A
        self.engine.propose_coalition("C", ["A"])
        
        # Result should be {A, B} or {A, C}, not both
        resolved = self.engine.resolve_coalitions(self.engine.proposals)
        agents_in_coalitions = [a for c in resolved for a in c]
        self.assertEqual(len(agents_in_coalitions), len(set(agents_in_coalitions)))
        self.assertEqual(len(resolved), 1)

class TestHardenDeterminism(unittest.TestCase):
    def test_env_determinism(self):
        seed = 123
        env1 = CascadeEnvironment(seed=seed)
        env2 = CascadeEnvironment(seed=seed)
        
        env1.reset(difficulty="medium")
        env2.reset(difficulty="medium")
        
        env1.inject_drift(1)
        env2.inject_drift(1)
        
        self.assertEqual(env1.current_drift_type, env2.current_drift_type)
        self.assertEqual(env1._task.process_description, env2._task.process_description)

class TestHardenDrift(unittest.TestCase):
    def test_drift_invalidation_flag(self):
        env = CascadeEnvironment()
        env.reset(difficulty="easy")
        env.inject_drift(1)
        
        actions = {"a1": {"findings": []}} # Empty action
        obs, rewards, done, info = env.step(actions)
        
        # Meta-data should reflect drift
        self.assertTrue(obs["a1"]["metadata"]["drift_occurred"])

class TestMetaLearnerQuality(unittest.TestCase):
    def test_duplicate_prevention(self):
        ml = MetaLearner(initial_agents=[])
        ml.spawn_specialist("auditor", "GDPR")
        second_spawn = ml.spawn_specialist("auditor", "GDPR")
        self.assertIsNone(second_spawn)

if __name__ == "__main__":
    unittest.main()
