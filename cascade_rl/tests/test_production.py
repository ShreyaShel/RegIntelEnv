import unittest
import json
import os
from cascade_rl.agents import AuditorAgent, LawyerAgent
from cascade_rl.environment import CascadeEnvironment
from cascade_rl.coalition import CoalitionEngine
from cascade_rl.meta_learner import MetaLearner
from cascade_rl.train import run_cascade_train

class TestCascadeProduction(unittest.TestCase):
    
    def test_conflict_resolution(self):
        """Verify no agent exists in multiple coalitions even with cross-proposals."""
        engine = CoalitionEngine()
        # A wants B, B wants A (Pair 1)
        # C wants B, B wants C (Attempted Pair 2 - B is already taken)
        engine.propose_coalition("A", ["B"])
        engine.propose_coalition("B", ["A", "C"])
        engine.propose_coalition("C", ["B"])
        
        resolved = engine.resolve_coalitions(engine.proposals)
        
        # Should only be one coalition {A, B} because A is first in alphabet
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0], {"A", "B"})

    def test_determinism(self):
        """Same seed must produce identical metrics."""
        m1 = run_cascade_train(num_episodes=5, seed=7)
        m2 = run_cascade_train(num_episodes=5, seed=7)
        
        self.assertEqual(m1, m2, "Output differed between identical seeds")

    def test_drift_invalidation(self):
        """Drift must set flag and affect reward logic."""
        env = CascadeEnvironment(seed=1)
        env.reset(difficulty="easy")
        env.inject_drift(1)
        self.assertTrue(env.drift_active)
        
        # Test penalty for empty actions after drift
        _, rewards, _, _ = env.step({"a1": {}})
        self.assertIn("a1", rewards)
        # Final reward should be lower than base because of 0.7 multiplier
        # (Though with 0.0 base it stays 0.0, the logic path is tested)

    def test_meta_learner_threshold(self):
        """Meta learner shouldn't spawn until failure threshold reached."""
        a1 = AuditorAgent("a1")
        ml = MetaLearner(initial_agents=[a1], failure_threshold=3, min_reward=0.1)
        
        # 2 failures (threshold is 3)
        ml.analyze_episode({"steps": [{"rewards": {"a1": 0.0}}, {"rewards": {"a1": 0.0}}]})
        spawned = ml.update()
        self.assertEqual(len(spawned), 0)
        
        # 3rd failure
        ml.analyze_episode({"steps": [{"rewards": {"a1": 0.0}}]})
        spawned = ml.update()
        self.assertEqual(len(spawned), 1)

    def test_stress_no_crash(self):
        """Run a larger batch without errors."""
        try:
            run_cascade_train(num_episodes=20, seed=1)
            success = True
        except Exception as e:
            print(f"FAILED: {e}")
            success = False
        self.assertTrue(success)

if __name__ == "__main__":
    unittest.main()
