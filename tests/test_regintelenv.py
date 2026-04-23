"""
Tests for RegIntelEnv
======================
Run with:  pytest tests/ -v
"""

from __future__ import annotations

import sys
import os

# Allow imports from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import (
    ActionType,
    ComplianceStatus,
    DifficultyLevel,
    RegAction,
    RegObservation,
    RegReward,
    RegState,
    StepResult,
)
from tasks import TASK_REGISTRY, get_task, get_task_by_difficulty
from grader import ComplianceGrader, get_grader
from server.reg_intel_environment import RegIntelEnvironment


# ---------------------------------------------------------------------------
# Task tests
# ---------------------------------------------------------------------------

class TestTasks:
    def test_all_tasks_exist(self):
        assert "task_gdpr_retention_easy" in TASK_REGISTRY
        assert "task_aiact_credit_medium" in TASK_REGISTRY
        assert "task_nis2_critical_hard" in TASK_REGISTRY

    def test_get_task_by_difficulty(self):
        for diff in ["easy", "medium", "hard"]:
            task = get_task_by_difficulty(diff)
            assert task.difficulty == diff

    def test_task_invalid_difficulty_raises(self):
        with pytest.raises(ValueError):
            get_task_by_difficulty("extreme")

    def test_task_invalid_id_raises(self):
        with pytest.raises(ValueError):
            get_task("nonexistent_task")

    def test_easy_task_structure(self):
        task = get_task_by_difficulty("easy")
        assert task.company_name
        assert task.process_description
        assert len(task.expected_issues) > 0
        assert len(task.expected_suggestions) > 0
        assert len(task.key_regulation_articles) > 0
        assert task.max_steps >= 1

    def test_medium_task_harder_than_easy(self):
        easy = get_task_by_difficulty("easy")
        medium = get_task_by_difficulty("medium")
        assert medium.max_steps >= easy.max_steps
        assert len(medium.expected_issues) > len(easy.expected_issues)

    def test_hard_task_hardest(self):
        medium = get_task_by_difficulty("medium")
        hard = get_task_by_difficulty("hard")
        assert len(hard.expected_issues) > len(medium.expected_issues)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_reg_action_defaults(self):
        action = RegAction(action_type=ActionType.ANALYZE)
        assert action.identified_issues == []
        assert action.suggestions == []
        assert action.confidence == 0.5

    def test_reg_action_full(self):
        action = RegAction(
            action_type=ActionType.FLAG,
            process_analyzed="Data Retention",
            identified_issues=["No retention policy"],
            compliance_status=ComplianceStatus.NON_COMPLIANT,
            suggestions=["Implement 90-day purge"],
            reasoning="GDPR Art.5 requires...",
            confidence=0.9,
            regulation_references=["GDPR Art.5"],
        )
        assert len(action.identified_issues) == 1
        assert action.compliance_status == ComplianceStatus.NON_COMPLIANT
        assert action.confidence == 0.9

    def test_reg_action_confidence_clamped(self):
        with pytest.raises(Exception):
            RegAction(action_type=ActionType.ANALYZE, confidence=1.5)

    def test_reg_reward_structure(self):
        reward = RegReward(
            total=0.75,
            issue_identification_score=0.8,
            suggestion_quality_score=0.7,
            regulation_accuracy_score=0.6,
            reasoning_quality_score=0.9,
            false_positive_penalty=0.0,
            explanation="Test explanation",
        )
        assert reward.total == 0.75
        assert 0.0 <= reward.total <= 1.0

    def test_reg_state_defaults(self):
        state = RegState()
        assert state.step_count == 0
        assert state.total_reward == 0.0
        assert not state.done
        assert state.episode_id  # auto-generated UUID

    def test_step_result_model(self):
        obs = RegObservation(
            task_id="test",
            difficulty=DifficultyLevel.EASY,
            step_number=1,
            max_steps=3,
            company_name="TestCo",
            industry="Tech",
            regulation_name="GDPR",
            regulation_summary="Test summary",
            process_description="Test process",
            process_name="Test Process",
        )
        reward = RegReward(total=0.5)
        result = StepResult(observation=obs, reward=reward, done=False)
        assert result.done is False
        assert result.reward.total == 0.5


# ---------------------------------------------------------------------------
# Grader tests
# ---------------------------------------------------------------------------

class TestGrader:
    def setup_method(self):
        self.task = get_task_by_difficulty("easy")
        self.grader = get_grader(self.task)

    def test_zero_score_for_empty_action(self):
        action = RegAction(action_type=ActionType.ANALYZE)
        reward = self.grader.grade(action)
        assert reward.total == 0.0
        assert reward.issue_identification_score == 0.0
        assert reward.suggestion_quality_score == 0.0

    def test_partial_score_for_single_issue(self):
        action = RegAction(
            action_type=ActionType.FLAG,
            identified_issues=["No documented data retention policy indefinite storage"],
            reasoning="The company keeps data without retention schedule.",
        )
        reward = self.grader.grade(action)
        assert reward.total > 0.0
        assert reward.issue_identification_score > 0.0

    def test_high_score_for_comprehensive_analysis(self):
        task = self.task
        action = RegAction(
            action_type=ActionType.CONCLUDE,
            identified_issues=task.expected_issues,
            suggestions=task.expected_suggestions,
            reasoning=(
                "GDPR Art.5 requires storage limitation — data should not be kept indefinitely. "
                "The company violates this because no retention schedule exists. "
                "Therefore, they must implement a 90-day purge. "
                "GDPR Art.17 grants right to erasure, but there is no deletion portal. "
                "GDPR Art.37 requires a formal DPO appointment. "
                "GDPR Art.46 mandates SCCs for US transfers — none exist. "
                "Because of these violations, the company should implement automated purge, "
                "notify employees, build a deletion portal, appoint a DPO, and sign SCCs. "
                "These recommendations must be implemented immediately."
            ),
            regulation_references=task.key_regulation_articles,
            confidence=0.95,
        )
        reward = self.grader.grade(
            action,
            cumulative_issues=task.expected_issues,
            cumulative_suggestions=task.expected_suggestions,
        )
        assert reward.total > 0.5
        assert reward.issue_identification_score > 0.5
        assert reward.suggestion_quality_score > 0.5

    def test_false_positive_penalty(self):
        task = self.task
        action = RegAction(
            action_type=ActionType.FLAG,
            identified_issues=task.false_issues,  # These are NOT real issues
        )
        reward = self.grader.grade(action)
        assert reward.false_positive_penalty > 0.0

    def test_regulation_accuracy_score(self):
        action = RegAction(
            action_type=ActionType.FLAG,
            identified_issues=["some issue"],
            regulation_references=self.task.key_regulation_articles,
        )
        reward = self.grader.grade(action)
        assert reward.regulation_accuracy_score > 0.0

    def test_reasoning_score_empty(self):
        action = RegAction(action_type=ActionType.ANALYZE, reasoning=None)
        reward = self.grader.grade(action)
        assert reward.reasoning_quality_score == 0.0

    def test_reasoning_score_detailed(self):
        action = RegAction(
            action_type=ActionType.ANALYZE,
            reasoning=(
                "GDPR Art.5(1)(e) requires storage limitation because personal data must not "
                "be kept longer than necessary. Therefore the current policy violates the regulation. "
                "As a result, the company should implement an automated deletion schedule. "
                "This means they must also appoint a formal DPO per Art.37, because the CFO cannot "
                "fulfil this role properly. Consequently, significant fines under GDPR Art.83 are possible. "
                "The US data transfer violates Art.46 since no standard contractual clauses exist. "
                "The company should sign SCCs with its analytics sub-processor immediately. "
                "I recommend a 90-day retention period for cancelled account data."
            ),
        )
        reward = self.grader.grade(action)
        assert reward.reasoning_quality_score > 0.3

    def test_grader_all_tasks(self):
        """Grader should work for all three task difficulties."""
        for diff in ["easy", "medium", "hard"]:
            task = get_task_by_difficulty(diff)
            grader = get_grader(task)
            action = RegAction(action_type=ActionType.ANALYZE)
            reward = grader.grade(action)
            assert 0.0 <= reward.total <= 1.0


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestEnvironment:
    def setup_method(self):
        self.env = RegIntelEnvironment()

    def test_reset_easy(self):
        obs = self.env.reset(difficulty="easy")
        assert obs.task_id == "task_gdpr_retention_easy"
        assert obs.difficulty == DifficultyLevel.EASY
        assert obs.step_number == 0
        assert not obs.done

    def test_reset_medium(self):
        obs = self.env.reset(difficulty="medium")
        assert obs.difficulty == DifficultyLevel.MEDIUM

    def test_reset_hard(self):
        obs = self.env.reset(difficulty="hard")
        assert obs.difficulty == DifficultyLevel.HARD

    def test_reset_by_task_id(self):
        obs = self.env.reset(task_id="task_nis2_critical_hard")
        assert obs.task_id == "task_nis2_critical_hard"

    def test_step_requires_reset(self):
        env = RegIntelEnvironment()
        action = RegAction(action_type=ActionType.ANALYZE)
        with pytest.raises(RuntimeError):
            env.step(action)

    def test_step_returns_step_result(self):
        self.env.reset(difficulty="easy")
        action = RegAction(action_type=ActionType.FLAG, identified_issues=["Test issue"])
        result = self.env.step(action)
        assert isinstance(result, StepResult)
        assert isinstance(result.reward, RegReward)
        assert isinstance(result.observation, RegObservation)

    def test_step_increments_step_count(self):
        self.env.reset(difficulty="easy")
        assert self.env.state().step_count == 0
        action = RegAction(action_type=ActionType.FLAG)
        self.env.step(action)
        assert self.env.state().step_count == 1

    def test_conclude_action_ends_episode(self):
        self.env.reset(difficulty="easy")
        action = RegAction(action_type=ActionType.CONCLUDE)
        result = self.env.step(action)
        assert result.done is True

    def test_max_steps_ends_episode(self):
        self.env.reset(difficulty="easy")
        task = self.env._task
        action = RegAction(action_type=ActionType.FLAG)
        for _ in range(task.max_steps - 1):
            result = self.env.step(action)
            if result.done:
                break
        # Last step
        result = self.env.step(action)
        assert result.done is True

    def test_step_after_done_raises(self):
        self.env.reset(difficulty="easy")
        conclude = RegAction(action_type=ActionType.CONCLUDE)
        self.env.step(conclude)
        with pytest.raises(RuntimeError):
            self.env.step(RegAction(action_type=ActionType.ANALYZE))

    def test_state_reflects_history(self):
        self.env.reset(difficulty="easy")
        issue = "Data kept indefinitely"
        action = RegAction(
            action_type=ActionType.FLAG,
            identified_issues=[issue],
        )
        self.env.step(action)
        state = self.env.state()
        assert issue in state.issues_found

    def test_reset_clears_state(self):
        self.env.reset(difficulty="easy")
        self.env.step(RegAction(action_type=ActionType.FLAG, identified_issues=["Issue 1"]))
        self.env.reset(difficulty="hard")  # reset to new task
        state = self.env.state()
        assert state.step_count == 0
        assert state.issues_found == []

    def test_full_easy_episode(self):
        """Integration: run a full easy episode and verify cumulative reward > 0."""
        obs = self.env.reset(difficulty="easy")
        task = get_task_by_difficulty("easy")
        total = 0.0
        for step_idx in range(task.max_steps):
            action = RegAction(
                action_type=ActionType.CONCLUDE if step_idx == task.max_steps - 1 else ActionType.FLAG,
                identified_issues=task.expected_issues[:3],
                suggestions=task.expected_suggestions[:3],
                regulation_references=task.key_regulation_articles[:2],
                reasoning=(
                    "GDPR Art.5 storage limitation is violated because data is kept indefinitely. "
                    "Therefore a retention policy is required. As a result, employees must be notified. "
                    "The DPO role must be formally appointed per Art.37."
                ),
                confidence=0.85,
            )
            result = self.env.step(action)
            total = result.observation.total_reward
            if result.done:
                break
        assert total > 0.0


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------

class TestAPIEndpoints:
    """Integration tests using TestClient."""

    def setup_method(self):
        # Import here to avoid circular deps at module load
        from fastapi.testclient import TestClient
        from server.app import app
        self.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "tasks_available" in data

    def test_tasks_endpoint(self):
        resp = self.client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert "task_gdpr_retention_easy" in data["tasks"]

    def test_reset_endpoint_easy(self):
        resp = self.client.post("/reset", json={"difficulty": "easy"})
        assert resp.status_code == 200
        obs = resp.json()
        assert obs["task_id"] == "task_gdpr_retention_easy"
        assert obs["difficulty"] == "easy"

    def test_reset_endpoint_invalid_difficulty(self):
        resp = self.client.post("/reset", json={"difficulty": "impossible"})
        assert resp.status_code == 400

    def test_step_endpoint(self):
        # First reset
        self.client.post("/reset", json={"difficulty": "easy"})
        # Then step
        resp = self.client.post("/step", json={
            "action": {
                "action_type": "flag",
                "identified_issues": ["No retention policy"],
                "suggestions": ["Implement 90-day purge"],
                "reasoning": "GDPR Art.5 storage limitation requires a retention schedule.",
                "regulation_references": ["GDPR Art.5"],
                "confidence": 0.8,
            }
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "reward" in data
        assert 0.0 <= data["reward"]["total"] <= 1.0

    def test_state_endpoint(self):
        self.client.post("/reset", json={"difficulty": "medium"})
        resp = self.client.get("/state")
        assert resp.status_code == 200
        state = resp.json()
        assert "episode_id" in state
        assert state["step_count"] == 0

    def test_step_without_reset_returns_400(self):
        # New client without any reset
        from fastapi.testclient import TestClient
        from server.app import app, get_env
        import server.app as app_module
        # Reset the global env
        app_module._env = None
        client = TestClient(app)
        resp = client.post("/step", json={
            "action": {"action_type": "analyze"}
        })
        assert resp.status_code == 400

    def test_web_interface_returns_html(self):
        resp = self.client.get("/web")
        assert resp.status_code == 200
        assert "RegIntelEnv" in resp.text
        assert "<html" in resp.text.lower()

    def test_docs_available(self):
        resp = self.client.get("/docs")
        assert resp.status_code == 200
