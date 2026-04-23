"""
RegIntelEnv – Core Environment
================================
Implements the OpenEnv-style Environment with:
  - reset(task_id, difficulty, seed) → RegObservation
  - step(action)                    → StepResult
  - state()                         → RegState
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Ensure project root (one level up from server/) is importable
_server_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_server_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from grader import ComplianceGrader, get_grader
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
from tasks import ComplianceTask, get_task, get_task_by_difficulty

logger = logging.getLogger(__name__)


class RegIntelEnvironment:
    """
    Regulatory Intelligence Environment (RegIntelEnv).

    AI agents analyze company processes against regulations and flag
    non-compliance. Supports three tasks of increasing complexity.

    Lifecycle:
        env = RegIntelEnvironment()
        obs = env.reset(difficulty="medium")
        result = env.step(action)
        state = env.state()
    """

    def __init__(self):
        self._state: RegState = RegState()
        self._task: Optional[ComplianceTask] = None
        self._grader: Optional[ComplianceGrader] = None
        self._episode_active: bool = False

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = "easy",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> RegObservation:
        """
        Start a new episode.

        Parameters
        ----------
        task_id : str, optional
            Specific task to load. If provided, overrides 'difficulty'.
        difficulty : str, optional
            'easy', 'medium', or 'hard'. Default: 'easy'.
        seed : int, optional
            Random seed (reserved for future use).
        episode_id : str, optional
            Explicit episode ID; auto-generated if not provided.

        Returns
        -------
        RegObservation
            Initial observation describing the company and process.
        """
        # Load task
        if task_id:
            task = get_task(task_id)
        else:
            task = get_task_by_difficulty(difficulty or "easy")

        self._task = task
        self._grader = get_grader(task)

        # Initialise state
        self._state = RegState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task.task_id,
            difficulty=DifficultyLevel(task.difficulty),
            total_reward=0.0,
            done=False,
        )
        self._episode_active = True

        logger.info(
            "Episode started | task=%s | difficulty=%s | episode_id=%s",
            task.task_id,
            task.difficulty,
            self._state.episode_id,
        )

        return self._build_observation(
            feedback=None,
            reward=0.0,
            done=False,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        action: RegAction,
        **kwargs: Any,
    ) -> StepResult:
        """
        Execute one agent action and return a graded result.

        Parameters
        ----------
        action : RegAction
            The agent's compliance analysis action this step.

        Returns
        -------
        StepResult
            observation, reward breakdown, done flag.
        """
        if not self._episode_active or self._task is None:
            raise RuntimeError("Environment is not active. Call reset() first.")

        if self._state.done:
            raise RuntimeError("Episode is already done. Call reset() to start a new episode.")

        # Accumulate episode history
        self._state.step_count += 1
        self._state.issues_found.extend(action.identified_issues)
        self._state.suggestions_given.extend(action.suggestions)
        self._state.regulation_refs_used.extend(action.regulation_references)

        # Grade the action
        reward: RegReward = self._grader.grade(
            action=action,
            step_number=self._state.step_count,
            cumulative_issues=list(self._state.issues_found),
            cumulative_suggestions=list(self._state.suggestions_given),
        )

        # Update cumulative reward
        self._state.total_reward = min(
            1.0, self._state.total_reward + reward.total / self._task.max_steps
        )

        # Determine if episode should end
        done = self._should_terminate(action, reward)
        self._state.done = done

        self._state.history.append({
            "step": self._state.step_count,
            "action_type": action.action_type.value,
            "issues": action.identified_issues,
            "suggestions": action.suggestions,
            "reward": reward.total,
            "explanation": reward.explanation,
        })

        logger.debug(
            "Step %d | action=%s | reward=%.4f | done=%s",
            self._state.step_count,
            action.action_type.value,
            reward.total,
            done,
        )

        # Build feedback message
        feedback = self._build_feedback(action, reward, done)

        obs = self._build_observation(
            feedback=feedback,
            reward=reward.total,
            done=done,
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "total_reward": self._state.total_reward,
            },
        )

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> RegState:
        """Return the current episode state."""
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_terminate(self, action: RegAction, reward: RegReward) -> bool:
        """Determine if the episode should end this step."""
        task = self._task
        # End if agent explicitly concludes
        if action.action_type == ActionType.CONCLUDE:
            return True
        # End if max steps reached
        if self._state.step_count >= task.max_steps:
            return True
        # End if agent achieves near-perfect score in a single step (early stop)
        if reward.total >= 0.95:
            return True
        return False

    def _build_observation(
        self,
        feedback: Optional[str],
        reward: float,
        done: bool,
    ) -> RegObservation:
        """Construct a RegObservation from current state."""
        task = self._task
        step = self._state.step_count

        # Unlock hints and drift based on current step
        hints = []
        for step_threshold, hint_text in task.hints.items():
            if step >= int(step_threshold):
                hints.append(hint_text)
                
        # ⚔️ ADVERSARIAL MODE: Inject stakeholder pressure into the observation
        for step_threshold, pressure_text in task.adversarial_injections.items():
            if step == int(step_threshold):
                hints.append(f"🔴 ADVERSARIAL PRESSURE: {pressure_text}")

        drift = task.drift_events.get(step)

        return RegObservation(
            task_id=task.task_id,
            difficulty=DifficultyLevel(task.difficulty),
            step_number=step,
            max_steps=task.max_steps,
            company_name=task.company_name,
            industry=task.industry,
            regulation_name=task.regulation_name,
            regulation_summary=task.regulation_summary,
            process_name=task.process_name,
            process_description=task.process_description,
            regulatory_drift=drift,
            feedback=feedback,
            hints=hints,
            issues_found_so_far=list(self._state.issues_found),
            suggestions_given_so_far=list(self._state.suggestions_given),
            done=done,
            reward=reward,
            total_reward=self._state.total_reward,
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": step,
                "max_steps": task.max_steps,
            },
        )

    def _build_feedback(
        self,
        action: RegAction,
        reward: RegReward,
        done: bool,
    ) -> str:
        """Generate human-readable feedback for the agent."""
        parts = [f"Step {self._state.step_count} feedback:"]

        if reward.issue_identification_score > 0.6:
            parts.append("✅ Good issue identification.")
        elif reward.issue_identification_score > 0.3:
            parts.append("⚠️ Partial issues identified — look deeper into the process.")
        else:
            parts.append("❌ Issues not well identified — re-read the process description carefully.")

        if reward.suggestion_quality_score > 0.6:
            parts.append("✅ Suggestions are actionable and specific.")
        elif reward.suggestion_quality_score > 0.3:
            parts.append("⚠️ Some suggestions noted — make them more specific and actionable.")
        else:
            parts.append("❌ Suggestions need improvement — link each to a specific issue.")

        if reward.regulation_accuracy_score > 0.5:
            parts.append("✅ Regulation references are accurate.")
        elif action.regulation_references:
            parts.append("⚠️ Some regulation references may be inaccurate — double-check article numbers.")
        else:
            parts.append("⚠️ No regulation articles cited — reference specific articles for higher scores.")

        if reward.false_positive_penalty > 0.0:
            parts.append("⚠️ Some flagged issues are not actual compliance violations.")

        if done and self._state.step_count >= self._task.max_steps:
            parts.append(f"📊 Episode complete. Final score: {self._state.total_reward:.3f}")

        return " | ".join(parts)
