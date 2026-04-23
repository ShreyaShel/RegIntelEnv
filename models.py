"""
RegIntelEnv – Pydantic Models
==============================
Defines the data structures for the Regulatory Intelligence Environment:

  - RegAction      : What an AI agent submits (findings + suggestions)
  - RegObservation : What the agent observes at each step
  - RegReward      : Structured reward breakdown (continuous 0.0–1.0)
  - RegState       : Episode-level tracking metadata
  - StepResult     : Combined output of a single step
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    UNCERTAIN = "uncertain"


class ActionType(str, Enum):
    ANALYZE = "analyze"          # Initial analysis of a process
    FLAG = "flag"                # Flag a specific non-compliance issue
    SUGGEST = "suggest"          # Suggest a remediation action
    CONCLUDE = "conclude"        # Final conclusion for the episode


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class RegAction(BaseModel):
    """
    An action submitted by the AI agent to the RegIntelEnv environment.

    The agent may submit one or more 'findings' and 'suggestions' per step.
    Each step should move closer to a complete compliance analysis.
    """

    action_type: ActionType = Field(
        ...,
        description="The type of action being taken this step"
    )

    # Findings (populated for ANALYZE / FLAG / CONCLUDE)
    process_analyzed: Optional[str] = Field(
        None,
        description="Name of the company process being analyzed"
    )
    identified_issues: List[str] = Field(
        default_factory=list,
        description="List of identified non-compliance issues"
    )
    compliance_status: Optional[ComplianceStatus] = Field(
        None,
        description="Agent's assessment of overall compliance status"
    )

    # Suggestions (populated for SUGGEST / CONCLUDE)
    suggestions: List[str] = Field(
        default_factory=list,
        description="Actionable remediation suggestions for identified issues"
    )

    # Reasoning (chain-of-thought)
    reasoning: Optional[str] = Field(
        None,
        description="Step-by-step reasoning trace supporting this action"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in this action (0.0–1.0)"
    )

    # References to regulation articles
    regulation_references: List[str] = Field(
        default_factory=list,
        description="Specific regulation articles referenced (e.g. 'GDPR Art.5')"
    )

    model_config = {"json_schema_extra": {"example": {
        "action_type": "flag",
        "process_analyzed": "Customer Data Retention",
        "identified_issues": ["Data kept beyond legal retention period"],
        "compliance_status": "non_compliant",
        "suggestions": ["Implement automated 90-day purge policy"],
        "reasoning": "GDPR Art.5(1)(e) requires data minimization...",
        "confidence": 0.9,
        "regulation_references": ["GDPR Art.5(1)(e)", "GDPR Art.17"]
    }}}


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class RegObservation(BaseModel):
    """
    What the agent observes after each step (or at reset).
    """

    # Task info
    task_id: str = Field(..., description="Unique task identifier")
    difficulty: DifficultyLevel = Field(..., description="Task difficulty level")
    step_number: int = Field(..., description="Current step number (0-indexed)")
    max_steps: int = Field(..., description="Maximum steps allowed for this task")

    # Company context
    company_name: str = Field(..., description="Name of the fictional company")
    industry: str = Field(..., description="Industry sector")
    regulation_name: str = Field(..., description="Applicable regulation name")
    regulation_summary: str = Field(..., description="Brief summary of the regulation")

    # Process being analyzed
    process_description: str = Field(
        ...,
        description="Detailed description of the company process to analyze"
    )
    process_name: str = Field(..., description="Short name of the process")

    # Regulatory Drift
    regulatory_drift: Optional[str] = Field(
        None,
        description="Real-time update or change in the applicable regulation occurring mid-process"
    )

    # Hints and feedback (updated each step)
    feedback: Optional[str] = Field(
        None,
        description="Feedback from the previous step grader"
    )
    hints: List[str] = Field(
        default_factory=list,
        description="Contextual hints unlocked based on progress"
    )
    issues_found_so_far: List[str] = Field(
        default_factory=list,
        description="Issues identified across all steps so far"
    )
    suggestions_given_so_far: List[str] = Field(
        default_factory=list,
        description="Suggestions given across all steps so far"
    )

    # Episode status
    done: bool = Field(default=False, description="Whether the episode is complete")
    reward: float = Field(default=0.0, description="Reward from last step")
    total_reward: float = Field(default=0.0, description="Cumulative reward so far")

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class RegReward(BaseModel):
    """
    Structured, continuous reward breakdown for a step.
    All sub-scores are in [0.0, 1.0]; total is their weighted average.
    """

    total: float = Field(
        ..., ge=0.0, le=1.0,
        description="Overall weighted reward (0.0–1.0)"
    )

    # Sub-scores
    issue_identification_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="How well the agent identified real compliance issues"
    )
    suggestion_quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Quality and actionability of remediation suggestions"
    )
    regulation_accuracy_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Accuracy of cited regulation references"
    )
    reasoning_quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Clarity and depth of reasoning trace"
    )
    false_positive_penalty: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Penalty for incorrectly flagged issues (higher = worse)"
    )

    explanation: str = Field(
        default="",
        description="Human-readable explanation of the reward breakdown"
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RegState(BaseModel):
    """
    Episode-level metadata maintained by the server.
    """

    episode_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this episode"
    )
    step_count: int = Field(default=0, description="Number of steps taken")
    task_id: str = Field(default="", description="Active task ID")
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.EASY,
        description="Active task difficulty"
    )
    total_reward: float = Field(
        default=0.0,
        description="Cumulative reward across all steps"
    )
    done: bool = Field(default=False)
    issues_found: List[str] = Field(default_factory=list)
    suggestions_given: List[str] = Field(default_factory=list)
    regulation_refs_used: List[str] = Field(default_factory=list)
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full history of actions and rewards for this episode"
    )


# ---------------------------------------------------------------------------
# StepResult  (returned by /step endpoint)
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Combined output returned after each step."""

    observation: RegObservation
    reward: RegReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
