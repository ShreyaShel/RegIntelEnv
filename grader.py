"""
RegIntelEnv – Grader System
============================
This is the "brain" of our scoring system. 
Instead of just checking if an agent is right or wrong, we look at several 
different dimensions of their performance. This makes the reward signal 
much more useful for training reinforcement learning models.

Grading dimensions:
  1. Violation Detection Score   – Did they find the real compliance gaps?
  2. Remediation Quality Score   – Are their fixes actionable and smart?
  3. Legal Accuracy Score        – Did they cite the right law?
  4. Reasoning Depth Score       – Is their thinking clear and logical?
"""

from __future__ import annotations
import re
import json
import os
import urllib.request
from typing import List, Optional, Dict, Any

from models import RegAction, RegReward
from tasks import ComplianceTask


# ---------------------------------------------------------------------------
# Weights for the final score (HACKATHON SPEC)
# ---------------------------------------------------------------------------

WEIGHTS = {
    "legal_accuracy": 0.3,
    "violation_detection": 0.3,
    "remediation_quality": 0.2,
    "reasoning_depth": 0.2,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _keyword_overlap_score(
    text: str,
    keyword_groups: List[List[str]],
    normalise: bool = True,
) -> float:
    if not keyword_groups:
        return 0.0

    text_lower = text.lower()
    matched = sum(
        1
        for group in keyword_groups
        if any(kw.lower() in text_lower for kw in group)
    )
    if normalise:
        return matched / len(keyword_groups)
    return float(matched)


def _text_from_issues_and_suggestions(
    issues: List[str],
    suggestions: List[str],
    reasoning: Optional[str],
) -> str:
    parts = issues + suggestions
    if reasoning:
        parts.append(reasoning)
    return " ".join(parts)


def _match_regulation_articles(
    cited: List[str],
    expected: List[str],
) -> float:
    if not expected:
        return 1.0

    cited_lower = [c.lower().strip() for c in cited]
    expected_lower = [e.lower().strip() for e in expected]

    correct = sum(
        1 for exp in expected_lower if any(exp in c for c in cited_lower)
    )
    recall = correct / len(expected)

    false_cited = sum(
        1 for c in cited_lower
        if not any(exp in c for exp in expected_lower)
    )
    penalty = min(false_cited * 0.05, 0.3)

    return max(0.0, recall - penalty)


def _score_reasoning(reasoning: Optional[str]) -> float:
    if not reasoning:
        return 0.0

    score = 0.0
    word_count = len(reasoning.split())

    if word_count >= 150:
        score += 0.4
    elif word_count >= 50:
        score += 0.2
    elif word_count >= 20:
        score += 0.1

    if re.search(r"art\.\s*\d+|article\s+\d+|recital\s+\d+", reasoning, re.IGNORECASE):
        score += 0.3

    causal_keywords = ["because", "therefore", "thus", "this means", "as a result",
                       "consequently", "violates", "requires", "mandates"]
    hits = sum(1 for kw in causal_keywords if kw.lower() in reasoning.lower())
    score += min(hits * 0.05, 0.2)

    if any(kw in reasoning.lower() for kw in ["should", "must", "recommend", "implement"]):
        score += 0.1

    return min(score, 1.0)


def _false_positive_penalty(
    issues: List[str],
    task: ComplianceTask,
) -> float:
    if not task.false_issues or not issues:
        return 0.0

    combined_text = " ".join(issues).lower()
    wrong_flags = sum(
        1 for fi in task.false_issues
        if any(kw.lower() in combined_text for kw in fi.split()[:4])
    )
    return min(wrong_flags * 0.25, 1.0)


# ---------------------------------------------------------------------------
# LLM-Powered Grader (Expert Layer)
# ---------------------------------------------------------------------------

class LLMGrader:
    SYSTEM_PROMPT = """You are a senior regulatory compliance auditor. 
Evaluate the agent's identified issues and suggestions against the ground truth.
Check if the legal reasoning is sound and the citations are accurate."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("HF_TOKEN")
        self.base_url = os.environ.get("API_BASE_URL")

    def evaluate_semantically(self, task: ComplianceTask, action: RegAction) -> Dict[str, Any]:
        # Implementation for semantic evaluation...
        return {"suggested_score_adjustment": 0, "legal_feedback": "Automated evaluation completed."}

# ---------------------------------------------------------------------------
# Main grader
# ---------------------------------------------------------------------------

class ComplianceGrader:
    def __init__(self, task: ComplianceTask):
        self.task = task
        self.llm_grader = LLMGrader()

    def grade(
        self,
        action: RegAction,
        step_number: int = 0,
        cumulative_issues: Optional[List[str]] = None,
        cumulative_suggestions: Optional[List[str]] = None,
    ) -> RegReward:
        task = self.task

        all_issues = list(cumulative_issues or []) + list(action.identified_issues)
        all_suggestions = list(cumulative_suggestions or []) + list(action.suggestions)
        all_refs = list(action.regulation_references)

        full_text = _text_from_issues_and_suggestions(
            all_issues, all_suggestions, action.reasoning
        )

        violation_detection = _keyword_overlap_score(full_text, task.partial_issue_keywords)
        remediation_quality = _keyword_overlap_score(full_text, task.partial_suggestion_keywords)
        legal_accuracy = _match_regulation_articles(all_refs, task.key_regulation_articles)
        reasoning_depth = _score_reasoning(action.reasoning)
        
        fp_penalty = _false_positive_penalty(list(action.identified_issues), task)

        total = (
            WEIGHTS["legal_accuracy"]      * legal_accuracy
            + WEIGHTS["violation_detection"] * violation_detection
            + WEIGHTS["remediation_quality"] * remediation_quality
            + WEIGHTS["reasoning_depth"]     * reasoning_depth
        )
        
        total = max(0.0, total - (0.1 * fp_penalty))

        explanation = (
            f"Legal: {legal_accuracy:.2f} | Violation: {violation_detection:.2f} | "
            f"Remediation: {remediation_quality:.2f} | Depth: {reasoning_depth:.2f} | "
            f"Penalty: -{0.1*fp_penalty:.2f} | Final: {total:.3f}"
        )

        return RegReward(
            total=round(total, 4),
            issue_identification_score=round(violation_detection, 4),
            suggestion_quality_score=round(remediation_quality, 4),
            regulation_accuracy_score=round(legal_accuracy, 4),
            reasoning_quality_score=round(reasoning_depth, 4),
            false_positive_penalty=round(fp_penalty, 4),
            explanation=explanation,
        )


def get_grader(task: ComplianceTask) -> ComplianceGrader:
    return ComplianceGrader(task)
