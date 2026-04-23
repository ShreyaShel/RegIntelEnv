"""
RegIntelEnv – Grader System
============================
This is the "brain" of our scoring system. 
Instead of just checking if an agent is right or wrong, we look at several 
different dimensions of their performance. This makes the reward signal 
much more useful for training reinforcement learning models.

Grading dimensions:
  1. Issue Identification Score  – Did they find the real compliance gaps?
  2. Suggestion Quality Score    – Are their fixes actionable and smart?
  3. Regulation Accuracy Score   – Did they cite the right law (e.g. GDPR Art.5)?
  4. Reasoning Quality Score     – Is their thinking clear and logical?
  5. False-Positive Penalty      – Did they hallucinate issues that aren't there?
"""


from __future__ import annotations

import re
from typing import List, Optional

from models import RegAction, RegReward
from tasks import ComplianceTask


# ---------------------------------------------------------------------------
# Weights for the final score
# ---------------------------------------------------------------------------

WEIGHTS = {
    "issue_identification": 0.35,
    "suggestion_quality": 0.30,
    "regulation_accuracy": 0.20,
    "reasoning_quality": 0.15,
}
FALSE_POSITIVE_PENALTY_WEIGHT = 0.20   # max deduction fraction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _keyword_overlap_score(
    text: str,
    keyword_groups: List[List[str]],
    normalise: bool = True,
) -> float:
    """
    Compute partial coverage score.

    For each group of keywords (representing one expected item), the text
    gets credit if ANY keyword in the group is found (case-insensitive).
    Score = (number of groups matched) / (total groups).
    Returns a value in [0.0, 1.0].
    """
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
    """Combine agent output into a single string for keyword matching."""
    parts = issues + suggestions
    if reasoning:
        parts.append(reasoning)
    return " ".join(parts)


def _match_regulation_articles(
    cited: List[str],
    expected: List[str],
) -> float:
    """
    Measure accuracy of cited regulation articles.
    Partial credit: each correctly cited article earns proportional score.
    False articles cited earn a small penalty.
    """
    if not expected:
        return 1.0  # nothing to check

    cited_lower = [c.lower().strip() for c in cited]
    expected_lower = [e.lower().strip() for e in expected]

    # How many expected articles were mentioned
    correct = sum(
        1 for exp in expected_lower if any(exp in c for c in cited_lower)
    )
    recall = correct / len(expected)

    # Penalty for hallucinated articles
    false_cited = sum(
        1 for c in cited_lower
        if not any(exp in c for exp in expected_lower)
    )
    penalty = min(false_cited * 0.05, 0.3)  # cap at 30% penalty

    return max(0.0, recall - penalty)


def _score_reasoning(reasoning: Optional[str]) -> float:
    """
    Heuristic quality score for the reasoning trace.
    Checks length, structure, use of legal terminology.
    """
    if not reasoning:
        return 0.0

    score = 0.0
    word_count = len(reasoning.split())

    # Length scoring: 50–300 words is ideal
    if word_count >= 300:
        score += 0.4
    elif word_count >= 150:
        score += 0.3
    elif word_count >= 50:
        score += 0.2
    elif word_count >= 20:
        score += 0.1

    # Structure: mentions specific article numbers
    if re.search(r"art\.\s*\d+|article\s+\d+|recital\s+\d+", reasoning, re.IGNORECASE):
        score += 0.3

    # Mentions cause-effect reasoning
    causal_keywords = ["because", "therefore", "thus", "this means", "as a result",
                       "consequently", "violates", "requires", "mandates"]
    hits = sum(1 for kw in causal_keywords if kw.lower() in reasoning.lower())
    score += min(hits * 0.05, 0.2)

    # Mentions remediation
    if any(kw in reasoning.lower() for kw in ["should", "must", "recommend", "implement"]):
        score += 0.1

    return min(score, 1.0)


def _false_positive_penalty(
    issues: List[str],
    task: ComplianceTask,
) -> float:
    """
    Compute penalty for issues that are explicitly marked as false/non-existent.
    Returns a value in [0.0, 1.0] where 1.0 = maximum penalty.
    """
    if not task.false_issues or not issues:
        return 0.0

    combined_text = " ".join(issues).lower()
    wrong_flags = sum(
        1 for fi in task.false_issues
        if any(kw.lower() in combined_text for kw in fi.split()[:4])
    )
    # Each false issue = 0.25 penalty, capped at 1.0
    return min(wrong_flags * 0.25, 1.0)


# ---------------------------------------------------------------------------
# Main grader
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LLM-Powered Grader (Advanced Semantic Evaluation)
# ---------------------------------------------------------------------------

import json
import os
import urllib.request
from typing import Dict, Any

class LLMGrader:
    """
    Acts as a 'Senior Legal Counsel' to evaluate the quality of compliance work.
    This uses LLM API credits to provide high-fidelity feedback.
    """
    
    SYSTEM_PROMPT = """You are a senior regulatory compliance auditor. 
Evaluate the agent's identified issues and suggestions against the ground truth.
Check if the legal reasoning is sound and the citations are accurate.

Respond ONLY with valid JSON:
{
  "semantic_score": 0.0 to 1.0,
  "feedback": "Detailed explanation of strengths and weaknesses...",
  "suggested_score_adjustment": -0.2 to 0.2
}"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("HF_TOKEN")
        self.base_url = os.environ.get("API_BASE_URL") # Required for HF Inference Endpoints

    def evaluate_semantically(self, task: ComplianceTask, action: RegAction) -> Dict[str, Any]:
        """Performs the LLM-based semantic audit of the agent's action."""
        if os.getenv("EXPERT_JUDGE") == "0":
            return {"semantic_score": 0.5, "feedback": "Expert Judge Disabled (Simulation Mode)", "suggested_score_adjustment": 0}
            
        if not self.api_key or not self.base_url:
            return {"semantic_score": 0.5, "feedback": "HF_TOKEN or API_BASE_URL missing. Skipping semantic evaluation.", "suggested_score_adjustment": 0}

        prompt = f"""
TASK CONTEXT:
Company: {task.company_name}
Industry: {task.industry}
Regulation: {task.regulation_name}

GROUND TRUTH (What should be found):
Expected Issues: {task.expected_issues}
Expected Suggestions: {task.expected_suggestions}

AGENT SUBMISSION:
Identified Issues: {action.identified_issues}
Suggestions: {action.suggestions}
Reasoning: {action.reasoning}
Citations: {action.regulation_references}

Evaluate the QUALITY and ACCURACY of this submission.
"""
        try:
            # Simple stdlib HTTP POST to avoid adding heavy deps like 'openai' package to the server if not present
            body = json.dumps({
                "model": os.environ.get("MODEL_NAME", "gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }).encode("utf-8")
            
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                result = json.loads(resp.read().decode())
                return json.loads(result["choices"][0]["message"]["content"])
        except Exception as e:
            return {"semantic_score": 0.5, "legal_feedback": f"LLM Grading failed: {e}", "suggested_score_adjustment": 0}

class ComplianceGrader:
    """
    Grades a single RegAction against the ground-truth ComplianceTask.
    Returns a RegReward with detailed sub-scores and an explanation.
    """

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

        # 1. Rule-based scoring (Heuristics)
        all_issues = list(cumulative_issues or []) + list(action.identified_issues)
        all_suggestions = list(cumulative_suggestions or []) + list(action.suggestions)
        all_refs = list(action.regulation_references)

        full_text = _text_from_issues_and_suggestions(
            all_issues, all_suggestions, action.reasoning
        )

        issue_score = _keyword_overlap_score(full_text, task.partial_issue_keywords)
        suggestion_score = _keyword_overlap_score(full_text, task.partial_suggestion_keywords)
        reg_score = _match_regulation_articles(all_refs, task.key_regulation_articles)
        reasoning_score = _score_reasoning(action.reasoning)
        fp_penalty = _false_positive_penalty(list(action.identified_issues), task)

        # ---- 6. Anti-Reward Hacking (Rule 8) --------------------------
        # Penalty for keyword stuffing / extreme verbosity
        word_count = len(full_text.split())
        verbosity_penalty = 0.0
        if word_count > 1000: # Clearly just dumping text
            verbosity_penalty = 0.3
        elif word_count > 500:
            verbosity_penalty = 0.1
            
        # Format Compliance check (Rule 7)
        # Ensure at least one issue and one suggestion are present if flagging
        format_score = 1.0
        if action.action_type == "flag" and (not action.identified_issues or not action.suggestions):
            format_score = 0.5

        # ---- Weighted Total -------------------------------------------
        raw_total = (
            WEIGHTS["issue_identification"]   * issue_score
            + WEIGHTS["suggestion_quality"]   * suggestion_score
            + WEIGHTS["regulation_accuracy"]  * reg_score
            + WEIGHTS["reasoning_quality"]    * reasoning_score
        )
        
        # 2. LLM-based semantic adjustment (The "Pro" layer)
        llm_result = self.llm_grader.evaluate_semantically(task, action)
        semantic_adj = llm_result.get("suggested_score_adjustment", 0)
        legal_feedback = llm_result.get("legal_feedback", "")

        # 3. Final Calculation
        # Total = (Base * Format) + Adj - Penalties
        penalty_deduction = (FALSE_POSITIVE_PENALTY_WEIGHT * fp_penalty) + verbosity_penalty
        total = max(0.0, min(1.0, (raw_total * format_score) + semantic_adj - penalty_deduction))

        explanation = (
            f"Base: {raw_total:.2f} | Format: x{format_score} | Adj: {semantic_adj:+.2f} | "
            f"Penalties: -{penalty_deduction:.2f} | Final: {total:.3f}\\n"
            f"LEGAL FEEDBACK: {legal_feedback}"
        )

        return RegReward(
            total=round(total, 4),
            issue_identification_score=round(issue_score, 4),
            suggestion_quality_score=round(suggestion_score, 4),
            regulation_accuracy_score=round(reg_score, 4),
            reasoning_quality_score=round(reasoning_score, 4),
            false_positive_penalty=round(fp_penalty, 4),
            explanation=explanation,
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_grader(task: ComplianceTask) -> ComplianceGrader:
    return ComplianceGrader(task)
