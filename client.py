"""
RegIntelEnv – Synchronous Python Client
=========================================
A simple synchronous HTTP client for interacting with a running RegIntelEnv
server (local or Hugging Face Spaces).

Usage:
    from client import RegIntelClient, RegAction

    with RegIntelClient(base_url="http://localhost:7860") as client:
        obs = client.reset(difficulty="medium")
        print(obs["company_name"])

        result = client.step({
            "action_type": "flag",
            "identified_issues": ["CreditLens v3 is high-risk AI not registered in EU database"],
            "suggestions": ["Register in EU AI database before deployment"],
            "reasoning": "AI Act Art.6 + Annex III classify credit scoring as high-risk...",
            "regulation_references": ["AI Act Art.6", "AI Act Annex III"],
            "confidence": 0.9,
        })
        print(result["reward"]["total"])

    # Or direct usage without context manager:
    client = RegIntelClient(base_url="https://your-hf-space.hf.space")
    obs = client.reset(difficulty="easy")
    state = client.state()
    client.close()
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx


class RegIntelClient:
    """
    Synchronous HTTP client for RegIntelEnv.

    Wraps the REST API (reset / step / state / tasks / health) with
    clean Python methods, handling serialisation and error checking.

    Parameters
    ----------
    base_url : str
        Base URL of the running RegIntelEnv server.
        Default: http://localhost:7860
    timeout : float
        HTTP request timeout in seconds. Default: 30.0
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )

    # ------------------------------------------------------------------ #
    # Context manager support                                              #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "RegIntelClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    # ------------------------------------------------------------------ #
    # Core API methods                                                     #
    # ------------------------------------------------------------------ #

    def health(self) -> Dict[str, Any]:
        """Check whether the server is up and healthy."""
        return self._get("/health")

    def tasks(self) -> Dict[str, Any]:
        """List all available compliance tasks."""
        return self._get("/tasks")

    def reset(
        self,
        difficulty: str = "easy",
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Reset the environment and start a new episode.

        Parameters
        ----------
        difficulty : str
            'easy', 'medium', or 'hard'.
        task_id : str, optional
            Specific task ID — overrides difficulty if provided.
        seed : int, optional
            Random seed (for reproducibility, reserved for future use).
        episode_id : str, optional
            Custom episode identifier.

        Returns
        -------
        dict  — RegObservation
        """
        payload: Dict[str, Any] = {"difficulty": difficulty}
        if task_id:
            payload["task_id"] = task_id
        if seed is not None:
            payload["seed"] = seed
        if episode_id:
            payload["episode_id"] = episode_id
        return self._post("/reset", payload)

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit an agent action.

        Parameters
        ----------
        action : dict
            Must contain at least ``action_type``. Optional fields:
            - identified_issues: list[str]
            - suggestions: list[str]
            - reasoning: str
            - regulation_references: list[str]
            - compliance_status: str
            - confidence: float  (0.0–1.0)

        Returns
        -------
        dict  — StepResult  { observation, reward, done, info }
        """
        return self._post("/step", {"action": action})

    def state(self) -> Dict[str, Any]:
        """Return the current episode state (RegState)."""
        return self._get("/state")

    # ------------------------------------------------------------------ #
    # Convenience helpers                                                  #
    # ------------------------------------------------------------------ #

    def run_episode(
        self,
        difficulty: str = "easy",
        actions: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Run a full episode with a pre-defined list of actions.

        Useful for scripted evaluation without an LLM in the loop.

        Parameters
        ----------
        difficulty : str
            Task difficulty level.
        actions : list of dicts, optional
            If provided, steps through these actions in order.
            The last action is automatically forced to ``conclude``.

        Returns
        -------
        dict with keys: episode_id, difficulty, step_rewards, total_reward,
                        final_issues, final_suggestions
        """
        obs = self.reset(difficulty=difficulty)
        max_steps = obs["max_steps"]

        if actions is None:
            # Default: single blank conclude action
            actions = [{"action_type": "conclude"}]

        step_rewards = []
        final_obs = obs

        for i, action in enumerate(actions[:max_steps]):
            if i == len(actions) - 1 or i == max_steps - 1:
                action = {**action, "action_type": "conclude"}
            result = self.step(action)
            step_rewards.append(result["reward"]["total"])
            final_obs = result["observation"]
            if result["done"]:
                break

        return {
            "episode_id": final_obs["metadata"].get("episode_id", ""),
            "difficulty": difficulty,
            "step_rewards": step_rewards,
            "total_reward": final_obs.get("total_reward", 0.0),
            "final_issues": final_obs.get("issues_found_so_far", []),
            "final_suggestions": final_obs.get("suggestions_given_so_far", []),
        }

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _get(self, path: str) -> Dict[str, Any]:
        resp = self._client.get(path)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._client.post(path, json=body)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            raise RuntimeError(
                f"RegIntelEnv server returned {exc.response.status_code}: {detail}"
            ) from exc
        return resp.json()


# --------------------------------------------------------------------------- #
# CLI quick-test                                                               #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    base = os.environ.get("BASE_URL", "http://localhost:7860")
    print(f"Connecting to RegIntelEnv at {base}...")

    with RegIntelClient(base_url=base) as client:
        # Health check
        h = client.health()
        print(f"Health: {h['status']} | Tasks: {h['tasks_available']}")

        # Quick reset + step on easy task
        print("\n--- Easy Task Demo ---")
        obs = client.reset(difficulty="easy")
        print(f"Company : {obs['company_name']}")
        print(f"Process : {obs['process_name']}")

        result = client.step({
            "action_type": "flag",
            "identified_issues": [
                "No documented retention schedule — data kept indefinitely after cancellation",
                "Employees of cancelled accounts not notified about continued data storage",
                "No self-service deletion mechanism for individual data subjects",
                "DPO role fulfilled by CFO as secondary duty — not a formal appointment",
                "US data transfer to analytics sub-processor lacks Standard Contractual Clauses",
            ],
            "suggestions": [
                "Implement 90-day automated purge policy for cancelled customer data",
                "Send retention notification emails to affected employees upon cancellation",
                "Build a self-service data deletion portal for individuals",
                "Formally appoint a qualified Data Protection Officer",
                "Sign Standard Contractual Clauses with US analytics sub-processor",
            ],
            "reasoning": (
                "GDPR Art.5(1)(e) requires personal data be kept no longer than necessary. "
                "Therefore indefinite retention after account cancellation violates this principle. "
                "GDPR Art.12-14 requires transparency — employees must be notified. "
                "GDPR Art.17 grants right to erasure, which requires a deletion mechanism. "
                "GDPR Art.37 mandates formal DPO appointment for processors of this scale. "
                "GDPR Art.46 requires adequate transfer safeguards — SCCs must be signed. "
                "Because all these violations exist simultaneously, the company must implement "
                "a comprehensive remediation plan covering retention, notification, deletion, "
                "DPO appointment, and international transfer mechanisms. "
                "I recommend prioritising the SCC signature and DPO appointment immediately."
            ),
            "regulation_references": [
                "GDPR Art.5", "GDPR Art.12", "GDPR Art.17", "GDPR Art.37", "GDPR Art.46"
            ],
            "confidence": 0.92,
        })
        r = result["reward"]
        print(f"\nReward breakdown:")
        print(f"  Issue Identification : {r['issue_identification_score']:.3f}")
        print(f"  Suggestion Quality   : {r['suggestion_quality_score']:.3f}")
        print(f"  Regulation Accuracy  : {r['regulation_accuracy_score']:.3f}")
        print(f"  Reasoning Quality    : {r['reasoning_quality_score']:.3f}")
        print(f"  FP Penalty           : {r['false_positive_penalty']:.3f}")
        print(f"  ─────────────────────────────")
        print(f"  TOTAL                : {r['total']:.3f}")
        print(f"\nFeedback: {result['observation']['feedback']}")
        print(f"Done: {result['done']}")
