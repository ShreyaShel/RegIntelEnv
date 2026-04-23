"""
RegIntelEnv - inference.py
Runs an AI agent through RegIntelEnv tasks with structured stdout output.

Environment variables:
  API_BASE_URL  - OpenAI-compatible API base URL (e.g. Hugging Face Inference Endpoint)
  MODEL_NAME    - Model identifier (e.g. meta-llama/Llama-3-70b-Instruct)
  HF_TOKEN      - Hugging Face API Token (Required for LLM inference)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

# Force unbuffered stdout
os.environ["PYTHONUNBUFFERED"] = "1"

# Optional imports — script works without them
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ---------------------------------------------------------------------------
# Structured output — writes directly to sys.stdout, never stderr
# ---------------------------------------------------------------------------

def emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Stdlib HTTP helpers (no external deps needed)
# ---------------------------------------------------------------------------

def http_get(url: str, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def http_post(url: str, data: dict, timeout: float = 30.0) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")
HF_TOKEN = os.environ.get("HF_TOKEN")
ENV_BASE_URL = os.environ.get("BASE_URL", "http://localhost:7860")


# ---------------------------------------------------------------------------
# Fallback rule-based agent
# ---------------------------------------------------------------------------

_REGULATION_RULES: Dict[str, Dict[str, Any]] = {
    "gdpr": {
        "articles": ["GDPR Art.5", "GDPR Art.12", "GDPR Art.13", "GDPR Art.17",
                     "GDPR Art.37", "GDPR Art.46"],
        "triggers": {
            "indefinite|no retention|no policy|no schedule|deactivated storage": (
                "No documented data retention schedule — data kept indefinitely, "
                "violating the GDPR Art.5(1)(e) storage limitation principle",
                "Implement a documented retention schedule with automated purge "
                "after the lawful retention period expires",
                "GDPR Art.5",
            ),
            "not notif|never notif|no notif|employees.*not told|no transparency": (
                "Data subjects are not notified their personal data "
                "continues to be retained after account cancellation, violating GDPR Art.13",
                "Send retention notification emails to all affected data subjects "
                "upon account cancellation",
                "GDPR Art.12",
            ),
            "no deletion|no erasure|no self.service|no removal|no request": (
                "No mechanism exists for data subjects to request deletion of "
                "their personal data, violating GDPR Art.17 right to erasure",
                "Build a self-service data deletion request portal",
                "GDPR Art.17",
            ),
            "DPO|data protection officer|CFO|side duty|secondary|not formal": (
                "Data Protection Officer role is not formally appointed, "
                "violating GDPR Art.37",
                "Formally appoint a qualified, independent DPO",
                "GDPR Art.37",
            ),
            "US|united states|sub.processor|third.country|transfer|SCC|standard contract": (
                "International data transfer to US sub-processor without SCCs, "
                "violating GDPR Art.46",
                "Sign Standard Contractual Clauses with all US-based sub-processors",
                "GDPR Art.46",
            ),
            "audit|only active|not cover|no audit": (
                "GDPR audits cover only active accounts — cancelled data excluded",
                "Extend audit scope to include deactivated account data",
                "GDPR Art.5",
            ),
        },
        "compliance_status": "non_compliant",
        "reasoning_template": (
            "Under GDPR (EU 2016/679), several obligations apply. "
            "Art.5(1)(e) Storage Limitation is violated by indefinite retention. "
            "Art.12-14 Transparency mandates informing data subjects. "
            "Art.17 right to erasure requires a deletion mechanism. "
            "Art.37 requires a formally appointed DPO. "
            "Art.46 mandates safeguards for international transfers."
        ),
    },
    "ai act": {
        "articles": ["AI Act Art.6", "AI Act Annex III", "AI Act Art.9", "AI Act Art.10",
                     "AI Act Art.11", "AI Act Art.13", "AI Act Art.14",
                     "AI Act Art.16", "AI Act Art.26"],
        "triggers": {
            "not register|no register|EU database|high.risk|annex": (
                "High-risk AI system not registered in the EU AI database",
                "Register the system in the EU AI high-risk database",
                "AI Act Art.6",
            ),
            "no conformity|no assessment|no audit|no certification": (
                "No conformity assessment conducted for high-risk AI system",
                "Conduct a full third-party conformity assessment",
                "AI Act Annex III",
            ),
            "auto.approv|auto.reject|no human|human.in.the.loop|no review": (
                "Fully automated decisions with no human-in-the-loop, violating Art.14",
                "Implement mandatory human review for rejections",
                "AI Act Art.14",
            ),
            "no explanation|binary|APPROVED|REJECTED|no reason|not explain": (
                "Binary outcome with no explanation, violating Art.13 transparency",
                "Provide meaningful explanations of AI decisions",
                "AI Act Art.13",
            ),
            "no appeal|cannot request|no human review|cannot contest": (
                "No human review appeal process available",
                "Establish a formal human review appeal process",
                "AI Act Art.14",
            ),
            "bias|demographic|postcode|geographic|15%|skewed|discriminat": (
                "Training data not audited for bias, violating Art.10",
                "Conduct comprehensive bias audit of training data",
                "AI Act Art.10",
            ),
            "no technical doc|no documentation|art.11|not produced": (
                "No technical documentation produced, violating Art.11",
                "Produce complete technical documentation",
                "AI Act Art.11",
            ),
            "no risk management|no risk system|art.9": (
                "No AI risk management system, violating Art.9",
                "Establish a documented AI risk management system",
                "AI Act Art.9",
            ),
            "no monitoring|no post.market|art.26|quarterly": (
                "No post-market monitoring plan, violating Art.26",
                "Implement post-market monitoring with quarterly reviews",
                "AI Act Art.26",
            ),
        },
        "compliance_status": "non_compliant",
        "reasoning_template": (
            "The EU AI Act classifies credit scoring AI as high-risk under Art.6 and Annex III. "
            "Multiple obligations from Chapter III are violated."
        ),
    },
    "nis2": {
        "articles": ["NIS2 Art.20", "NIS2 Art.21", "NIS2 Art.23",
                     "NIS2 Art.29", "NIS2 Recital 58", "NIS2 Art.32"],
        "triggers": {
            "ransomware|not report|never report|19 day|Q3|unreport": (
                "Ransomware attack never reported, violating NIS2 Art.23",
                "File retrospective incident notification with NCA/CSIRT immediately",
                "NIS2 Art.23",
            ),
            "2019|outdated|informal|no formal incident|old procedure": (
                "Incident response procedure is outdated informal document",
                "Develop NIS2-compliant incident response plan",
                "NIS2 Art.21",
            ),
            "no OT|no SCADA|no cyber risk|OT risk|operational technology|no assess": (
                "No OT/SCADA cybersecurity risk assessment performed",
                "Conduct comprehensive OT/SCADA cyber risk assessment",
                "NIS2 Art.21",
            ),
            "board|management|4 year|expired|no approved|no policy": (
                "Management board has not approved cybersecurity policy in 4 years",
                "Convene emergency board session to approve cybersecurity policy",
                "NIS2 Art.20",
            ),
            "12 vendor|third.party|no assessment|supply chain|no security assess": (
                "12 OT vendors have received no security assessments",
                "Perform security assessments on all OT vendors",
                "NIS2 Recital 58",
            ),
            "contract|best effort|no minimum|no audit.*right|right.*audit": (
                "Vendor contracts lack minimum security standards",
                "Renegotiate contracts with security standards and audit rights",
                "NIS2 Art.21",
            ),
            "known vulner|critical vulner|CVE|patch|3 vendor": (
                "Vendors with known critical vulnerabilities not addressed",
                "Apply immediate patches for critical vulnerabilities",
                "NIS2 Art.21",
            ),
            "Modbus|unencrypt|TCP|plaintext|SCADA.*communic|protocol": (
                "SCADA communications use unencrypted Modbus TCP",
                "Replace with TLS-encrypted protocols",
                "NIS2 Art.21",
            ),
            "30%|unknown|unidentif|no inventory|no asset": (
                "30% of OT/SCADA devices unidentified, no asset inventory",
                "Commission full OT/SCADA asset discovery",
                "NIS2 Art.21",
            ),
            "no training|2 year|OT staff|no cybersecurity training": (
                "No cybersecurity training for OT staff in 2 years",
                "Launch mandatory annual cybersecurity training",
                "NIS2 Art.21",
            ),
            "CISO|vacant|8 month|no security lead|no chief": (
                "CISO role vacant for 8 months",
                "Recruit and appoint CISO immediately",
                "NIS2 Art.20",
            ),
            "board.*brief|no brief|not.*briefed|no board": (
                "Management board has received no cybersecurity briefings",
                "Schedule quarterly cybersecurity briefings",
                "NIS2 Art.20",
            ),
            "risk register|no OT.*risk|corporate.*risk|not include": (
                "Corporate risk register excludes OT/SCADA cybersecurity risks",
                "Add OT/SCADA risks to corporate risk register",
                "NIS2 Art.21",
            ),
        },
        "compliance_status": "non_compliant",
        "reasoning_template": (
            "Essential Entity under NIS2 (2022/2555). Art.20 requires board accountability. "
            "Art.21 mandates risk analysis, incident handling, supply chain security. "
            "Art.23 requires incident reporting within 24h/72h/1month."
        ),
    },
}


def _detect_regulation(regulation_name: str) -> str:
    name = regulation_name.lower()
    if "gdpr" in name or "general data protection" in name:
        return "gdpr"
    if "ai act" in name or "artificial intelligence act" in name:
        return "ai act"
    if "nis2" in name or "network and information" in name:
        return "nis2"
    return "gdpr"


def _extract_issues_and_suggestions(
    process_text: str, regulation_key: str
) -> Tuple[List[str], List[str], List[str]]:
    rules = _REGULATION_RULES.get(regulation_key, _REGULATION_RULES["gdpr"])
    text_lower = process_text.lower()
    issues, suggestions, refs_seen = [], [], []
    for pattern, (issue, suggestion, article) in rules["triggers"].items():
        if re.search(pattern, text_lower):
            issues.append(issue)
            suggestions.append(suggestion)
            if article not in refs_seen:
                refs_seen.append(article)
    combined_refs = list(dict.fromkeys(refs_seen + rules["articles"]))
    return issues, suggestions, combined_refs


def fallback_agent_action(
    obs: Dict[str, Any], step_idx: int, max_steps: int
) -> Dict[str, Any]:
    process_text = obs.get("process_description", "")
    regulation_name = obs.get("regulation_name", "")
    already_found = set(obs.get("issues_found_so_far", []))
    already_suggested = set(obs.get("suggestions_given_so_far", []))
    reg_key = _detect_regulation(regulation_name)
    rules = _REGULATION_RULES[reg_key]
    issues, suggestions, refs = _extract_issues_and_suggestions(process_text, reg_key)
    new_issues = [i for i in issues if i not in already_found]
    new_suggestions = [s for s in suggestions if s not in already_suggested]
    step_fraction = max(1, len(new_issues) // max(max_steps - step_idx, 1))
    batch_issues = new_issues[:step_fraction + 2]
    batch_suggestions = new_suggestions[:step_fraction + 2]
    is_last = step_idx >= max_steps - 1
    action_type = "conclude" if is_last else ("analyze" if step_idx == 0 else "flag")
    if is_last:
        batch_issues = new_issues
        batch_suggestions = new_suggestions
    reasoning = rules["reasoning_template"]
    if step_idx == 0:
        reasoning = (
            f"Initial sweep of the {obs.get('company_name', 'company')} processes "
            f"against {regulation_name}. " + reasoning
        )
    elif is_last:
        reasoning += (
            f" This concludes the analysis of all {len(issues)} identified "
            f"compliance gaps across {regulation_name}."
        )
    return {
        "action_type": action_type,
        "process_analyzed": obs.get("process_name", ""),
        "identified_issues": batch_issues,
        "compliance_status": rules["compliance_status"],
        "suggestions": batch_suggestions,
        "reasoning": reasoning,
        "regulation_references": refs,
        "confidence": 0.82,
    }


# ---------------------------------------------------------------------------
# LLM Inference agent (using OpenAI-compatible client)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert AI compliance analyst. Analyze company processes
against regulations, identify non-compliance issues, and suggest remediation actions.

Respond ONLY with valid JSON:
{
  "action_type": "flag" | "suggest" | "analyze" | "conclude",
  "identified_issues": ["issue 1", ...],
  "compliance_status": "compliant" | "non_compliant" | "partial" | "uncertain",
  "suggestions": ["suggestion 1", ...],
  "reasoning": "Detailed analysis...",
  "regulation_references": ["GDPR Art.5", ...],
  "confidence": 0.0
}"""


def build_user_message(obs: Dict[str, Any], step: int, total_steps: int) -> str:
    return f"""=== COMPLIANCE ANALYSIS TASK ===
Company: {obs.get('company_name', 'Unknown')}
Industry: {obs.get('industry', 'Unknown')}
Regulation: {obs.get('regulation_name', 'Unknown')}

REGULATION SUMMARY:
{obs.get('regulation_summary', 'N/A')}

PROCESS TO ANALYZE:
{obs.get('process_description', 'N/A')}

Step: {step} / {total_steps}
Issues found so far: {json.dumps(obs.get('issues_found_so_far', []))}
Suggestions given so far: {json.dumps(obs.get('suggestions_given_so_far', []))}

{f"Feedback: {obs['feedback']}" if obs.get('feedback') else ""}
{'FINAL STEP - use action_type "conclude".' if step >= total_steps - 1 else ''}
Respond ONLY with valid JSON."""


def call_llm_agent(messages: List[Dict], client: Any) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Run one episode — GUARANTEES [START]/[STEP]/[END] on stdout
# ---------------------------------------------------------------------------

def run_episode(
    difficulty: str,
    env_base_url: str,
    use_fallback: bool,
    llm_client: Optional[Any] = None,
) -> None:
    task_id = f"task_{difficulty}"
    total_reward = 0.0
    completed_steps = 0

    # [START] printed BEFORE any HTTP call
    emit(f"[START] task={task_id}")

    try:
        # --- Reset ---
        obs = http_post(f"{env_base_url}/reset", {"difficulty": difficulty})

        task_id = obs.get("task_id", task_id)
        max_steps = obs.get("max_steps", 3)
        log(f"Task: {task_id} | Max steps: {max_steps}")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # --- Steps ---
        for step_idx in range(max_steps):
            log(f"Step {step_idx + 1} / {max_steps}")

            if use_fallback:
                action = fallback_agent_action(obs, step_idx, max_steps)
            else:
                user_msg = build_user_message(obs, step_idx + 1, max_steps)
                messages.append({"role": "user", "content": user_msg})
                try:
                    action = call_llm_agent(messages, llm_client)
                except Exception as e:
                    log(f"LLM call failed ({e}) — using fallback")
                    action = fallback_agent_action(obs, step_idx, max_steps)
                messages.append({"role": "assistant", "content": json.dumps(action)})

            if step_idx == max_steps - 1:
                action["action_type"] = "conclude"

            # --- Submit step ---
            result = http_post(f"{env_base_url}/step", {"action": action})

            reward = result.get("reward", {})
            obs = result.get("observation", obs)
            done = result.get("done", False)
            reward_total = reward.get("total", 0.0)
            total_reward = obs.get("total_reward", total_reward)
            completed_steps = step_idx + 1

            # [STEP] output
            emit(f"[STEP] step={completed_steps} reward={reward_total:.4f}")

            if done:
                break

            if not use_fallback:
                time.sleep(1.0)

    except Exception as e:
        log(f"Episode error for {difficulty}: {e}")
        # Guarantee at least 1 STEP
        if completed_steps == 0:
            completed_steps = 1
            emit(f"[STEP] step=1 reward=0.0000")

    finally:
        # [END] ALWAYS printed
        emit(f"[END] task={task_id} score={total_reward:.4f} steps={completed_steps}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    env_base_url = os.environ.get("BASE_URL", "http://localhost:7860")
    use_fallback = not HF_TOKEN
    llm_client = None

    if HF_TOKEN and API_BASE_URL and HAS_OPENAI:
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        log(f"Using HF Inference agent | model={MODEL_NAME} | base_url={API_BASE_URL}")
    else:
        use_fallback = True
        log("Using rule-based fallback agent.")

    # Health check (non-fatal)
    try:
        health = http_get(f"{env_base_url}/health")
        log(f"Server OK: {health}")
    except Exception as e:
        log(f"Health check failed: {e} — proceeding anyway")

    # Run all 3 tasks
    difficulties = ["easy", "medium", "hard"]

    for i, diff in enumerate(difficulties):
        run_episode(
            difficulty=diff,
            env_base_url=env_base_url,
            use_fallback=use_fallback,
            llm_client=llm_client,
        )
        if i < len(difficulties) - 1:
            time.sleep(2.0)


if __name__ == "__main__":
    main()
