"""
inference.py — Baseline inference script for Email Triage OpenEnv.

Structured stdout logs in [START] / [STEP] / [END] format.
Uses OpenAI client for all LLM calls.
"""

import os
import json
import sys
import time
import requests
from openai import OpenAI

# ── Environment variables (EXACTLY as required by checklist) ──────────────────
# Defaults are set ONLY for API_BASE_URL and MODEL_NAME (NOT HF_TOKEN)
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-api-base-url>")
MODEL_NAME   = os.getenv("MODEL_NAME", "<your-active-model-name>")
HF_TOKEN     = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# All LLM calls use the OpenAI client configured via these variables
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

TASK_IDS = ["task_easy", "task_medium", "task_hard"]


# ── Environment helpers ───────────────────────────────────────────────────────

def env_reset(task_id: str, seed: int = 42) -> dict:
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict) -> dict:
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ── LLM prompt & call ─────────────────────────────────────────────────────────

def build_prompt(obs: dict) -> str:
    return f"""You are an expert customer support triage agent.

Read the email below and respond with a JSON action object ONLY — no explanation, no markdown.

TASK: {obs['task_description']}

EMAIL:
Subject: {obs['subject']}
From: {obs['sender']}
Date: {obs['timestamp']}
Attachments: {obs.get('attachments', [])}

Body:
{obs['body']}

Respond with ONLY valid JSON:
{{
  "urgency": "<low|medium|high|critical>",
  "department": "<billing|technical|general|escalation|sales>",
  "reply_draft": "<professional reply or null>",
  "escalate": <true|false>,
  "tags": ["<tag1>", "<tag2>"]
}}

urgency: low=general question, medium=billing/minor, high=broken feature, critical=security/legal/threat
department: billing=invoice/payment, technical=bugs/API, general=feedback, escalation=legal/security/angry-VIP, sales=upgrade/pricing
escalate: true only for security issues, legal threats, or unresolved VIP complaints"""


def call_llm(prompt: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a customer support triage expert. Always respond with valid JSON only, no markdown."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=600,
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(task_id: str, seed: int = 42) -> dict:
    obs = env_reset(task_id=task_id, seed=seed)
    email_id = obs["email_id"]

    # Stdout logs follow the required structured format (START/STEP/END) exactly
    print(json.dumps({
        "event": "[START]",
        "task_id": task_id,
        "email_id": email_id,
        "subject": obs["subject"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))
    sys.stdout.flush()

    prompt = build_prompt(obs)
    action = call_llm(prompt)

    # Safe defaults
    action.setdefault("urgency", "low")
    action.setdefault("department", "general")
    action.setdefault("reply_draft", None)
    action.setdefault("escalate", False)
    action.setdefault("tags", [])

    result = env_step(action)
    reward = result["reward"]

    print(json.dumps({
        "event": "[STEP]",
        "task_id": task_id,
        "email_id": email_id,
        "action": action,
        "reward": reward,
        "done": result["done"],
        "grader_details": result.get("info", {}).get("grader_details", {}),
    }))
    sys.stdout.flush()

    print(json.dumps({
        "event": "[END]",
        "task_id": task_id,
        "email_id": email_id,
        "final_reward": reward,
        "status": "success",
    }))
    sys.stdout.flush()

    return {"task_id": task_id, "reward": reward}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(json.dumps({
        "event": "[START]",
        "run": "baseline_inference",
        "model": MODEL_NAME,
        "env_url": ENV_URL,
        "tasks": TASK_IDS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))
    sys.stdout.flush()

    scores = []
    for task_id in TASK_IDS:
        try:
            result = run_task(task_id, seed=42)
            scores.append(result)
        except Exception as e:
            print(json.dumps({
                "event": "[STEP]",
                "task_id": task_id,
                "error": str(e),
                "reward": 0.0,
            }))
            print(json.dumps({
                "event": "[END]",
                "task_id": task_id,
                "final_reward": 0.0,
                "status": "error",
                "error": str(e),
            }))
            sys.stdout.flush()
            scores.append({"task_id": task_id, "reward": 0.0})

    avg = round(sum(s["reward"] for s in scores) / len(scores), 4)

    print(json.dumps({
        "event": "[END]",
        "run": "baseline_inference",
        "scores": scores,
        "average_reward": avg,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
