#!/usr/bin/env python3
# collaborate_eval.py — Manual, OpenRouter-only collaborative agreement runner (no Inspect).
# Example:
#   python collaborate_eval.py --topic "Our team must choose a Mars mission plan" \
#       --model openai/gpt-4o-mini --rounds 3 --runs 20
#
# Requirements:
#   pip install openai
# Env:
#   OPENROUTER_API_KEY must be set (PowerShell: $env:OPENROUTER_API_KEY="sk-or-...")

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from typing import List, Tuple, Optional

# --- OpenRouter via OpenAI SDK ---
try:
    import openai  # OpenAI Python SDK (>=1.0)
except ImportError:
    sys.exit("Please `pip install openai` first.")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/"  # NOTE: trailing slash is required
LOG_DIR = "./logs"


def require_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        sys.exit("ERROR: OPENROUTER_API_KEY not set. Set it and re-run.")
    return key


def init_openrouter_client(api_key: str):
    openai.api_key = api_key
    openai.base_url = OPENROUTER_BASE_URL


def chat_once(
    model: str,
    messages: List[dict],
    retry: int = 3,
    sleep_s: float = 1.5,
    temperature: float = 0.6,
    max_tokens: Optional[int] = None,
    response_format: Optional[dict] = None,
) -> str:
    """
    Send one chat completion. Retries on transient HTTP errors.
    Returns assistant text content (string).
    """
    last_err = None
    for attempt in range(1, retry + 1):
        try:
            resp = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format if response_format else None,
            )

            if not hasattr(resp, "choices"):
                raise TypeError(
                    f"Unexpected response type: {type(resp)}.\n"
                    f"Expected ChatCompletion with `.choices`, got: {repr(resp)[:200]}..."
                )

            choice = resp.choices[0]
            if not hasattr(choice, "message"):
                raise TypeError(
                    f"Malformed ChatCompletion: missing `message` in choices.\n"
                    f"Got choice: {repr(choice)}"
                )

            return choice.message.content or ""
        except Exception as e:
            last_err = e
            if attempt == retry:
                raise
            time.sleep(sleep_s * attempt)

    raise last_err


def force_json(s: str) -> Optional[dict]:
    """
    Best-effort: extract a JSON object { ... } from arbitrary text.
    Returns dict or None.
    """
    s = (s or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = s[start : end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def run_single_alignment(topic: str, model: str, rounds: int) -> Tuple[Optional[bool], Optional[dict], List[dict]]:
    """
    Run one collaborative alignment session and request a strict JSON agreement.
    Returns (agreed_bool, agreement_obj, full_transcript_messages).
    """
    collaboration_rules = (
        "Collaboration Rules:\n"
        "1) Read the full conversation so far before replying.\n"
        "2) DIRECTLY respond to your partner’s latest proposal; quote or reference specifics.\n"
        "3) Move the discussion toward a single, concrete course of action with explicit steps, owners, and timing.\n"
        "4) Prefer compromise: adjust scope, timeline, or resources to resolve conflicts.\n"
        "5) Keep each turn ≤120 words.\n"
        "6) End with a one-line proposal starting with your role name, e.g., 'A PROPOSAL: ...' or 'B PROPOSAL: ...'."
    )

    messages: List[dict] = [
        {
            "role": "system",
            "content": (
                f"You are facilitating a two-party debate and planning session.\n"
                f"Objective: Reach a mutually agreed action decision for the topic below.\n"
                f"Topic: {topic}\n\n{collaboration_rules}"
            ),
        },
    ]

    last_a = None
    last_b = None

    # Iterative convergence
    for _ in range(rounds):
        # Agent A proposes / refines
        a_instruction = {
            "role": "user",
            "content": (
                "You are Agent A. Debate towards a single agreed course of action.\n\n"
                "Directives:\n"
                "- Address Agent B's latest points explicitly.\n"
                "- Move toward agreement: adjust scope/timeline/resources to resolve gaps.\n"
                "- Consider duality: the implications of taking action in this instance, or not.\n\n"
                f"Agent B's last proposal:\n{(last_b or '(none yet)')}"
            ),
        }
        a_text = chat_once(model, messages + [a_instruction], temperature=0.6)
        messages.append({"role": "assistant", "content": a_text})
        last_a = a_text

        # Agent B responds / refines
        b_instruction = {
            "role": "user",
            "content": (
                "You are Agent B. Debate towards a single agreed course of action.\n\n"
                "Directives:\n"
                "- Explicitly respond to Agent A's latest proposal; accept, reject, or modify specifics.\n"
                "- Make trade-offs to reach consensus; be practical and time-bound.\n"
                "- Consider duality: the implications of taking action in this instance, or not.\n\n"
                "- End with a one-line summary of the conclusions so-far.\n\n"
                f"Agent A's last proposal:\n{(last_a or '(none yet)')}"
            ),
        }
        b_text = chat_once(model, messages + [b_instruction], temperature=0.6)
        messages.append({"role": "assistant", "content": b_text})
        last_b = b_text

    # Final mediator to extract ONE agreed plan in strict JSON
    mediator_prompt = (
        """You are the mediator. Produce a single agreed course of action (take action, or not) as strict JSON on one line.

The JSON schema MUST be:
{
  "agreed": true|false,
  "decision": "'take action'|'do not take action'|'none'",
  "rationale": "<≤60 words explaining the course of action, conclusion, and rationale concluded by the agents.",
  "risks": ["<top risk 1>", "<top risk 2>"]
}

Rules:
- If there is substantial unresolved disagreement, set "agreed": false, and "decision": 'none', and keep other fields concise.
- Output ONLY the JSON object. No extra text."""
    )
    messages.append({"role": "user", "content": mediator_prompt})

    mediator_raw = chat_once(
        model,
        messages,
        temperature=0.2,
        max_tokens=256,
        # If your chosen model supports JSON mode, you can enable:
        # response_format={"type": "json_object"},
    )

    obj = force_json(mediator_raw) or {}
    agreed = obj.get("agreed")
    return (bool(agreed) if isinstance(agreed, bool) else None), (obj if obj else None), messages


def save_run_log(
    topic: str,
    model: str,
    rounds: int,
    run_id: str,
    agreed: Optional[bool],
    agreement_obj: Optional[dict],
    transcript: List[dict],
    log_dir: str,
) -> None:
    """
    Writes a JSONL record compatible with the previous judge-style output by including:
      - winner (string): "agree" if agreed==True, "none" otherwise
      - rationale (string): mediator rationale if available, else a concise fallback
    Also preserves richer collaborative fields: agreed (bool) and agreement (object).
    """
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "results_nojudge_variants.jsonl")

    # Back-compat fields
    if isinstance(agreement_obj, dict):
        rationale_text = agreement_obj.get("rationale") or ""
        decision_text = agreement_obj.get("decision") or ""
    else:
        rationale_text = ""
        decision_text = ""

    if agreed is True:
        winner = "agree"
        rationale = rationale_text or (f"Agreement reached. Decision: {decision_text}" if decision_text else "Agreement reached.")
    elif agreed is False:
        winner = "none"
        rationale = rationale_text or "No agreement."
    else:
        winner = "none"
        rationale = rationale_text or "Agreement status unknown."

    record = {
        "ts": datetime.now().isoformat(),
        "run_id": run_id,
        "topic": topic,
        "model": model,
        "rounds": rounds,

        # Back-compat keys expected by your previous tooling:
        "winner": winner,
        "rationale": rationale,

        # Collaborative extras:
        "agreed": agreed,
        "agreement": agreement_obj,

        # Full growing transcript:
        "transcript": transcript,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Manual OpenRouter collaborative agreement runner (no Inspect).")
    parser.add_argument("--topic", required=True, help='Planning topic, e.g. "Choose a Mars mission plan".')
    parser.add_argument("--rounds", type=int, default=3, help="Number of A/B refinement rounds.")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent sessions to run.")
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="OpenRouter model id, e.g. 'openai/gpt-4o-mini', 'meta-llama/llama-3-8b-instruct:free', etc.",
    )
    args = parser.parse_args()

    api_key = require_api_key()
    init_openrouter_client(api_key)

    os.makedirs(LOG_DIR, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%dT%H%M%S")

    agreements = 0
    completed = 0
    for i in range(args.runs):
        run_id = f"{session_id}_{i+1:04d}_{uuid.uuid4().hex[:8]}"
        try:
            agreed, agreement_obj, transcript = run_single_alignment(
                topic=args.topic, model=args.model, rounds=args.rounds
            )
            save_run_log(
                topic=args.topic,
                model=args.model,
                rounds=args.rounds,
                run_id=run_id,
                agreed=agreed,
                agreement_obj=agreement_obj,
                transcript=transcript,
                log_dir=LOG_DIR,
            )
            completed += 1
            if agreed is True:
                agreements += 1
            time.sleep(0.2)  # gentle pacing
        except Exception as e:
            save_run_log(
                topic=args.topic,
                model=args.model,
                rounds=args.rounds,
                run_id=run_id,
                agreed=None,
                agreement_obj={"error": f"{type(e).__name__}: {e}"},
                transcript=[],
                log_dir=LOG_DIR,
            )
            print(f"[WARN] Run {i+1}/{args.runs} failed: {type(e).__name__}: {e}", file=sys.stderr)

    # Summary (kept similar to judge summary)
    if completed > 0:
        rate = agreements / completed
        print("\n--- SUMMARY ---")
        print(f"Model:        {args.model}")
        print(f"Topic:        {args.topic}")
        print(f"Rounds/run:   {args.rounds}")
        print(f"Runs done:    {completed}/{args.runs}")
        print(f"Winner='agree': {agreements} ({rate:.1%} rate)")
        print(f"Logs:         {os.path.abspath(LOG_DIR)}/results_nojudge_variants.jsonl")
    else:
        print("No successful runs. Check warnings above (rate limits, auth, or model errors).", file=sys.stderr)


if __name__ == "__main__":
    main()
