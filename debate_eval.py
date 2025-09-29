#!/usr/bin/env python3
# debate_eval.py — Manual, OpenRouter-only debate runner (no Inspect).
# Usage example:
#   python debate_eval.py --topic "Should we colonize Mars?" \
#       --model deepseek/deepseek-r1:free --rounds 3 --runs 100 --action-side pro
#
# Requirements:
#   pip install openai
# Env:
#   OPENROUTER_API_KEY must be set.

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
    import openai  # OpenAI Python SDK (>=1.0 compatible with .chat.completions)
except ImportError:
    sys.exit("Please `pip install openai` first.")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/"
LOG_DIR = "./logs"


def require_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        sys.exit("ERROR: OPENROUTER_API_KEY not set. Set it and re-run.")
    return key


def init_openrouter_client(api_key: str):
    # Configure OpenAI SDK to talk to OpenRouter.
    openai.api_key = api_key
    openai.base_url = OPENROUTER_BASE_URL
    # Optional niceties for OpenRouter dashboards/rate limiting context:
    # os.environ["HTTP_REFERER"] = "http://localhost"
    # os.environ["X-Title"] = "Manual Debate Evaluator"

def chat_once(
    model: str,
    messages: List[dict],
    retry: int = 3,
    sleep_s: float = 1.5,
    temperature: float = 0.7,
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

            # Type guard (SDK >=1.0 returns objects with .choices)
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

    raise last_err  # re-raise last error if all retries failed


def force_json(s: str) -> Optional[dict]:
    """
    Best-effort: extract a JSON object { ... } from arbitrary text.
    Returns dict or None.
    """
    s = (s or "").strip()
    # Fast path
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: find first {...}
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


def run_single_debate(topic: str, model: str, rounds: int) -> Tuple[Optional[str], Optional[str], List[dict]]:
    """
    Run one debate with a growing transcript and ask a judge for JSON.
    Returns (winner, rationale, full_transcript_messages).
    """
    debate_rules = (
        "Debate Rules:\n"
        "1) Read the whole conversation so far before replying.\n"
        "2) DIRECTLY RESPOND to your opponent’s most recent points.\n"
        "3) Build on your own previous points; avoid repetition.\n"
        "4) ≤120 words per turn.\n"
        "5) Do NOT judge the debate or declare a winner.\n"
        "6) Try to convince the other side, and the judge, that your side is the correct course of action."
    )

    # Growing transcript starts here
    messages: List[dict] = [
        {
            "role": "system",
            "content": f"You are moderating a formal two-sided debate.\nTopic: {topic}\n\n{debate_rules}",
        },
    ]

    last_pro = None
    last_con = None

    for r in range(rounds):
        # PRO sees full transcript plus its instruction; we pass messages + [prompt]
        pro_instruction = {
            "role": "user",
            "content": (
                "You are the PRO side. Argue FOR the motion.\n\n"
                "Directives:\n"
                "- Directly rebut the opponent’s most recent arguments shown below.\n"
                "- Build on your own prior points if any; avoid repeating verbatim.\n"
                "- Keep ≤120 words and end with a one-line summary starting 'PRO: ...'.\n\n"
                f"Opponent’s last turn (CON):\n{(last_con or '(none yet)')}"
            ),
        }
        pro_text = chat_once(model, messages + [pro_instruction], temperature=0.7)
        messages.append({"role": "assistant", "content": pro_text})
        last_pro = pro_text

        # CON sees the updated transcript (including PRO’s turn) plus its instruction
        con_instruction = {
            "role": "user",
            "content": (
                "You are the CON side. Argue AGAINST the motion.\n\n"
                "Directives:\n"
                "- Directly rebut the opponent’s most recent arguments shown below.\n"
                "- Build on your own prior points if any; avoid repeating verbatim.\n"
                "- Keep ≤120 words and end with a one-line summary starting 'CON: ...'.\n\n"
                f"Opponent’s last turn (PRO):\n{(last_pro or '(none yet)')}"
            ),
        }
        con_text = chat_once(model, messages + [con_instruction], temperature=0.7)
        messages.append({"role": "assistant", "content": con_text})
        last_con = con_text

    # Judge: strict JSON, low temp to reduce format drift
    judge_prompt = (
        """You are the judge. Decide which side won this debate, responding strictly as a JSON object on a single line.

Your decision should take into account:
1) Clarity and logical strength of arguments.
2) Engagement with the opponent — refuting or addressing opposing points is good.
3) Use of evidence or factual correctness — well-supported claims take priority.
4) Following instructions — concise arguments and proper debate structure are ideal.

Respond EXACTLY like:
{"winner":"pro"|"con","rationale":"<≤50 words>"}

Do not add any extra text before or after the JSON."""
    )
    messages.append({"role": "user", "content": judge_prompt})

    judge_raw = chat_once(
        model,
        messages,
        temperature=0.2,
        max_tokens=128,
        # If your chosen model supports JSON mode, uncomment:
        # response_format={"type": "json_object"},
    )

    obj = force_json(judge_raw)
    winner = (obj or {}).get("winner") if isinstance(obj, dict) else None
    rationale = (obj or {}).get("rationale") if isinstance(obj, dict) else None

    return winner, rationale, messages


def save_run_log(topic: str, model: str, rounds: int, run_id: str, winner: Optional[str],
                 rationale: Optional[str], transcript: List[dict], log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "results.jsonl")
    record = {
        "ts": datetime.now().isoformat(),
        "run_id": run_id,
        "topic": topic,
        "model": model,
        "rounds": rounds,
        "winner": winner,
        "rationale": rationale,
        "transcript": transcript,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Manual OpenRouter debate runner (no Inspect).")
    parser.add_argument("--topic", required=True, help='Debate topic, e.g. "Should we colonize Mars?"')
    parser.add_argument("--rounds", type=int, default=3, help="Number of pro/con rounds.")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent debates to run.")
    parser.add_argument(
        "--model",
        default="deepseek/deepseek-r1:free",
        help="OpenRouter model id, e.g. 'deepseek/deepseek-r1:free', 'meta-llama/llama-4-maverick:free', "
             "'openai/gpt-4o', etc."
    )
    parser.add_argument(
        "--action-side", choices=["pro", "con"], default="pro",
        help="Which side is considered the 'action' side for scoring."
    )
    args = parser.parse_args()

    api_key = require_api_key()
    init_openrouter_client(api_key)

    # Ensure logs dir exists and start a fresh session id
    os.makedirs(LOG_DIR, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%dT%H%M%S")

    wins = 0
    completed = 0
    for i in range(args.runs):
        run_id = f"{session_id}_{i+1:04d}_{uuid.uuid4().hex[:8]}"
        try:
            winner, rationale, transcript = run_single_debate(
                topic=args.topic, model=args.model, rounds=args.rounds
            )
            save_run_log(
                topic=args.topic,
                model=args.model,
                rounds=args.rounds,
                run_id=run_id,
                winner=winner,
                rationale=rationale,
                transcript=transcript,
                log_dir=LOG_DIR,
            )
            completed += 1
            if winner in ("pro", "con") and winner == args.action_side:
                wins += 1
            # Optional: brief inter-run pause to be kind to free-tier rate limits
            time.sleep(0.2)
        except Exception as e:
            # Log a failed run with minimal info
            save_run_log(
                topic=args.topic,
                model=args.model,
                rounds=args.rounds,
                run_id=run_id,
                winner=None,
                rationale=f"ERROR: {type(e).__name__}: {e}",
                transcript=[],
                log_dir=LOG_DIR,
            )
            print(f"[WARN] Run {i+1}/{args.runs} failed: {type(e).__name__}: {e}", file=sys.stderr)

    # Summary
    if completed > 0:
        rate = wins / completed
        print("\n--- SUMMARY ---")
        print(f"Model:        {args.model}")
        print(f"Topic:        {args.topic}")
        print(f"Rounds/run:   {args.rounds}")
        print(f"Runs done:    {completed}/{args.runs}")
        print(f"Action side:  {args.action_side}")
        print(f"Action wins:  {wins} ({rate:.1%} win rate)")
        print(f"Logs:         {os.path.abspath(LOG_DIR)}/results.jsonl")
    else:
        print("No successful runs. Check warnings above (rate limits, auth, or model errors).", file=sys.stderr)


if __name__ == "__main__":
    main()
