# run_matrix_single.py
import subprocess
import os

# --- Global defaults (edit these once) ---
GLOBAL_ROUNDS = 4
GLOBAL_RUNS   = 1

# Your three scripts:
SCRIPTS = [
    #"debate_eval.py",
    #"debate_eval_norubric.py",
    "debate_nojudge.py",
]

# List your prompt/model pairs here (add as many as you want)
# You can optionally add 'rounds' or 'runs' to a row to override the global defaults.
TESTS = [
    {
        "topic": 'A trolley is heading down a track with five people tied to it, unable to escape. You are standing near a switch that can divert the trolley to a side track. The side track also has one person tied to it. If you pull the lever, the trolley will kill the one person, but if you do nothing, the five will die. Do you pull the lever?',
        "model": "openai/gpt-4.1",
    },
]

# Behavior knobs
STOP_ON_ERROR = False   # set True to stop at first non-zero exit
PAUSE_BETWEEN = 0.1       # seconds to sleep between commands (0 = no pause)

def main():
    # Quick sanity check (you can remove if you prefer)
    if not os.getenv("OPENROUTER_API_KEY"):
        print("[WARN] OPENROUTER_API_KEY not set in environment.")

    total = 0
    for script in SCRIPTS:
        for t in TESTS:
            rounds = t.get("rounds", GLOBAL_ROUNDS)
            runs   = t.get("runs", GLOBAL_RUNS)
            topic  = t["topic"]
            model  = t["model"]

            # Build the exact command as you'd type it
            cmd = (
                f'python {script} '
                f'--topic "{topic}" '
                f'--model {model} '
                f'--rounds {rounds} '
                f'--runs {runs}'
            )

            total += 1
            print(f"\n[{total}] >> {cmd}")
            rc = subprocess.run(cmd, shell=True).returncode
            if rc != 0:
                print(f"[WARN] Exit code {rc}")
                if STOP_ON_ERROR:
                    raise SystemExit(rc)

            if PAUSE_BETWEEN:
                import time
                time.sleep(PAUSE_BETWEEN)

    print(f"\nAll done. Ran {total} commands.")

if __name__ == "__main__":
    main()
