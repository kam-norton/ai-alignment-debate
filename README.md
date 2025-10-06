# AI Project: Debate & Evaluation Framework

This repository provides a lightweight framework for running scripted evaluations of language models.  
It supports batch execution of experiments (e.g., debates, evaluations without judges, single responses) using OpenRouter-hosted models.

---

## Prerequisites

- **Conda** environment with Python 3.9+  
- An **OpenRouter API key** (sign up at [https://openrouter.ai](https://openrouter.ai))  

---

## Setup

1. Import and activate the conda environment, provided above (yml):

   ```bash
   conda activate alignment
   ```

2. Move into the project directory:

   ```bash
   cd .../ai-alignment-debate
   ```

3. Set your OpenRouter API key as an environment variable:

   **Windows (PowerShell):**
   ```powershell
   set OPENROUTER_API_KEY=sk-or-yourkeyhere
   ```

   **macOS / Linux (bash/zsh):**
   ```bash
   export OPENROUTER_API_KEY=sk-or-yourkeyhere
   ```

---

## Main Scripts

### 1. `run_matrix.py`
- Entry point to run a batch of experiments defined in the script.  
- It iterates through the `SCRIPTS` list and runs each with specified topics, models, and parameters.  
- Useful for structured, repeatable experiments.
- Can be adjusted and adapted by users.

### 2. `debate_nojudge.py`
- Runs a debate simulation **without** an evaluator/judge.  
- Configurable with model names and topic prompts inside the script.

### 3. `single_response.py`
- Gets a single model’s response for a topic.  
- Useful for sanity checks, debugging, or quick experiments.

---

## Example Usage

### Run the full matrix of experiments
```bash
python run_matrix.py
```

### Run a debate without a judge (with dummy variables)
```bash
python debate_nojudge.py --topic "Is AI alignment possible?" --model "openai/gpt-5-nano"
```

### Run a single response test
```bash
python single_response.py --topic "What is the capital of France?" --model "openai/gpt-4.1-mini"
```

---

## Notes

- **API Key**: Every script requires the `OPENROUTER_API_KEY` environment variable to be set.  
- **Script configs**:  
  - `run_matrix.py` uses a pre-defined list of scripts (`debate_eval.py`, `debate_eval_norubric.py`, `debate_nojudge.py`, etc.).  
  - `debate_nojudge.py` and `single_response.py` accept arguments for models and topics (see examples above).  
- **Logs**: Every result is stored in respective log files in the 'logs' directory, and the MATLAB scripts provide a way to use or process the results and information. 
---

## Next Steps

- Add your own scripts to the `SCRIPTS` list in `run_matrix.py` to integrate them into the experiment matrix.  
- Tweak the debate or evaluation logic inside each script to explore new configurations.  

---
