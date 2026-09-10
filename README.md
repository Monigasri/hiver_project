# Hiver AI Customer Support Agent

## Overview

Evidence-first Python prototype for one support brand in Kaggle's Customer Support on Twitter data. It predicts a controlled intent, retrieves prior customer-to-agent resolutions, drafts from that evidence, and returns `AUTO_HANDLE` or `ESCALATE` with an explanation.

The primary user interface is a React + Vite dashboard in `frontend/`; the service is FastAPI in `backend/`. Legacy Streamlit files are retained only for optional local annotation/review.

## Problem

Historical support replies are useful evidence, but not a policy source. The agent has no account, financial, ticketing, or other external side effects.

## Selected Brand

Pending: no dataset is present. `scripts/prepare_data.py` writes the actual brand and observed statistics to `data/processed/selected_brand.json`, using customer volume, agent reply volume, and replied-conversation volume rather than popularity alone.

## Dataset

Place the downloaded Kaggle CSV in `data/raw/`. `--limit` creates a deterministic row-prefix sample. Cleaning preserves original text, normalizes timestamps, removes duplicate tweet IDs and empty text, and pairs direct outbound replies with inbound parents.

## Architecture

`prepare → conversation-ID split → intent classifier → semantic retrieval → escalation → grounded draft`. Golden conversation IDs are excluded from training and retrieval.

## Intent Taxonomy

The supplied eight-intent YAML is review-required starter material, not claimed selected-brand research. Review it against actual samples and maintain definitions, inclusion/exclusion rules, and representative data examples before human annotation.

## Retrieval

Sentence Transformer embeddings with FAISS are preferred. A TF-IDF fallback supports offline inspection. Evidence contains historical customer text, response, similarity, and conversation ID.

## Response Generation

The offline draft reuses only the closest historical response. It does not invent policy, credits, refunds, URLs, timelines, or actions. The optional LLM rubric is separate from routing.

## Escalation

Security/fraud/legal terms, low classifier confidence, and weak evidence escalate. Configure thresholds only using development data.

## Evaluation

`golden_set.csv` must contain 150–250 human-confirmed intent and escalation labels. `development_labels.csv` is independently human-labelled baseline training data. Evaluation refuses incomplete labels or conversation-ID overlap.

## Baselines

Majority class, TF-IDF + Logistic Regression, and the AI agent run on the exact same golden IDs. Outputs include macro/per-intent metrics, matrices, agent predictions, escalation false negatives, plots, and pattern-selected failure examples.

## Results

Pending real data and labels. No statistics, scores, agreement, or failures have been claimed.

## LLM-as-Judge

Optional structured 1–5 ratings: correctness, grounding, helpfulness, brand appropriateness, safety, escalation appropriateness, and overall. It saves concise reasons only.

## Human Agreement

After evaluation: `python scripts/create_human_judge_template.py`. Complete the 50 independent ratings, then calculate exact agreement, quadratic weighted kappa, and Spearman correlation.

## Failure Analysis

The evaluator selects the five highest-frequency actual intent-confusion patterns and preserves representative IDs. It does not invent causes or examples.

## What Is Misleading About the Headline Number?

Accuracy can conceal class imbalance, unsafe escalation false negatives, manual-sample uncertainty, distribution shift, retrieval mismatch, and the difference between offline labels and customer satisfaction. Compare macro-F1 with the majority baseline; LLM-judge agreement is not customer impact.

## Installation

Python 3.11+ is required.

```powershell
cd D:\hiver-project\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd D:\hiver-project\frontend
npm install
```

## Environment Variables

Copy `.env.example` to `.env`. `OPENAI_API_KEY` is only required for LLM judging. `SUPPORT_BRAND` overrides automatic selection.

## Reproducing Results

After human review, this fixed-seed sample workflow targets under 15 minutes, excluding first model download:

```powershell
cd D:\hiver-project
backend\.venv\Scripts\python.exe scripts\prepare_data.py --limit 250000 --golden-size 200
backend\.venv\Scripts\python.exe scripts\discover_intents.py
backend\.venv\Scripts\python.exe scripts\create_golden_template.py
# Human review: golden_set.csv and development_labels.csv
backend\.venv\Scripts\python.exe scripts\build_index.py
backend\.venv\Scripts\python.exe scripts\run_evaluation.py
backend\.venv\Scripts\python.exe scripts\plot_evaluation.py
backend\.venv\Scripts\python.exe -m pytest -q
```

## Running the Agent

`backend\.venv\Scripts\python.exe scripts\run_agent.py --text "My payment is failing"`

## Running the Demo

In one PowerShell terminal:

```powershell
cd D:\hiver-project\backend
.\.venv\Scripts\Activate.ps1
uvicorn app:app --reload --port 8001
```

In another:

```powershell
cd D:\hiver-project\frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The frontend calls `http://127.0.0.1:8001` by default; override it with `VITE_API_URL`. `GET /health`, `GET /api/info`, `POST /api/agent`, `POST /api/evaluate`, and `POST /api/golden` are available. Example:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8001/api/agent -ContentType 'application/json' -Body '{"message":"I cannot access my account"}'
```

## Project Structure

`src/` application; `scripts/` commands; `evaluation/` metrics/judge; `data/golden/` labels; `report/` take-home material.

## Limitations

No raw dataset, selected brand, manual labels, results, judge scores, or failure examples exist in this checkout. Direct reply pairing does not recover every multi-turn thread.

## Future Work

Conversation-aware reranking, threshold calibration, temporal evaluation, a larger independent human panel, and monitored feedback.
