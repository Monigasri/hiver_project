# Hiver AI Customer Support Agent — Engineering Report

## 1. Problem Framing

Customer support teams handle repetitive questions on social channels. Historical resolutions contain useful patterns, but must not be treated as authoritative policy. This system classifies an incoming AmazonHelp-style message, retrieves similar past resolutions, drafts a grounded reply, and explicitly chooses **AUTO_HANDLE** vs **ESCALATE**.

## 2. Why This Is Useful for Hiver / the Brand

Hiver routes shared inboxes and customer conversations. An evidence-first agent can suggest drafts and flag risky messages for humans, reducing handle time while keeping escalation explicit for fraud, account access, and ambiguous cases.

## 3. System Architecture

```
Customer message → TF-IDF taxonomy classifier → FAISS retrieval (39k cases)
                → deterministic escalation policy → grounded reply (LLM optional)
                → structured JSON + historical evidence
```

Frontend: React (Vite) · Backend: FastAPI · Data: chunked CSV pipeline · Index: Sentence-Transformers + FAISS

## 4. Data and Sampling

| Field | Value |
|-------|-------|
| Source | Kaggle *Customer Support on Twitter* (`twcs.csv`, 516MB) |
| Sample | Deterministic first **500,000** rows, chunk size 100,000, seed 42 |
| Brand | **AmazonHelp** (selection score on outbound + direct replies) |
| Conversations | 39,299 paired customer→agent direct replies |
| Retrieval corpus | 38,899 (golden 200 held out at conversation ID) |

## 5. Intent Taxonomy

Ten keyword-priority intents derived from AmazonHelp corpus review:

| Intent | Count |
|--------|------:|
| delivery_problem | 8,247 |
| order_status | 3,906 |
| return_or_refund | 1,313 |
| prime_membership | 1,599 |
| account_or_security | 1,074 |
| payment_or_pricing | 834 |
| product_availability | 1,088 |
| technical_or_content | 1,992 |
| customer_service | 1,786 |
| other_or_unclear | 17,260 |

`other_or_unclear` is large because many tweets lack strong keyword signal; narrowing further without human review risks false precision.

## 6. Retrieval Approach

`all-MiniLM-L6-v2` embeddings with FAISS inner-product search. Golden conversation IDs are excluded from the retrieval corpus. Evidence returns conversation ID, customer text, agent reply, and similarity score.

## 7. Reply Generation

Without `OPENAI_API_KEY`: returns the closest historical AmazonHelp reply plus an explicit fallback note. With API key: LLM drafts from evidence-only prompt forbidding invented refunds, account actions, or policy claims.

## 8. Escalation Policy

Deterministic rules escalate on: security/fraud/legal/PII keywords, account_or_security intent, low classifier confidence (<0.45), weak retrieval similarity (<0.35), very short ambiguous messages, and unsupported action requests without strong evidence.

## 9. Evaluation Methodology

- **Golden set:** 200 conversation-held-out examples (`golden_set.csv`); **HUMAN LABELING REQUIRED** for intent + escalation fields.
- **Baselines:** majority class, TF-IDF + Logistic Regression, AI agent — same golden IDs.
- **Retrieval:** Recall@3, Precision@3, MRR with intent-coherent relevance (documented in `evaluation/retrieval_eval.py`).
- **LLM judge:** optional 1–5 rubric; requires API key.
- **Human agreement:** template for 50 paired ratings; **HUMAN RATINGS REQUIRED FOR AGREEMENT ANALYSIS**.

## 10. Baselines

Implemented in `evaluation/baselines.py` and orchestrated by `scripts/run_evaluation.py`. Metrics: accuracy, macro P/R/F1, per-intent, confusion matrix; escalation accuracy/P/R/F1 and false negatives.

## 11. Results

**Intent / escalation headline metrics:** pending — `golden_set.csv` has machine `suggested_intent` only; `gold_intent` and `gold_escalation` are empty until human verification.

**Retrieval (proxy-label eval, k=3):** run `python scripts/run_retrieval_eval.py` after index build. Proxy labels use the same deterministic taxonomy; treat as diagnostic, not human ground truth.

## 12. Failure Modes

Framework in `scripts/run_evaluation.py` selects top intent-confusion patterns from real golden mispredictions. **Pending human labels** — no fabricated failure examples reported.

## 13. What is misleading about my headline number?

Any future accuracy headline would hide: (1) 500k deterministic sample vs full 3M dataset, (2) single brand AmazonHelp, (3) severe class imbalance and 44% `other_or_unclear`, (4) keyword taxonomy ≠ semantic truth, (5) only 200 golden examples, (6) proxy retrieval relevance without human relevance labels, (7) LLM judge ≠ customer satisfaction, (8) conservative escalation may inflate perceived safety while missing nuanced auto-handle cases, (9) direct-reply pairing misses multi-turn context, (10) temporal shift in Twitter support behavior.

## 14. What Was NOT Built

Production ticketing integration, live Twitter ingestion, multi-brand routing, active learning loop, calibrated threshold tuning on verified labels, full human annotation (200) and judge agreement panel (50).

## 15. One More Week

Complete human golden annotation; calibrate escalation thresholds on development set; conversation-aware reranking; temporal holdout; expand failure analysis with reviewer notes; run LLM judge + human agreement; A/B compare fallback vs LLM replies on verified subset.
