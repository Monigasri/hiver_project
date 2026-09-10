# Engineering Decision Log

| # | Decision | Reason | Alternative considered | Tradeoff |
|---|----------|--------|------------------------|----------|
| 1 | Select **AmazonHelp** via outbound reply volume score | High direct-reply volume yields many paired customer→agent resolutions for retrieval | Pick largest tweet volume brand | May not generalize to smaller brands |
| 2 | **Deterministic 500k row-prefix sample** | Memory-safe, reproducible on a 516MB CSV without loading 3M rows | Full dataset or random sampling | Sample may miss rare intents and temporal drift |
| 3 | **Conversation-ID split** for golden holdout | Prevents the same thread appearing in retrieval and evaluation | Random tweet split | Smaller but leakage-safe evaluation set |
| 4 | **Keyword-priority taxonomy** from AmazonHelp corpus | Transparent, auditable labels tied to real message text | Banking77 transfer or pure clustering | `other_or_unclear` remains large (~44%) |
| 5 | **TF-IDF taxonomy classifier** for live routing | Fast, deterministic, no training dependency at inference | Fine-tuned transformer | Lower semantic accuracy than neural classifiers |
| 6 | **Sentence-Transformers + FAISS** retrieval | Strong semantic recall over 39k historical cases | MongoDB/Elasticsearch only | First build encodes full corpus; cached index mitigates |
| 7 | **Persist FAISS index** to disk | Avoid re-encoding 39k tweets on every API request | Rebuild embeddings each startup | Must invalidate cache when corpus changes |
| 8 | **Explicit deterministic escalation rules** | Safety-critical routing must not rely on LLM alone | LLM-only escalation | More conservative escalations, fewer auto-handles |
| 9 | **Account/security always escalates** when detected | Reduces risk of automated replies on sensitive access issues | Auto-handle with high similarity | Higher escalation rate for account intents |
| 10 | **Historical reply fallback** without API key | System runs offline; tests and demo work without secrets | Require OpenAI for all replies | Replies mirror nearest neighbor, less fluent |
| 11 | **Optional LLM drafting** with evidence-only prompt | Better fluency while constraining hallucinations | Unconstrained generation | Requires API key; still not a policy authority |
| 12 | **Human labels required** for golden metrics | Assignment forbids fabricated evaluation | Use model labels as ground truth | Blocks headline scores until annotation completes |
| 13 | **LLM judge separated from agent** | Judge rubric is evaluation-only, not production routing | Same model decides and scores | Extra API cost; agreement needs human panel |
| 14 | **No MongoDB / external DB** | Simplifies reproducible local demo | Document store for cases | CSV + FAISS sufficient for take-home scale |
| 15 | **React + FastAPI split** | Clear API contract for assignment endpoints | Streamlit-only UI | Requires CORS/port configuration |
