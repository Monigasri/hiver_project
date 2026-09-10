"""Optional LLM-as-judge over agent predictions. Requires OPENAI_API_KEY."""
import json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from evaluation.llm_judge import judge_response

pred_path=Path('evaluation/results/agent_predictions.csv')
if not pred_path.exists(): raise SystemExit('Run scripts/run_evaluation.py first to create agent_predictions.csv.')
rows=[]; out=Path('evaluation/results')
for row in pd.read_csv(pred_path).itertuples(index=False):
 try:
  score=judge_response(row.customer_message,row.reply,row.retrieved_evidence)
  rows.append({'id':row.id,'conversation_id':row.conversation_id,**score})
 except RuntimeError as exc:
  raise SystemExit(str(exc))
 except Exception as exc:
  rows.append({'id':row.id,'conversation_id':row.conversation_id,'error':str(exc)})
pd.DataFrame(rows).to_csv(out/'llm_judge_scores.csv',index=False)
(out/'llm_judge_summary.json').write_text(json.dumps({'n':len(rows),'note':'Scores require OPENAI_API_KEY; not fabricated.'},indent=2),encoding='utf-8')
print(f'Saved {len(rows)} judge rows to evaluation/results/llm_judge_scores.csv')
