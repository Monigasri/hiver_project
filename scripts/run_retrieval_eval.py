"""Run retrieval evaluation when golden labels include intents (human or proxy)."""
import json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.config import SETTINGS
from src.intent_labels import assign_intents
from evaluation.retrieval_eval import evaluate_retrieval

out=Path('evaluation/results'); out.mkdir(parents=True,exist_ok=True)
gold=pd.read_csv(SETTINGS.golden_dir/'golden_set.csv'); corpus=pd.read_csv(SETTINGS.processed_dir/'retrieval_cases.csv')
if gold.gold_intent.fillna('').eq('').all():
 gold=gold.copy(); gold['proxy_intent']=assign_intents(gold.customer_message)
 print('HUMAN LABELING RECOMMENDED: using proxy intent labels from deterministic taxonomy for retrieval eval only.')
result=evaluate_retrieval(gold,corpus,k=3)
(out/'retrieval_metrics.json').write_text(json.dumps({k:result[k] for k in ['k','definition','n_evaluated','precision_at_k','recall_at_k','mrr']},indent=2),encoding='utf-8')
pd.DataFrame(result['rows']).to_csv(out/'retrieval_eval_rows.csv',index=False)
print(json.dumps({k:result[k] for k in ['k','n_evaluated','precision_at_k','recall_at_k','mrr']},indent=2))
