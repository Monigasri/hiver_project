"""Leakage-safe, label-gated evaluation; no result is emitted without human labels."""
import sys
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import pandas as pd
from src.config import SETTINGS
from src.taxonomy import load_taxonomy
from src.agent import SupportAgent
from evaluation.baselines import majority_classifier, tfidf_logreg
from evaluation.metrics import intent_metrics, escalation_metrics

out=Path('evaluation/results'); out.mkdir(parents=True,exist_ok=True)
gold=pd.read_csv(SETTINGS.golden_dir/'golden_set.csv'); corpus=pd.read_csv(SETTINGS.processed_dir/'retrieval_cases.csv'); dev=pd.read_csv(SETTINGS.golden_dir/'development_labels.csv')
if any(c not in gold or gold[c].fillna('').eq('').any() for c in ['gold_intent','gold_escalation']): raise SystemExit('golden_set.csv needs complete human-confirmed labels.')
if 'train_intent' not in dev or dev.train_intent.fillna('').eq('').any(): raise SystemExit('development_labels.csv needs complete human-confirmed training labels.')
if set(gold.conversation_id.astype(str)) & set(corpus.conversation_id.astype(str)): raise SystemExit('LEAKAGE: golden conversation IDs exist in retrieval corpus.')
train=corpus.merge(dev[['conversation_id','train_intent']],on='conversation_id',how='inner')
if train.empty: raise SystemExit('No labelled development conversations joined.')
names={x['name'] for x in load_taxonomy(SETTINGS.taxonomy_path)['intents']}
if not set(gold.gold_intent).issubset(names) or not set(train.train_intent).issubset(names): raise SystemExit('A human label is outside taxonomy.yaml.')
preds={}
for name,model in [('majority',majority_classifier()),('tfidf_logreg',tfidf_logreg())]: model.fit(train.customer_message,train.train_intent); preds[name]=list(model.predict(gold.customer_message))
agent=SupportAgent(load_taxonomy(SETTINGS.taxonomy_path),train.drop(columns='train_intent')); rows=[]
for row in gold.itertuples(index=False):
 r=agent.run(row.customer_message); preds.setdefault('ai_agent',[]).append(r['intent'])
 rows.append({'id':row.id,'conversation_id':row.conversation_id,'customer_message':row.customer_message,'gold_intent':row.gold_intent,'predicted_intent':r['intent'],'intent_confidence':r['intent_confidence'],'gold_escalation':row.gold_escalation,'predicted_escalation':r['decision'],'escalation_reason':r['escalation_reason'],'reply':r['reply'],'retrieved_evidence':json.dumps(r['evidence'])})
pd.DataFrame(rows).to_csv(out/'agent_predictions.csv',index=False)
metrics={name:intent_metrics(list(gold.gold_intent),p) for name,p in preds.items()}; escalation=escalation_metrics(list(gold.gold_escalation),[r['predicted_escalation'] for r in rows])
(out/'metrics.json').write_text(json.dumps(metrics,indent=2)); (out/'escalation_metrics.json').write_text(json.dumps(escalation,indent=2))
pd.DataFrame([{'system':k,'accuracy':v['accuracy'],'macro_precision':v['macro_precision'],'macro_recall':v['macro_recall'],'macro_f1':v['macro_f1']} for k,v in metrics.items()]).to_csv(out/'comparison.csv',index=False)
wrong=pd.DataFrame(rows).query('gold_intent != predicted_intent').copy(); wrong['pattern']=wrong.gold_intent+' -> '+wrong.predicted_intent; failures=[]
for pattern in wrong.pattern.value_counts().head(5).index:
 r=wrong[wrong.pattern==pattern].iloc[0]; failures.append({'failure_mode':'intent_confusion','pattern':pattern,'real_example_id':int(r.id),'customer_message':r.customer_message,'expected':r.gold_intent,'actual':r.predicted_intent,'why_it_failed':'Observed classification mismatch; root cause requires reviewer inspection.','proposed_improvement':'Review taxonomy boundary and add labelled development examples.'})
(out/'failure_analysis.json').write_text(json.dumps(failures,indent=2)); print(json.dumps({'intent_metrics':metrics,'escalation':escalation},indent=2))
