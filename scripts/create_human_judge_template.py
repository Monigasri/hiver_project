import pandas as pd
from pathlib import Path
p=Path('evaluation/results/agent_predictions.csv')
if not p.exists(): raise SystemExit('Run run_evaluation.py first.')
df=pd.read_csv(p).head(50)
out=df[['id','customer_message','reply','retrieved_evidence']].copy()
for c in ['correctness','grounding','helpfulness','brand_appropriateness','safety','escalation_appropriateness','overall']: out[c]=''
out.to_csv('evaluation/human_annotation.csv',index=False); print('Created 50-case human rating template.')
