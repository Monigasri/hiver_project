import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr
RUBRICS=['correctness','grounding','helpfulness','brand_appropriateness','safety','escalation_appropriateness','overall']
def agreement(csv_path, judge_csv=None):
 human=pd.read_csv(csv_path); judge=pd.read_csv(judge_csv) if judge_csv else human
 merged=human.merge(judge,on='id',suffixes=('_human','_judge')); out={'n':len(merged),'rubrics':{}}
 if len(merged)<2: raise ValueError('At least two paired human and judge ratings are required.')
 for key in RUBRICS:
  a,b=merged[f'{key}_human'],merged[f'{key}_judge']; out['rubrics'][key]={'exact_agreement':float((a==b).mean()),'weighted_kappa':float(cohen_kappa_score(a,b,weights='quadratic')),'spearman':float(spearmanr(a,b).statistic)}
 return out
