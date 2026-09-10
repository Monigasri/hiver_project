"""Retrieval evaluation using intent-coherent relevance on held-out golden cases."""
from __future__ import annotations
import pandas as pd
from src.retrieval import CaseRetriever
from src.intent_labels import assign_intents

def _precision_at_k(relevant:set[str],retrieved:list[str],k:int)->float:
 top=retrieved[:k]
 return float(sum(x in relevant for x in top)/k) if top else 0.0

def _recall_at_k(relevant:set[str],retrieved:list[str],k:int)->float:
 if not relevant: return 0.0
 return float(len(relevant & set(retrieved[:k]))/len(relevant))

def _mrr(relevant:set[str],retrieved:list[str])->float:
 for rank,item in enumerate(retrieved,1):
  if item in relevant: return 1.0/rank
 return 0.0

def evaluate_retrieval(gold:pd.DataFrame,corpus:pd.DataFrame,k:int=3)->dict:
 """Relevance: retrieved conversation is relevant if its deterministic taxonomy intent
 matches the golden example's intent label (human or proxy from same assignment rule)."""
 corpus=corpus.copy(); corpus['proxy_intent']=assign_intents(corpus.customer_message)
 retriever=CaseRetriever(corpus)
 rows=[]; p=[]; r=[]; mrr=[]
 for row in gold.itertuples(index=False):
  intent=str(getattr(row,'gold_intent',None) or getattr(row,'proxy_intent',None) or '').strip()
  if not intent: continue
  relevant=set(corpus.loc[corpus.proxy_intent==intent,'conversation_id'].astype(str))
  retrieved=[e['conversation_id'] for e in retriever.search(row.customer_message,k=k,exclude_ids={str(row.conversation_id)})]
  p.append(_precision_at_k(relevant,retrieved,k)); r.append(_recall_at_k(relevant,retrieved,k)); mrr.append(_mrr(relevant,retrieved))
  rows.append({'id':getattr(row,'id',None),'conversation_id':row.conversation_id,'intent':intent,'retrieved':retrieved,'precision_at_k':p[-1],'recall_at_k':r[-1],'mrr':mrr[-1]})
 if not rows: raise ValueError('No labelled golden rows available for retrieval evaluation.')
 return {'k':k,'definition':'Relevant if retrieved conversation shares the same deterministic taxonomy intent as the query golden label.','n_evaluated':len(rows),'precision_at_k':float(sum(p)/len(p)),'recall_at_k':float(sum(r)/len(r)),'mrr':float(sum(mrr)/len(mrr)),'rows':rows}
