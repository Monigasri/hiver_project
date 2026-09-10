"""FastAPI service wrapping the evidence-first support agent."""
from __future__ import annotations
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.config import SETTINGS
from src.taxonomy import load_taxonomy
from src.agent import SupportAgent

app=FastAPI(title="Hiver AI Support Agent",version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

class AgentRequest(BaseModel): message: str = Field(min_length=1,max_length=4000)
class Evidence(BaseModel): conversation_id:str; similarity:float; customer_message:str; agent_response:str
class AgentResponse(BaseModel): intent:str; intent_confidence:float; decision:Literal['AUTO_HANDLE','ESCALATE']; escalation_reason:str; reply:str; evidence:list[Evidence]
class GoldenLabel(BaseModel): id:int; gold_intent:str; gold_escalation:Literal['AUTO_HANDLE','ESCALATE']; notes:str=''

def artifacts_ready() -> bool:
 return SETTINGS.taxonomy_path.exists() and (SETTINGS.processed_dir/'retrieval_cases.csv').exists()

@lru_cache(maxsize=1)
def get_agent() -> SupportAgent:
 if not artifacts_ready():
  raise HTTPException(status_code=503,detail="Dataset artifacts are not ready. Put the Kaggle CSV in data/raw and run python scripts/prepare_data.py, then python scripts/discover_intents.py.")
 return SupportAgent(load_taxonomy(SETTINGS.taxonomy_path),pd.read_csv(SETTINGS.processed_dir/'retrieval_cases.csv'))

@lru_cache(maxsize=1)
def get_selection() -> dict|None:
 if not SETTINGS.selection_path.exists(): return None
 return json.loads(SETTINGS.selection_path.read_text(encoding="utf-8"))

@lru_cache(maxsize=1)
def get_retrieval_count() -> int:
 path=SETTINGS.processed_dir/'retrieval_cases.csv'
 return len(pd.read_csv(path)) if path.exists() else 0

@app.get('/health')
def health(): return {'status':'ok','artifacts_ready':artifacts_ready()}

@app.get('/api/info')
def info():
 selection=get_selection()
 return {
  'artifacts_ready':artifacts_ready(),
  'selected_brand':selection.get('brand') if selection else None,
  'brand_statistics':selection.get('statistics') if selection else None,
  'retrieval_count':get_retrieval_count(),
  'retrieval_backend':get_agent().retrieval_backend if artifacts_ready() else None,
  'llm_enabled':bool(SETTINGS.openai_api_key),
  'system':'taxonomy classifier + sentence-transformers/FAISS retrieval (TF-IDF fallback) + deterministic escalation',
 }

@app.post('/api/agent',response_model=AgentResponse)
def agent(payload:AgentRequest): return get_agent().run(payload.message.strip())

@app.post('/api/evaluate')
def evaluate():
 results=ROOT/'evaluation'/'results'/'metrics.json'
 if not results.exists():
  raise HTTPException(409,'Evaluation is pending complete human labels and scripts/run_evaluation.py.')
 payload={
  'intent_metrics':json.loads(results.read_text(encoding='utf-8')),
  'escalation_metrics':json.loads((ROOT/'evaluation'/'results'/'escalation_metrics.json').read_text(encoding='utf-8')) if (ROOT/'evaluation'/'results'/'escalation_metrics.json').exists() else None,
  'retrieval_metrics':json.loads((ROOT/'evaluation'/'results'/'retrieval_metrics.json').read_text(encoding='utf-8')) if (ROOT/'evaluation'/'results'/'retrieval_metrics.json').exists() else None,
  'status':'complete',
 }
 return payload

@app.post('/api/golden')
def save_golden(label:GoldenLabel):
 path=SETTINGS.golden_dir/'golden_set.csv'
 if not path.exists(): raise HTTPException(409,'Golden annotation file has not been created yet.')
 df=pd.read_csv(path)
 if label.id not in set(df.id): raise HTTPException(404,'Golden example not found.')
 df.loc[df.id==label.id,['gold_intent','gold_escalation','notes']]=[label.gold_intent,label.gold_escalation,label.notes]
 df.to_csv(path,index=False)
 return {'saved':True,'id':label.id}
