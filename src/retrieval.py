from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .config import SETTINGS

_MODEL_CACHE = {}
def _get_encoder(model_name: str = "all-MiniLM-L6-v2"):
 if model_name not in _MODEL_CACHE:
  from sentence_transformers import SentenceTransformer
  _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
 return _MODEL_CACHE[model_name]

class CaseRetriever:
 def __init__(self,cases:pd.DataFrame,index_dir:Path|None=None):
  self.cases=cases.reset_index(drop=True)
  self.backend="empty"; self.matrix=None; self.encoder=None; self.index=None
  self.index_dir=index_dir or SETTINGS.processed_dir
  texts=self.cases.customer_message_normalized.fillna(self.cases.customer_message).astype(str).tolist() if len(cases) else []
  if not texts: return
  meta=self._meta_payload(texts)
  if self._load_cached(meta):
   return
  try:
   import faiss
   self.encoder=_get_encoder("all-MiniLM-L6-v2")
   embeddings=self.encoder.encode(texts,normalize_embeddings=True,show_progress_bar=False).astype("float32")
   self.index=faiss.IndexFlatIP(embeddings.shape[1]); self.index.add(embeddings); self.backend="sentence-transformers+faiss"
   self._save_cached(meta,embeddings)
  except Exception:
   self.vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words="english")
   self.matrix=self.vectorizer.fit_transform(texts)
   self.backend="tfidf-fallback"

 def _meta_payload(self,texts:list[str])->dict:
  digest=hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
  return {"backend":"sentence-transformers+faiss","case_count":len(texts),"text_digest":digest,"model":"all-MiniLM-L6-v2"}

 def _load_cached(self,meta:dict)->bool:
  index_path=self.index_dir/SETTINGS.faiss_index_path.name
  meta_path=self.index_dir/SETTINGS.faiss_meta_path.name
  if not index_path.exists() or not meta_path.exists(): return False
  try:
   saved=json.loads(meta_path.read_text(encoding="utf-8"))
   if saved.get("text_digest")!=meta["text_digest"] or saved.get("case_count")!=meta["case_count"]: return False
   import faiss
   self.encoder=_get_encoder(saved.get("model","all-MiniLM-L6-v2"))
   self.index=faiss.read_index(str(index_path)); self.backend="sentence-transformers+faiss"
   return True
  except Exception:
   return False

 def _save_cached(self,meta:dict,embeddings:np.ndarray)->None:
  try:
   if len(self.cases)<1000: return
   import faiss
   self.index_dir.mkdir(parents=True,exist_ok=True)
   faiss.write_index(self.index,str(self.index_dir/SETTINGS.faiss_index_path.name))
   (self.index_dir/SETTINGS.faiss_meta_path.name).write_text(json.dumps(meta,indent=2),encoding="utf-8")
  except Exception:
   pass

 def search(self,text:str,k:int=3,exclude_ids:set[str]|None=None)->list[dict]:
  if not len(self.cases): return []
  exclude_ids=exclude_ids or set()
  fetch=min(k+len(exclude_ids),len(self.cases))
  if self.backend=="sentence-transformers+faiss":
   scores,idx=self.index.search(self.encoder.encode([text],normalize_embeddings=True).astype("float32"),fetch)
   pairs=((int(i),float(s)) for i,s in zip(idx[0],scores[0]) if int(i)>=0)
  else:
   if self.matrix is None: return []
   scores=cosine_similarity(self.vectorizer.transform([text]),self.matrix)[0]; idx=np.argsort(scores)[::-1][:fetch]
   pairs=((int(i),float(scores[i])) for i in idx)
  out=[]
  for i,score in pairs:
   cid=str(self.cases.iloc[i].conversation_id)
   if cid in exclude_ids: continue
   out.append({"conversation_id":cid,"similarity":round(score,4),"customer_message":self.cases.iloc[i].customer_message,"agent_response":self.cases.iloc[i].agent_response})
   if len(out)>=k: break
  return out
