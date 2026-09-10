from __future__ import annotations
from .classifier import TaxonomyClassifier
from .retrieval import CaseRetriever
from .escalation import decide
from .response_generator import grounded_reply
from .config import SETTINGS

class SupportAgent:
 def __init__(self,taxonomy,cases,min_intent:float|None=None,min_similarity:float|None=None):
  self.classifier=TaxonomyClassifier(taxonomy)
  self.retriever=CaseRetriever(cases)
  self.min_intent=min_intent if min_intent is not None else SETTINGS.min_intent_confidence
  self.min_similarity=min_similarity if min_similarity is not None else SETTINGS.min_retrieval_similarity
  self.retrieval_backend=self.retriever.backend

 def run(self,text:str,exclude_conversation_ids:set[str]|None=None)->dict:
  prediction=self.classifier.predict(text)
  evidence=self.retriever.search(text,3,exclude_conversation_ids)
  route=decide(text,prediction["intent"],prediction["confidence"],evidence,self.min_intent,self.min_similarity)
  if route["decision"]=="AUTO_HANDLE":
   reply=grounded_reply(text,prediction["intent"],evidence,SETTINGS.openai_api_key,SETTINGS.openai_model)
  else:
   reply="Thanks for reaching out. Our support team will review your request and help further."
  return {"intent":prediction["intent"],"intent_confidence":prediction["confidence"],"decision":route["decision"],"escalation_reason":route["reason"],"reply":reply,"evidence":evidence}
