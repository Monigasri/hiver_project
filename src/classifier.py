from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TaxonomyClassifier:
 def __init__(self,taxonomy:dict):
  self.names=[i["name"] for i in taxonomy["intents"]]
  self.docs=[" ".join([i["name"].replace("_"," "),i["definition"],*i.get("inclusion",[])]) for i in taxonomy["intents"]]
  self.vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words="english").fit(self.docs)
  self.matrix=self.vectorizer.transform(self.docs)
 def predict(self,text:str)->dict:
  scores=cosine_similarity(self.vectorizer.transform([text]),self.matrix)[0]
  ix=int(np.argmax(scores)); confidence=float(max(scores))
  return {"intent":self.names[ix],"confidence":round(confidence,4)}
