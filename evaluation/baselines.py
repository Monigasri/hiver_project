from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
def majority_classifier(): return DummyClassifier(strategy='most_frequent')
def tfidf_logreg(): return make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=1),LogisticRegression(max_iter=1000,class_weight='balanced'))
