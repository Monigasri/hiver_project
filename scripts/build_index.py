import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import SETTINGS
import pandas as pd
from src.retrieval import CaseRetriever
cases=pd.read_csv(SETTINGS.processed_dir/'retrieval_cases.csv')
r=CaseRetriever(cases)
print(f'Retriever built with {r.backend} for {len(cases)} leakage-safe historical cases.')
