import argparse,json,pandas as pd,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import SETTINGS
from src.taxonomy import load_taxonomy
from src.agent import SupportAgent
p=argparse.ArgumentParser();p.add_argument('--text',required=True);a=p.parse_args()
taxonomy=load_taxonomy(SETTINGS.taxonomy_path); cases=pd.read_csv(SETTINGS.processed_dir/'retrieval_cases.csv')
print(json.dumps(SupportAgent(taxonomy,cases).run(a.text),indent=2,default=str))
