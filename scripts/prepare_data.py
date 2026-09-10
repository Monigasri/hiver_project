import argparse, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import SETTINGS
from src.data_pipeline import find_raw_file,prepare_chunked

p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=500_000,help='Deterministic first-N-row sample; use --limit 0 for the full file.');p.add_argument('--brand');p.add_argument('--golden-size',type=int,default=200);p.add_argument('--chunk-size',type=int,default=100_000);a=p.parse_args()
raw=find_raw_file(SETTINGS.raw_dir); limit=None if a.limit==0 else a.limit
result=prepare_chunked(raw,SETTINGS.processed_dir,SETTINGS.golden_dir,limit,a.brand or SETTINGS.brand,a.golden_size,SETTINGS.random_seed,a.chunk_size)
train=pd.read_csv(SETTINGS.processed_dir/'retrieval_cases.csv'); dev=train.sample(n=min(1000,len(train)),random_state=SETTINGS.random_seed)
pd.DataFrame({'conversation_id':dev.conversation_id,'customer_message':dev.customer_message,'train_intent':''}).to_csv(SETTINGS.golden_dir/'development_labels.csv',index=False)
print(result)
