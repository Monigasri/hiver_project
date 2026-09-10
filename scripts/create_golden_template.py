import pandas as pd,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import SETTINGS
from src.taxonomy import load_taxonomy
from src.classifier import TaxonomyClassifier
p=SETTINGS.golden_dir/'golden_template.csv'; df=pd.read_csv(p).reset_index(drop=True)
taxonomy=load_taxonomy(SETTINGS.taxonomy_path); clf=TaxonomyClassifier(taxonomy)
suggested=[clf.predict(str(x))['intent'] for x in df.customer_message]
out=pd.DataFrame({'id':range(1,len(df)+1),'conversation_id':df.conversation_id,'customer_message':df.customer_message,'suggested_intent':suggested,'gold_intent':'','gold_escalation':'','annotation_status':'pending','notes':''})
out.to_csv(SETTINGS.golden_dir/'golden_set.csv',index=False)
print(f'Created golden_set.csv with {len(out)} rows and machine suggestions. Human must verify gold_intent and gold_escalation.')
