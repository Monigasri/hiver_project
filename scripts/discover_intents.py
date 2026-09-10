"""Build a transparent, data-backed AmazonHelp taxonomy from real corpus patterns."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import yaml
from src.config import SETTINGS
from src.intent_labels import PRIORITY_CATEGORIES

DESCRIPTIONS={
 'delivery_problem':'Delivery attempts, delays, lost packages, or delivery instructions.',
 'order_status':'Questions about an existing order, order number, or order arrival.',
 'return_or_refund':'Returns, refunds, or reimbursement status.',
 'prime_membership':'Prime membership, benefits, cancellation, or subscription questions.',
 'account_or_security':'Account access, unauthorised use, password, or suspicious activity.',
 'payment_or_pricing':'Charges, payment methods, price, billing, or overcharge concerns.',
 'product_availability':'Stock, availability, product listing, or item-specific questions.',
 'technical_or_content':'App, website, device, digital content, or playback issues.',
 'customer_service':'Requests to contact support, complaints about support, or manager requests.',
 'other_or_unclear':'Messages without enough evidence for a narrower taxonomy intent.',
}
cases=pd.read_csv(SETTINGS.processed_dir/'retrieval_cases.csv')
if cases.empty: raise SystemExit('Run prepare_data.py first.')
texts=cases.customer_message.fillna('').astype(str).str.lower()
assigned=pd.Series(False,index=cases.index); intents=[]
for number,(name,keywords) in enumerate(PRIORITY_CATEGORIES,1):
 mask=texts.str.contains('|'.join(__import__('re').escape(k) for k in keywords),regex=True) if keywords else ~assigned
 mask=mask & ~assigned; assigned|=mask
 label=name.replace('_',' ').title()
 examples=cases.loc[mask,'customer_message'].head(3).astype(str).tolist()
 intents.append({'intent_id':f'amazonhelp_{number:02d}','name':name,'display_name':label,'description':DESCRIPTIONS[name],'definition':DESCRIPTIONS[name],'inclusion':keywords or ['no high-signal category match'],'exclusion':['Messages assigned to a more specific preceding taxonomy category.'] if keywords else [],'representative_examples':examples,'examples':examples,'number_of_examples':int(mask.sum())})
taxonomy={'brand':str(cases.brand.iloc[0]),'method':{'approach':'deterministic keyword-supported taxonomy review','corpus_size':len(cases),'priority_order':[x[0] for x in PRIORITY_CATEGORIES],'note':'Counts and examples are from the selected-brand retrieval corpus; human review is still required before golden annotation.'},'intents':intents}
SETTINGS.taxonomy_path.write_text(yaml.safe_dump(taxonomy,sort_keys=False,allow_unicode=True),encoding='utf-8')
for i in intents: print(f"{i['name']}: {i['number_of_examples']}")
