import pandas as pd
from src.data_pipeline import normalize_text, load_and_clean, build_cases, split_cases
from src.taxonomy import DEFAULT_TAXONOMY,save_taxonomy,load_taxonomy
from src.classifier import TaxonomyClassifier
from src.retrieval import CaseRetriever
from src.escalation import decide
from src.agent import SupportAgent
from evaluation.metrics import intent_metrics

def cases(): return pd.DataFrame({'conversation_id':['1','2'],'customer_message':['my payment failed','cannot login'],'customer_message_normalized':['my payment failed','cannot login'],'agent_response':['Please contact us','Reset your password']})
def test_normalize(): assert normalize_text(' a\n b ')=='a b'
def test_clean_and_pairs(tmp_path):
 p=tmp_path/'x.csv';pd.DataFrame({'tweet_id':[1,2,2],'author_id':['u','brand','brand'],'inbound':[True,False,False],'created_at':['2020-01-01']*3,'text':['help','reply','duplicate'],'in_response_to_tweet_id':[None,1,1]}).to_csv(p,index=False)
 df=load_and_clean(p); assert len(df)==2; assert len(build_cases(df,'brand'))==1
def test_split_has_no_leakage():
 train,golden=split_cases(pd.concat([cases()]*10,ignore_index=True).assign(conversation_id=lambda x:range(len(x))),2);assert not set(train.conversation_id)&set(golden.conversation_id)
def test_taxonomy_round_trip(tmp_path):
 p=tmp_path/'t.yaml';save_taxonomy(p);assert load_taxonomy(p)['intents'][0]['name']=='account_access'
def test_classifier_contract():
 result=TaxonomyClassifier(DEFAULT_TAXONOMY).predict('my payment was charged twice');assert result['intent'] in [x['name'] for x in DEFAULT_TAXONOMY['intents']];assert 0<=result['confidence']<=1
def test_retrieval(): assert CaseRetriever(cases()).search('payment problem',1)[0]['conversation_id']=='1'
def test_escalation(): assert decide('my account was hacked','account_or_security',.9,[{'similarity':.9}])['decision']=='ESCALATE'
def test_agent_shape():
 r=SupportAgent(DEFAULT_TAXONOMY,cases()).run('my payment failed');assert {'intent','intent_confidence','decision','escalation_reason','reply','evidence'}<=set(r)
def test_metrics(): assert intent_metrics(['a','b'],['a','b'])['accuracy']==1
