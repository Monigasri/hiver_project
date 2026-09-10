import streamlit as st
import pandas as pd
from src.config import SETTINGS
from src.taxonomy import load_taxonomy
path=SETTINGS.golden_dir/'golden_set.csv'
st.title('Golden-set annotation')
st.caption('Fill gold_intent and gold_escalation. suggested_intent is a machine hint only.')
if not path.exists(): st.info('Run prepare_data.py then create_golden_template.py first.'); st.stop()
df=pd.read_csv(path); intents=[x['name'] for x in load_taxonomy(SETTINGS.taxonomy_path)['intents']]
verified=int((~df.gold_intent.fillna('').eq('')) & (~df.gold_escalation.fillna('').eq(''))).sum()
st.write(f'Human-verified: {verified}/{len(df)}')
pending=df[df.gold_intent.fillna('').eq('') | df.gold_escalation.fillna('').eq('')]
if pending.empty: st.success('All golden examples are labelled.'); st.stop()
r=pending.iloc[0]
st.caption(f"Example {r.id} · conversation {r.conversation_id}")
st.write(r.customer_message)
if 'suggested_intent' in df.columns and str(r.suggested_intent).strip(): st.info(f'Machine suggestion (unverified): {r.suggested_intent}')
default=intent=r.suggested_intent if 'suggested_intent' in df.columns and r.suggested_intent in intents else intents[0]
intent=st.selectbox('Intent',intents,index=intents.index(default) if default in intents else 0)
escalate=st.radio('Escalate?', ['AUTO_HANDLE','ESCALATE'])
notes=st.text_input('Notes (optional)',value=str(r.notes) if 'notes' in df.columns else '')
if st.button('Save and show next'):
 df.loc[df.id==r.id,['gold_intent','gold_escalation','notes','annotation_status']]=[intent,escalate,notes,'human_verified']
 df.to_csv(path,index=False); st.rerun()
