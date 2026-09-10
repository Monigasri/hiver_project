import streamlit as st
import pandas as pd
from src.config import SETTINGS
from src.taxonomy import load_taxonomy
from src.agent import SupportAgent
st.title('Historical-evidence support agent')
try:
 text=st.text_area('Customer message')
 if st.button('Run') and text:
  agent=SupportAgent(load_taxonomy(SETTINGS.taxonomy_path),pd.read_csv(SETTINGS.processed_dir/'retrieval_cases.csv'))
  result=agent.run(text);st.json(result);st.caption('Replies are only auto-handled when evidence and confidence meet configured thresholds.')
except FileNotFoundError: st.info('Run prepare_data.py and discover_intents.py first.')
