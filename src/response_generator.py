from __future__ import annotations
import json
import os

FALLBACK_NOTE=" [Draft grounded in the closest historical AmazonHelp resolution; no LLM API key configured.]"

LLM_SYSTEM="""You draft customer-support replies for AmazonHelp on Twitter.
Use ONLY the supplied historical evidence. Do not invent policies, refunds, credits, account actions, URLs, or timelines.
Do not claim any action was performed. Do not expose private information.
Be concise and professional. If evidence is insufficient, say the team will review further.
Never mention that you are an AI."""

def _template_reply(message:str,intent:str,evidence:list[dict])->str:
 if not evidence: return "Thanks for reaching out. Our support team will review this and help further."
 base=str(evidence[0]["agent_response"]).strip()
 if not os.getenv("OPENAI_API_KEY"): return base + FALLBACK_NOTE
 return base

def _llm_reply(message:str,intent:str,evidence:list[dict],api_key:str,model:str)->str:
 from openai import OpenAI
 evidence_block="\n\n".join(f"Example {i+1} (similarity {e['similarity']}):\nCustomer: {e['customer_message']}\nSupport: {e['agent_response']}" for i,e in enumerate(evidence))
 prompt=f"Customer message:\n{message}\n\nPredicted intent: {intent.replace('_',' ')}\n\nHistorical evidence:\n{evidence_block}\n\nDraft one concise support reply grounded only in the evidence."
 raw=OpenAI(api_key=api_key).chat.completions.create(model=model,messages=[{"role":"system","content":LLM_SYSTEM},{"role":"user","content":prompt}],temperature=0.2).choices[0].message.content
 return (raw or _template_reply(message,intent,evidence)).strip()

def grounded_reply(message:str,intent:str,evidence:list[dict],api_key:str|None=None,model:str|None=None)->str:
 if not evidence: return "Thanks for reaching out. Our support team will review this and help further."
 key=api_key or os.getenv("OPENAI_API_KEY")
 if key:
  try:
   return _llm_reply(message,intent,evidence,key,model or os.getenv("OPENAI_MODEL","gpt-4o-mini"))
  except Exception:
   pass
 return _template_reply(message,intent,evidence)
