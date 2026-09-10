from __future__ import annotations
import re

HIGH_RISK=r"\b(fraud|scam|hacked|hack|stolen|identity|police|lawyer|legal|lawsuit|sue|card number|social security|ssn|unauthorized|unauthorised|compromised|breached)\b"
SENSITIVE_PII=r"\b(passport|driver.?s license|bank account|routing number|cvv|pin code)\b"
UNSUPPORTED_ACTION=r"\b(delete my account|close my account|cancel subscription|issue refund|send refund|wire transfer|chargeback)\b"

def decide(text:str,intent:str,intent_confidence:float,evidence:list[dict],min_intent:float=.45,min_similarity:float=.35)->dict:
 lowered=text.lower().strip()
 if re.search(HIGH_RISK,lowered,re.I):
  return {"decision":"ESCALATE","reason":"Sensitive security, fraud, or legal language requires human handling."}
 if re.search(SENSITIVE_PII,lowered,re.I):
  return {"decision":"ESCALATE","reason":"Message appears to involve sensitive personal information."}
 if intent in {"account_or_security"} and intent_confidence>=min_intent:
  return {"decision":"ESCALATE","reason":"Account or security issues are routed to a human specialist."}
 if intent_confidence<min_intent:
  return {"decision":"ESCALATE","reason":"Intent confidence is below the configured threshold."}
 if not evidence or evidence[0]["similarity"]<min_similarity:
  return {"decision":"ESCALATE","reason":"Historical evidence is insufficiently similar to ground a response."}
 if len(lowered.split())<4:
  return {"decision":"ESCALATE","reason":"Message is too ambiguous to answer safely without clarification."}
 if re.search(UNSUPPORTED_ACTION,lowered,re.I) and (not evidence or evidence[0]["similarity"]<0.55):
  return {"decision":"ESCALATE","reason":"Requested action needs human approval and stronger historical support."}
 return {"decision":"AUTO_HANDLE","reason":"Intent and historical evidence meet configured confidence thresholds."}
