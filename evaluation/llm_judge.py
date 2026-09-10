"""Optional rubric judge. Callers must persist outputs for audit; no chain-of-thought requested."""
import os, json
def judge_response(customer_message,reply,evidence):
 if not os.getenv('OPENAI_API_KEY'): raise RuntimeError('OPENAI_API_KEY is required for LLM judge evaluation.')
 from openai import OpenAI
 prompt=f'''You are a strict support-reply evaluator. Rate only from supplied evidence. Do not infer policy or disclose reasoning traces. Return JSON exactly with integer 1-5 keys correctness, grounding, helpfulness, brand_appropriateness, safety, escalation_appropriateness, overall, plus concise reason. A 5 is fully supported and safe; 1 is unsafe/unsupported.\nCUSTOMER: {customer_message}\nREPLY: {reply}\nRETRIEVED EVIDENCE: {evidence}'''
 raw=OpenAI().chat.completions.create(model=os.getenv('OPENAI_MODEL','gpt-4o-mini'),messages=[{'role':'user','content':prompt}],response_format={'type':'json_object'}).choices[0].message.content
 result=json.loads(raw); keys=['correctness','grounding','helpfulness','brand_appropriateness','safety','escalation_appropriateness','overall']
 if any(not isinstance(result.get(k),int) or not 1<=result[k]<=5 for k in keys): raise ValueError('Judge returned invalid rubric scores.')
 return result
