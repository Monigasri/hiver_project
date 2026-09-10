import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
try:
 import app as backend_app
 from app import AgentRequest
except ImportError:
 pytest.skip('Install backend requirements to run API tests.',allow_module_level=True)
def test_health(): assert backend_app.health()['status']=='ok'
def test_rejects_empty_message():
 with pytest.raises(Exception): AgentRequest(message='')
def test_agent_explains_missing_dataset():
 if not backend_app.artifacts_ready():
  with pytest.raises(Exception) as error: backend_app.agent(AgentRequest(message='I cannot access my account'))
  assert getattr(error.value,'status_code',None)==503

def test_agent_normal_support_request():
 if backend_app.artifacts_ready():
  res = backend_app.agent(AgentRequest(message="Where is my package? It was supposed to arrive today."))
  assert res['intent'] in ["delivery_problem", "order_status", "customer_service"]
  assert isinstance(res['reply'], str)
  assert isinstance(res['evidence'], list)

def test_agent_account_issue():
 if backend_app.artifacts_ready():
  res = backend_app.agent(AgentRequest(message="I cannot access my Amazon account"))
  assert res['decision'] == "ESCALATE"
  assert "account" in res['escalation_reason'].lower() or "confidence" in res['escalation_reason'].lower()

def test_agent_security_issue():
 if backend_app.artifacts_ready():
  res = backend_app.agent(AgentRequest(message="Someone has accessed my Amazon account and I think it has been compromised"))
  assert res['decision'] == "ESCALATE"
  assert "security" in res['escalation_reason'].lower() or "sensitive" in res['escalation_reason'].lower() or "threshold" in res['escalation_reason'].lower() or "account" in res['escalation_reason'].lower()

def test_agent_ambiguous_issue():
 if backend_app.artifacts_ready():
  res = backend_app.agent(AgentRequest(message="help"))
  assert res['decision'] == "ESCALATE"
  assert "ambiguous" in res['escalation_reason'].lower() or "threshold" in res['escalation_reason'].lower() or "confidence" in res['escalation_reason'].lower()

