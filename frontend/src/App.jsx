import {useEffect,useState} from 'react';
import {analyze,info} from './api';

const score=x=>`${Math.round(x*100)}%`;

export default function App(){
 const [message,setMessage]=useState("I can't access my Amazon account");
 const [result,setResult]=useState(null);
 const [error,setError]=useState('');
 const [loading,setLoading]=useState(false);
 const [meta,setMeta]=useState(null);
 useEffect(()=>{info().then(setMeta).catch(()=>setMeta(null))},[]);
 async function submit(){
  if(!message.trim())return;
  setLoading(true);setError('');
  try{setResult(await analyze(message))}
  catch(e){setError(e.message||'Failed to reach backend. Ensure API is running on port 8001.');setResult(null)}
  finally{setLoading(false)}
 }
 return <main>
  <header>
   <h1>Hiver AI Support Agent</h1>
   <p>AI-assisted customer support grounded in historical {meta?.selected_brand||'brand'} conversations</p>
  </header>
  <section className="panel">
   <label htmlFor="message">Customer message</label>
   <textarea id="message" value={message} onChange={e=>setMessage(e.target.value)} placeholder="Enter a customer message..." rows={4}/>
   <button onClick={submit} disabled={loading||!message.trim()}>{loading?'Analyzing…':'Analyze Message'}</button>
   {error&&<div className="error">{error}</div>}
  </section>
  {result&&<section className="results">
   <div className="grid">
    <Card title="Intent"><strong>{result.intent.replaceAll('_',' ')}</strong><span>{score(result.intent_confidence)} confidence</span></Card>
    <Card title="Decision"><strong className={result.decision==='ESCALATE'?'escalate':'auto'}>{result.decision.replace('_','-')}</strong><span>{result.escalation_reason}</span></Card>
   </div>
   <Card title="Draft reply"><p className="reply">{result.reply}</p><button className="secondary" onClick={()=>navigator.clipboard.writeText(result.reply)}>Copy Reply</button></Card>
   <Card title="Historical Evidence">
    {result.evidence.length?result.evidence.map((e,i)=><article className="evidence" key={i}><b>{score(e.similarity)} similarity · conversation {e.conversation_id}</b><p><em>Customer:</em> {e.customer_message}</p><p><em>Support:</em> {e.agent_response}</p></article>):<p>No similar historical cases retrieved.</p>}
    <small>Replies are grounded in retrieved historical AmazonHelp resolutions, not invented policy.</small>
   </Card>
  </section>}
  <footer>Backend: {meta?.retrieval_backend||'unknown'} · Retrieval corpus: {meta?.retrieval_count??'—'} cases · LLM: {meta?.llm_enabled?'enabled':'fallback mode'}{meta&&!meta.artifacts_ready?' · Dataset setup required.':''}</footer>
 </main>
}
function Card({title,children}){return <section className="card"><h2>{title}</h2>{children}</section>}
