const BASE=import.meta.env.VITE_API_URL || 'http://127.0.0.1:8001';
export async function analyze(message){const r=await fetch(`${BASE}/api/agent`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message})});if(!r.ok){const b=await r.json().catch(()=>({}));throw new Error(b.detail||'Analysis failed');}return r.json()}
export async function info(){const r=await fetch(`${BASE}/api/info`);return r.json()}
