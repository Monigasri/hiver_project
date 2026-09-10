"""Schema-tolerant loader and conversation reconstruction for Kaggle Twitter CS."""
from __future__ import annotations
import re, json
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED = {"tweet_id", "author_id", "inbound", "created_at", "text"}

def find_raw_file(raw_dir: Path) -> Path:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No CSV found. Download customer_support_replies.csv from Kaggle and place it in data/raw/.")
    # Prefer the canonical Kaggle filename; otherwise avoid accidentally choosing a small sample.
    return next((p for p in files if p.name.lower()=="twcs.csv"), max(files,key=lambda p:p.stat().st_size))

def _id(value: object) -> str:
    """Canonicalise CSV IDs read as either strings or floats without altering genuine IDs."""
    value=str(value)
    return value[:-2] if value.endswith('.0') else value

def prepare_chunked(path: Path, processed_dir: Path, golden_dir: Path, limit: int | None=None, brand_override: str|None=None, golden_size: int=200, seed: int=42, chunk_size: int=100_000) -> dict:
    """Three-pass bounded-memory preparation of the large Kaggle CSV.

    Pass one records schema/quality and ranks support accounts. Pass two retains only
    direct responses for the selected account. Pass three fetches only their inbound parents.
    """
    header=pd.read_csv(path,nrows=0); columns=list(header.columns); missing=REQUIRED-set(columns)
    if missing: raise ValueError(f"Unsupported schema; missing columns: {sorted(missing)}")
    usecols=['tweet_id','author_id','inbound','created_at','text','in_response_to_tweet_id']
    totals=Counter(); nulls=Counter(); outbound=Counter(); direct=Counter(); seen=set(); duplicate_ids=0; rows=0
    for chunk in pd.read_csv(path,usecols=usecols,chunksize=chunk_size,dtype={'tweet_id':'string','author_id':'string','in_response_to_tweet_id':'string','text':'string'},keep_default_na=True,nrows=limit):
        rows+=len(chunk); totals['rows']+=len(chunk)
        for col in usecols: nulls[col]+=int(chunk[col].isna().sum())
        ids=chunk.tweet_id.dropna().map(_id)
        for tweet_id in ids:
            if tweet_id in seen: duplicate_ids+=1
            else: seen.add(tweet_id)
        outbound_rows=chunk[~chunk.inbound.fillna(False).astype(bool) & chunk.author_id.notna()]
        outbound.update(outbound_rows.author_id.astype(str))
        direct.update(outbound_rows.loc[outbound_rows.in_response_to_tweet_id.notna(),'author_id'].astype(str))
    if not outbound: raise ValueError('No outbound support accounts were found.')
    ranking=pd.DataFrame([{'brand':b,'support_agent_tweets':int(n),'direct_agent_replies':int(direct[b])} for b,n in outbound.items()])
    ranking['selection_score']=0.45*np.log1p(ranking.support_agent_tweets)+0.55*np.log1p(ranking.direct_agent_replies)
    ranking=ranking.sort_values(['selection_score','direct_agent_replies'],ascending=False).reset_index(drop=True)
    brand=str(brand_override or ranking.iloc[0].brand)
    if brand not in set(ranking.brand): raise ValueError(f'Brand {brand} is not an outbound account in this dataset.')
    # Retain only one selected account's direct responses; this bounded frame is the final corpus candidate.
    agent_frames=[]; parent_ids=set()
    for chunk in pd.read_csv(path,usecols=usecols,chunksize=chunk_size,dtype={'tweet_id':'string','author_id':'string','in_response_to_tweet_id':'string','text':'string'},keep_default_na=True,nrows=limit):
        mask=(~chunk.inbound.fillna(False).astype(bool)) & (chunk.author_id.astype(str)==brand) & chunk.in_response_to_tweet_id.notna() & chunk.text.notna()
        part=chunk.loc[mask].copy()
        if not part.empty:
            part['parent_id']=part.in_response_to_tweet_id.map(_id); parent_ids.update(part.parent_id); agent_frames.append(part)
    agents=pd.concat(agent_frames,ignore_index=True) if agent_frames else pd.DataFrame()
    customer_frames=[]
    for chunk in pd.read_csv(path,usecols=usecols,chunksize=chunk_size,dtype={'tweet_id':'string','author_id':'string','in_response_to_tweet_id':'string','text':'string'},keep_default_na=True,nrows=limit):
        ids=chunk.tweet_id.map(_id)
        part=chunk[chunk.inbound.fillna(False).astype(bool) & ids.isin(parent_ids) & chunk.text.notna()].copy()
        if not part.empty: part['parent_id']=part.tweet_id.map(_id); customer_frames.append(part)
    customers=pd.concat(customer_frames,ignore_index=True) if customer_frames else pd.DataFrame()
    if agents.empty or customers.empty: raise ValueError(f'No usable direct customer-agent cases for selected brand {brand}.')
    agents['agent_timestamp']=pd.to_datetime(agents.created_at,utc=True,errors='coerce'); customers['customer_timestamp']=pd.to_datetime(customers.created_at,utc=True,errors='coerce')
    agents=agents.dropna(subset=['agent_timestamp']).sort_values('agent_timestamp').drop_duplicates('parent_id')
    customers=customers.dropna(subset=['customer_timestamp']).drop_duplicates('parent_id')
    cases=agents.merge(customers[['parent_id','text','customer_timestamp']],on='parent_id',how='inner',suffixes=('_agent','_customer'))
    cases=pd.DataFrame({'conversation_id':cases.parent_id.astype(str),'customer_message':cases.text_customer.astype(str),'customer_message_normalized':cases.text_customer.map(normalize_text),'agent_response':cases.text_agent.astype(str),'customer_timestamp':cases.customer_timestamp,'agent_timestamp':cases.agent_timestamp,'brand':brand,'agent_tweet_id':cases.tweet_id.astype(str)}).drop_duplicates('conversation_id')
    train,golden=split_cases(cases,golden_size,seed)
    processed_dir.mkdir(parents=True,exist_ok=True); golden_dir.mkdir(parents=True,exist_ok=True)
    train.to_csv(processed_dir/'retrieval_cases.csv',index=False); cases.to_csv(processed_dir/'historical_cases.csv',index=False); golden.to_csv(golden_dir/'golden_template.csv',index=False)
    selected={'brand':brand,'source_file':path.name,'source_bytes':path.stat().st_size,'sampling':{'method':'deterministic row-prefix','limit':limit,'chunk_size':chunk_size,'seed':seed},'dataset':{'rows_read':rows,'columns':columns,'missing_values':dict(nulls),'duplicate_tweet_ids':duplicate_ids},'statistics':{'customer_tweets':int(len(customers)),'support_agent_tweets':int(len(agents)),'conversation_count':int(len(cases)),'conversations_with_agent_response':int(len(cases)),'usable_resolved_examples':int(len(cases)),'avg_conversation_length':2.0,'median_conversation_length':2.0},'selection_rule':'0.45*log1p(outbound support tweets) + 0.55*log1p(outbound direct replies)'}
    (processed_dir/'selected_brand.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    ranking.to_csv(processed_dir/'brand_ranking.csv',index=False)
    (processed_dir/'dataset_inspection.json').write_text(json.dumps(selected['dataset']|{'source_file':path.name,'source_bytes':path.stat().st_size},indent=2),encoding='utf-8')
    return selected

def normalize_text(value: object) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return re.sub(r"\s+", " ", text)

def load_and_clean(path: Path, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=limit, low_memory=False)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Unsupported schema; missing columns: {sorted(missing)}")
    df = df.copy()
    df["text_original"] = df["text"]
    df["text"] = df["text"].map(normalize_text)
    df = df[df["text"].str.len().gt(0)].drop_duplicates(subset=["tweet_id"], keep="first")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["tweet_id", "author_id", "created_at"])
    df["tweet_id"] = df["tweet_id"].astype(str)
    df["author_id"] = df["author_id"].astype(str)
    df["is_customer"] = df["inbound"].astype(str).str.lower().isin(["true", "1"])
    return df.sort_values("created_at").reset_index(drop=True)

def candidate_brand_stats(df: pd.DataFrame) -> pd.DataFrame:
    # Brand is the outbound author; inbound tweets inherit the support account of their direct reply where available.
    outbound = df[~df.is_customer]
    rows=[]
    for brand, group in outbound.groupby("author_id"):
        response_ids = set(group.get("in_response_to_tweet_id", pd.Series(dtype=str)).dropna().astype(str))
        customers = df[df.tweet_id.isin(response_ids) & df.is_customer]
        convo_ids = set(customers.tweet_id) | set(group.tweet_id)
        convo_lengths = [len(g) for _, g in df[df.tweet_id.isin(convo_ids)].groupby("tweet_id")]
        rows.append({"brand":brand,"total_tweets":int(len(group)+len(customers)),"customer_tweets":int(len(customers)),"support_agent_tweets":int(len(group)),"conversation_count":int(len(response_ids)),"conversations_with_agent_response":int(len(response_ids)),"avg_conversation_length":float(np.mean(convo_lengths) if convo_lengths else 0),"median_conversation_length":float(np.median(convo_lengths) if convo_lengths else 0)})
    result=pd.DataFrame(rows)
    if not result.empty:
        # Transparent volume-and-response score, deliberately not popularity alone.
        result["selection_score"] = np.log1p(result.customer_tweets)*0.4 + np.log1p(result.support_agent_tweets)*0.35 + np.log1p(result.conversations_with_agent_response)*0.25
        result=result.sort_values("selection_score",ascending=False).reset_index(drop=True)
    return result

def build_cases(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    agents=df[(~df.is_customer)&(df.author_id==str(brand))].copy()
    if "in_response_to_tweet_id" not in agents:
        raise ValueError("Dataset has no in_response_to_tweet_id; cannot pair customer replies.")
    agents["in_response_to_tweet_id"]=agents["in_response_to_tweet_id"].astype("string").str.replace(r"\.0$", "", regex=True)
    customers=df[["tweet_id","text_original","text","created_at","is_customer"]].rename(columns={"tweet_id":"parent_id","text_original":"customer_message","text":"customer_message_normalized","created_at":"customer_timestamp"})
    customers["parent_id"]=customers["parent_id"].astype("string")
    paired=agents.merge(customers,left_on="in_response_to_tweet_id",right_on="parent_id",how="inner")
    paired=paired[paired["is_customer_y"]].copy()
    paired["conversation_id"]=paired.parent_id.astype(str)
    paired["agent_response"]=paired.text_original
    paired["brand"]=str(brand)
    cols=["conversation_id","customer_message","customer_message_normalized","agent_response","customer_timestamp","created_at","brand","tweet_id"]
    return paired[cols].rename(columns={"created_at":"agent_timestamp","tweet_id":"agent_tweet_id"}).drop_duplicates("conversation_id")

def split_cases(cases: pd.DataFrame, golden_size: int=200, seed: int=42) -> tuple[pd.DataFrame,pd.DataFrame]:
    n=min(golden_size, max(0, len(cases)//5))
    golden=cases.sample(n=n,random_state=seed) if n else cases.iloc[0:0]
    return cases[~cases.conversation_id.isin(golden.conversation_id)].copy(),golden.copy()

def save_selected_brand(path: Path, brand: str, stats: pd.DataFrame) -> None:
    row=stats.loc[stats.brand.astype(str)==str(brand)].iloc[0].to_dict()
    path.write_text(json.dumps({"brand":str(brand),"selection_rule":"0.40*log1p(customer_tweets) + 0.35*log1p(support_agent_tweets) + 0.25*log1p(conversations_with_agent_response)","statistics":row},indent=2,default=float),encoding="utf-8")
