from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    golden_dir: Path = ROOT / "data" / "golden"
    taxonomy_path: Path = ROOT / "data" / "processed" / "taxonomy.yaml"
    selection_path: Path = ROOT / "data" / "processed" / "selected_brand.json"
    faiss_index_path: Path = ROOT / "data" / "processed" / "faiss.index"
    faiss_meta_path: Path = ROOT / "data" / "processed" / "retrieval_index_meta.json"
    brand: str | None = os.getenv("SUPPORT_BRAND") or None
    random_seed: int = 42
    min_intent_confidence: float = 0.45
    min_retrieval_similarity: float = 0.35
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SETTINGS = Settings()
