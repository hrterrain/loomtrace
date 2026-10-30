import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TRACES = ROOT / "traces"
DB_PATH = ROOT / "loomtrace.db"          # review queue + human feedback
CHECKPOINT_DB = ROOT / "checkpoints.db"  # LangGraph state after every node

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 0.6
COMPLETENESS_THRESHOLD = 0.8

TRACES.mkdir(exist_ok=True)
