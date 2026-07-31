"""Central configuration: paths and constants used across the project."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

# --- RAG ---
CHROMA_DIR = PROJECT_ROOT / "storage" / "chroma"
CHUNK_SIZE = 1200          # characters per chunk
CHUNK_OVERLAP = 200        # characters of overlap between chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # local, free, runs on CPU
RETRIEVAL_TOP_K = 6
ANTHROPIC_MODEL = "claude-sonnet-4-6"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- Generation backend ---
# "anthropic" (default, needs ANTHROPIC_API_KEY) or "local" (needs
# LOCAL_MODEL_PATH pointing at a GGUF file, runs via llama-cpp-python)
LLM_BACKEND = os.getenv("LLM_BACKEND", "anthropic")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH")

# --- Data ---
DATA_DIR = PROJECT_ROOT / "storage" / "market_data"

# --- Outputs (backtests, paper trading logs) ---
OUTPUT_DIR = PROJECT_ROOT / "outputs"

for d in (CHROMA_DIR, DATA_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)
