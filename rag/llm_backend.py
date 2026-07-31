"""
Pluggable generation backend for rag/query.py.

Two implementations behind one interface:
- AnthropicBackend: calls the Claude API (default, best quality/citation
  adherence, needs ANTHROPIC_API_KEY and internet).
- LocalLlamaBackend: runs a local GGUF model via llama-cpp-python -- the
  same stack Jarvis uses. No API key, no internet, runs on your RTX 3050.

Switch backends by setting LLM_BACKEND=local in .env and pointing
LOCAL_MODEL_PATH at a GGUF file. Everything upstream in rag/query.py is
unchanged either way -- it just calls generate(system, user).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, LLM_BACKEND, LOCAL_MODEL_PATH


class Backend:
    def generate(self, system: str, user: str, max_tokens: int = 1500) -> str:
        raise NotImplementedError


class AnthropicBackend(Backend):
    def __init__(self):
        import anthropic

        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key, "
                "or set LLM_BACKEND=local to use a local model instead."
            )
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def generate(self, system: str, user: str, max_tokens: int = 1500) -> str:
        response = self.client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in response.content if b.type == "text")


class LocalLlamaBackend(Backend):
    """Runs a local GGUF model. Quality depends heavily on which model you
    pick -- a 7-8B instruct model (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct)
    quantized to Q4_K_M is a reasonable fit for an RTX 3050's VRAM and gives
    noticeably better instruction-following than smaller models. This class
    loads the model once and keeps it resident, same pattern as Jarvis's
    Piper-in-memory fix.
    """

    _instance = None  # keep one loaded model resident across calls

    def __init__(self, n_gpu_layers: int = -1, n_ctx: int = 8192):
        from llama_cpp import Llama

        if not LOCAL_MODEL_PATH or not Path(LOCAL_MODEL_PATH).exists():
            raise RuntimeError(
                f"LOCAL_MODEL_PATH not set or file not found: {LOCAL_MODEL_PATH}. "
                "Download a GGUF model (e.g. from Hugging Face) and point "
                "LOCAL_MODEL_PATH at it in .env."
            )

        print(f"Loading local model from {LOCAL_MODEL_PATH} ...")
        self.llm = Llama(
            model_path=str(LOCAL_MODEL_PATH),
            n_gpu_layers=n_gpu_layers,  # -1 = offload as many layers as fit on GPU
            n_ctx=n_ctx,                # needs to be large enough for retrieved excerpts
            verbose=False,
        )

    def generate(self, system: str, user: str, max_tokens: int = 1500) -> str:
        result = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.2,  # low temperature: this is a grounded-answer task, not creative writing
        )
        return result["choices"][0]["message"]["content"]


def get_backend() -> Backend:
    """Returns a cached backend instance based on LLM_BACKEND in config."""
    if get_backend._cached is None:
        if LLM_BACKEND == "local":
            get_backend._cached = LocalLlamaBackend()
        else:
            get_backend._cached = AnthropicBackend()
    return get_backend._cached


get_backend._cached = None
