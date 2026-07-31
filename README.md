# Quant Assistant

A two-part project:

1. **RAG research assistant** over your quant finance library (Hull, Shreve,
   López de Prado, Harris, etc.) — ask it questions and it answers grounded in
   the actual book text, with citations.
2. **Quant trading pipeline** — fetch market data, generate signals, backtest
   them properly, size positions with risk limits, and paper-trade before any
   real money is involved.

These two parts are **intentionally separate**. The LLM explains concepts and
helps you write/debug strategy code. It does not generate trade signals
itself — those come from the statistical/ML models in `strategy/`, validated
by backtests. That split is the whole point: an LLM is a great tutor and
pair-programmer, a bad, unaccountable portfolio manager.

⚠️ **Not financial advice.** This is educational infrastructure. Backtested
performance does not predict future returns. Paper-trade for a meaningful
period before risking real capital, and understand the regulatory
implications in your jurisdiction if you plan to trade systematically or
manage money for others.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

## 1. RAG research assistant

Ingest your PDFs once (this chunks, embeds, and stores them locally — nothing
leaves your machine except the final Q&A call to Claude):

```bash
python cli.py ingest-books --pdf-dir /path/to/your/quantbooks
```

Then ask questions:

```bash
python cli.py ask "How does the triple-barrier method label financial data?"
python cli.py ask "Derive the Black-Scholes PDE the way Hull presents it"
```

Each answer cites which book and approximate location the grounding came
from, so you can go read the original for anything load-bearing.

### Using a local model instead of the Claude API

The embedding step (`rag/ingest.py`, `rag/query.py`) already runs 100% locally
via `sentence-transformers` — no API calls, no cost. The only network call in
the RAG pipeline is the final answer-generation step, which you can point at
a local GGUF model instead of Claude:

```bash
pip install llama-cpp-python  # add --extra-index-url for CUDA, see requirements.txt
```

Download a GGUF instruct model (Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct,
Q4_K_M quantization is a reasonable fit for an RTX 3050's VRAM), then in `.env`:

```
LLM_BACKEND=local
LOCAL_MODEL_PATH=/path/to/your-model.Q4_K_M.gguf
```

Everything else — chunking, embedding, retrieval, citations — is unchanged.
Answer quality will be noticeably below Claude, especially for adhering
strictly to "only use the provided excerpts" — smaller local models are more
prone to filling gaps from their own training data despite the system
prompt telling them not to. Worth knowing before you trust an answer.

## 2. Quant pipeline

Fetch data:

```bash
python cli.py fetch-data --tickers AAPL MSFT NVDA --start 2018-01-01
```

Backtest a signal (two built-in examples: momentum and mean-reversion —
see `strategy/signals.py` to add your own):

```bash
python cli.py backtest --ticker AAPL --signal momentum
```

This prints Sharpe ratio, CAGR, max drawdown, and win rate, and saves an
equity curve plot to `outputs/`.

Run the paper-trading loop (simulated only — no broker connection, no real
orders):

```bash
python cli.py paper-trade --tickers AAPL MSFT --signal momentum --capital 100000
```

This logs simulated fills, position sizes (volatility-targeted, via
`strategy/risk.py`), and equity to `outputs/paper_trading_log.csv` so you can
review performance before ever wiring up a real broker API.

## 3. Fine-tuning (optional, for response style — not facts)

Fine-tuning is the wrong tool for teaching the model your books' content —
that's what RAG already does, reliably, without hallucination risk. What
fine-tuning IS good for is shaping *behavior*: always structuring answers as
Thesis/Evidence/Risks/Position-sizing, always stating assumptions, refusing
bare "buy X" calls, and so on.

```bash
pip install -r finetune/requirements-finetune.txt

# 1. Build the seed instruction dataset (12 examples — extend this before
#    trusting the result; see finetune/build_dataset.py)
python finetune/build_dataset.py

# 2. QLoRA fine-tune a base model (needs a GPU — a laptop RTX 3050 is tight
#    for 7B models; consider a rented cloud GPU for this one-time step)
python finetune/train_lora.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --output-dir finetune/output/lora-adapter

# 3. Merge the adapter and convert to GGUF for local inference
python finetune/merge_and_export.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter-dir finetune/output/lora-adapter \
    --output-dir finetune/output/merged-model
# then follow the printed llama.cpp conversion steps
```

The resulting GGUF drops straight into `LOCAL_MODEL_PATH` and works with the
existing `rag/llm_backend.py` — no other code changes needed.

## Project layout

```
quant-assistant/
├── cli.py                  # entry point, wires everything together
├── config.py                # paths & constants
├── rag/
│   ├── ingest.py            # PDF -> chunks -> embeddings -> Chroma
│   ├── query.py              # retrieval + grounded answer w/ citations
│   └── llm_backend.py         # pluggable: Claude API or local GGUF model
├── data/
│   └── fetch_data.py         # yfinance OHLCV download
├── strategy/
│   ├── signals.py            # momentum, mean-reversion, ...
│   ├── backtest.py           # vectorized backtest + performance metrics
│   └── risk.py                # position sizing, stop-loss, kill switch
├── paper_trading/
│   └── simulator.py           # simulated daily trading loop, no live broker
└── finetune/                # optional: LoRA fine-tuning for response style
    ├── build_dataset.py
    ├── train_lora.py
    └── merge_and_export.py
```

## Extending it

- **New signals**: add a function to `strategy/signals.py` following the
  existing signature (`price_df -> position_series` in `[-1, 1]`), it'll
  show up in `--signal` automatically.
- **New risk rules**: `strategy/risk.py` has hooks for max position size,
  volatility targeting, and portfolio-level drawdown kill switches.
- **Live execution**: deliberately not included. When you're ready, this is
  where you'd add a broker adapter (e.g. Alpaca, Interactive Brokers) behind
  the same interface `paper_trading/simulator.py` uses — swap the simulated
  fill logic for real order submission, keep everything upstream identical.
