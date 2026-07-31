"""
Merges a trained LoRA adapter into its base model, producing a standalone
model directory. From there, use llama.cpp's convert script (separate repo,
not a pip package) to produce a GGUF file for rag/llm_backend.py's
LocalLlamaBackend.

Usage:
    python finetune/merge_and_export.py \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --adapter-dir finetune/output/lora-adapter \
        --output-dir finetune/output/merged-model
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="cpu",  # merge on CPU to avoid VRAM pressure; slower but reliable
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading LoRA adapter from {args.adapter_dir}")
    merged_model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))

    print("Merging adapter weights into base model...")
    merged_model = merged_model.merge_and_unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print(f"\nMerged model saved to {args.output_dir}")
    print("\nTo use this with rag/llm_backend.py's LocalLlamaBackend, convert to GGUF:")
    print("  1. git clone https://github.com/ggerganov/llama.cpp")
    print("  2. cd llama.cpp && pip install -r requirements.txt")
    print(f"  3. python convert_hf_to_gguf.py {args.output_dir} "
          f"--outfile {args.output_dir}.gguf --outtype q4_k_m")
    print(f"  4. In .env: LOCAL_MODEL_PATH={args.output_dir}.gguf")


if __name__ == "__main__":
    main()
