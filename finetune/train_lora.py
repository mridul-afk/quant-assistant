"""
QLoRA fine-tune for response STYLE/BEHAVIOR on quant_style_dataset.jsonl.

This is deliberately a small, cheap fine-tune -- a few hundred examples,
a few epochs, LoRA adapters only (not full fine-tuning). It is not trying
to teach the model finance facts; that's what RAG (rag/query.py) is for.

Hardware note: 4-bit QLoRA on a 7B model needs roughly 6-8GB VRAM. A laptop
RTX 3050 (typically 4GB) is tight for 7B -- either use a 3B model locally,
or rent a cloud GPU (RunPod/Vast.ai, a few dollars for a run this size) and
bring the resulting adapter back to run locally via llama-cpp-python.

Usage:
    pip install -r finetune/requirements-finetune.txt
    python finetune/train_lora.py \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --dataset finetune/quant_style_dataset.jsonl \
        --output-dir finetune/output/qwen2.5-7b-quant-style \
        --epochs 3

Then merge and convert to GGUF with finetune/merge_and_export.py so it can
be loaded by rag/llm_backend.py's LocalLlamaBackend, same as any other
local model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def format_example(example: dict) -> str:
    """Turns a {system, instruction, output} record into a chat-formatted
    training string. Uses a generic ChatML-style template; swap for your
    base model's actual chat template if it differs (check the model's
    tokenizer_config.json / apply_chat_template).
    """
    return (
        f"<|im_start|>system\n{example['system']}<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    )


def load_dataset_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune for quant-assistant response style")
    parser.add_argument("--base-model", required=True,
                         help="HF model id or local path, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=Path, default=Path("finetune/quant_style_dataset.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("finetune/output/lora-adapter"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=512,
                         help="Lower this (e.g. 256) if you still hit OOM on a small local GPU")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                         help="Disable gradient checkpointing (uses more VRAM but trains faster; "
                              "leave enabled on small/local GPUs)")
    parser.add_argument("--device-map", default="auto",
                         help="Use 'cuda:0' instead of 'auto' if loading hangs -- 'auto' runs a "
                              "memory-profiling pass that can stall on single-GPU quantized loads")
    args = parser.parse_args()

    # heavy deps imported lazily so `python cli.py backtest` etc. never
    # needs transformers/peft/bitsandbytes installed
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    records = load_dataset_jsonl(args.dataset)
    if len(records) < 20:
        print(
            f"Warning: only {len(records)} examples in {args.dataset}. "
            "A seed set this small will teach format/tone weakly at best -- "
            "extend the dataset toward 200-500 examples before trusting the "
            "result. Proceeding anyway since this is a starter run."
        )

    texts = [format_example(r) for r in records]
    dataset = Dataset.from_dict({"text": texts})

    print(f"Loading base model: {args.base_model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map=args.device_map,
    )

    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # required when gradient checkpointing is on

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        dataset_text_field="text",
        max_length=args.max_seq_length,  # trl >=1.0 renamed max_seq_length -> max_length
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,  # trl >=1.0 renamed tokenizer= -> processing_class=
    )

    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"\nLoRA adapter saved to {args.output_dir}")
    print("Next: python finetune/merge_and_export.py to merge and produce a GGUF for local inference.")


if __name__ == "__main__":
    main()
