"""CLI entry point for bytecanary."""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import EvalConfig
from .decode import ByteTokenizer
from .evaluate import Level0Evaluator
from .evaluate_level1 import Level1Evaluator


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="bytecanary",
        description="Evaluate UTF-8 structural validity of a language model's byte-level generation.",
    )
    p.add_argument("model", help="HuggingFace model name or local path")
    p.add_argument("--level", type=int, choices=[0, 1], default=None,
                   help="Evaluation level (0 or 1). Default: run both if level1-data is provided, else level 0 only.")
    p.add_argument("--eval-set", default=None, help="Custom Level 0 eval set JSON (default: bundled CJK 4000)")
    p.add_argument("--level1-data", default=None,
                   help="Path to Level 1 synthetic data directory (contains ja/ko/zh subdirs)")
    p.add_argument("--output-dir", default="bytecanary_results", help="Output directory")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")
    p.add_argument("--max-new-tokens", type=int, default=5, help="Tokens to generate per sample")
    p.add_argument("--device", default=None, help="Device (cuda, cuda:0, cpu, mps). Auto-detect if omitted.")
    p.add_argument("--dtype", default=None, choices=["float16", "bfloat16", "float32"], help="Model dtype")
    p.add_argument("--trial", action="store_true", help="Trial mode (limited samples)")
    p.add_argument("--trial-samples", type=int, default=256, help="Samples per language in trial mode")
    p.add_argument("--do-sample", action="store_true", help="Use sampling instead of greedy")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling p")
    p.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    p.add_argument("--trust-remote-code", action="store_true", help="Trust remote code for model/tokenizer")
    p.add_argument("--no-details", action="store_true", help="Skip saving detailed per-character results")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    config = EvalConfig(
        model=args.model,
        device=args.device,
        eval_set=args.eval_set,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        trial=args.trial,
        trial_samples=args.trial_samples,
        temperature=args.temperature,
        do_sample=args.do_sample,
        save_detailed=not args.no_details,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        level1_data=args.level1_data,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    device = _resolve_device(config.device)
    dtype = _resolve_dtype(config.dtype, device)

    print("=" * 72)
    print("ByteCanary - UTF-8 Structural Validity Evaluation")
    print("=" * 72)
    print(f"Model:  {config.model}")
    print(f"Device: {device}")
    print(f"Dtype:  {dtype}")
    print(f"Batch:  {config.batch_size}")
    print("=" * 72)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model, trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building byte map...")
    byte_tok = ByteTokenizer(tokenizer)
    print(f"  Type: {byte_tok._type}, space_prefix: {byte_tok.adds_space_prefix}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        torch_dtype=dtype,
        trust_remote_code=config.trust_remote_code,
    )
    model = model.to(device)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    levels = []
    if args.level is not None:
        levels = [args.level]
    elif config.level1_data:
        levels = [0, 1]
    else:
        levels = [0]

    if 0 in levels:
        evaluator = Level0Evaluator(model, tokenizer, byte_tok, config)
        evaluator.run()

    if 1 in levels:
        evaluator = Level1Evaluator(model, tokenizer, byte_tok, config)
        evaluator.run()

    print("\nDone.")
    return 0


def _resolve_device(device_str):
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(dtype_str, device):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    if device.type == "cuda":
        return torch.float16
    return torch.float32


if __name__ == "__main__":
    sys.exit(main())
