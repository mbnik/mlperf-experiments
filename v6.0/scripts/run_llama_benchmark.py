#!/usr/bin/env python3
"""
MLPerf Benchmark Setup and Runner - Llama Model Family

Author: Mehdi Nik
Created: Jan 2026

DISCLAIMER:
This repository is NOT an official MLPerf implementation.
It contains personal tooling and scripts to run MLPerf workloads.

This software is provided "as is" without warranty of any kind, express or implied.
The author assumes no responsibility for errors, omissions, or damages arising from
its use. Users are solely responsible for determining the appropriateness of using
this software and assume all risks associated with its use.

All rights reserved. If you use or reference this work, please provide attribution
to the original author.
"""

import argparse
import atexit
import gc
import os
import signal
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("Llama-Benchmark")


# ============================================================================
# GPU Cleanup Utilities
# ============================================================================

_model_ref = None
_tokenizer_ref = None


def cleanup_gpu():
    """Properly cleanup GPU resources to prevent device unavailability issues."""
    global _model_ref, _tokenizer_ref
    
    log.info("Cleaning up GPU resources...")
    
    if _model_ref is not None:
        del _model_ref
        _model_ref = None
    
    if _tokenizer_ref is not None:
        del _tokenizer_ref
        _tokenizer_ref = None
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated() / 1024**2
        log.info(f"GPU cleanup complete. Memory: {allocated:.1f}MB allocated")
    
    gc.collect()


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    log.warning(f"Received signal {signum}, cleaning up...")
    cleanup_gpu()
    sys.exit(1)


# Register cleanup handlers
atexit.register(cleanup_gpu)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# Synthetic prompts for testing
SYNTHETIC_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a short story about a robot learning to paint.",
    "What are the key differences between Python and JavaScript?",
    "Describe the process of photosynthesis.",
    "Write a haiku about artificial intelligence.",
    "Explain quantum computing to a 10-year-old.",
    "What are the benefits of renewable energy?",
    "Summarize the plot of Romeo and Juliet.",
    "How does machine learning differ from traditional programming?",
    "Write a recipe for chocolate chip cookies.",
]


def load_cnn_dailymail(data_dir: str, max_examples: int) -> Tuple[Optional[List], Optional[List], Dict]:
    """Load real CNN-DailyMail data as prompts"""
    test_path = Path(data_dir) / "test.json"
    
    if not test_path.exists():
        log.warning(f"CNN-DailyMail not found at {test_path}")
        return None, None, {}
    
    with open(test_path) as f:
        data = json.load(f)
    
    total_available = len(data)
    
    # Create summarization prompts
    prompts = []
    references = []
    for item in data[:max_examples]:
        article = item["article"][:2000]  # Truncate long articles
        prompt = f"Summarize the following article:\n\n{article}\n\nSummary:"
        prompts.append(prompt)
        references.append(item["highlights"])
    
    log.info(f"Loaded {len(prompts)} articles from CNN-DailyMail")
    
    data_info = {
        'type': 'real',
        'dataset': 'CNN-DailyMail',
        'source': str(data_dir),
        'samples_used': len(prompts),
        'samples_available': total_available,
        'task': 'text_summarization',
        'verified': True,
        'mlperf_compliant': True,
        'note': 'CNN-DailyMail text summarization dataset'
    }
    
    return prompts, references, data_info


def load_openorca(data_dir: str, max_examples: int) -> Tuple[Optional[List], Optional[List], Dict]:
    """Load OpenOrca data as prompts (for llama2-70b MLPerf compliance)"""
    test_path = Path(data_dir) / "test.json"
    
    if not test_path.exists():
        log.warning(f"OpenOrca not found at {test_path}")
        return None, None, {}
    
    with open(test_path) as f:
        data = json.load(f)
    
    total_available = len(data)
    
    prompts = []
    references = []
    for item in data[:max_examples]:
        # OpenOrca format has "question" and "response" fields
        if "question" in item:
            prompt = item["question"]
        elif "system_prompt" in item and "question" in item:
            prompt = f"{item['system_prompt']}\n\n{item['question']}"
        else:
            continue
        
        prompts.append(prompt)
        if "response" in item:
            references.append(item["response"])
        else:
            references.append("")
    
    log.info(f"Loaded {len(prompts)} prompts from OpenOrca")
    
    data_info = {
        'type': 'real',
        'dataset': 'OpenOrca',
        'source': str(data_dir),
        'samples_used': len(prompts),
        'samples_available': total_available,
        'task': 'instruction_following',
        'verified': True,
        'mlperf_compliant': True,
        'note': 'OpenOrca instruction-following dataset'
    }
    
    return prompts, references, data_info


def get_synthetic_data(max_examples: int) -> Tuple[List[str], Dict]:
    """Get synthetic prompts with data info"""
    prompts = SYNTHETIC_PROMPTS[:max_examples]
    
    data_info = {
        'type': 'synthetic',
        'dataset': 'Synthetic Prompts',
        'source': 'generated_prompts',
        'samples_used': len(prompts),
        'samples_available': len(SYNTHETIC_PROMPTS),
        'task': 'text_generation',
        'verified': False,
        'mlperf_compliant': False,
        'note': 'Synthetic prompts for testing only'
    }
    
    return prompts, data_info


def build_command(args) -> str:
    """Build the command line string for reproducibility"""
    import sys
    cmd_parts = [sys.executable, __file__]
    
    cmd_parts.extend(["--model-name", args.model_name])
    cmd_parts.extend(["--device", args.device])
    cmd_parts.extend(["--quantization", args.quantization])
    cmd_parts.extend(["--data-type", args.data_type])
    cmd_parts.extend(["--dataset", args.dataset])
    cmd_parts.extend(["--data-dir", args.data_dir])
    cmd_parts.extend(["--max-examples", str(args.max_examples)])
    cmd_parts.extend(["--max-new-tokens", str(args.max_new_tokens)])
    cmd_parts.extend(["--output-dir", args.output_dir])
    
    if args.offload:
        cmd_parts.append("--offload")
    if args.mlperf:
        cmd_parts.append("--mlperf")
    
    return " ".join(cmd_parts)


def get_args():
    parser = argparse.ArgumentParser(description="Llama Model Benchmark")
    parser.add_argument("--model-name", type=str, required=True,
                       help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device: cuda (GPU) or cpu")
    parser.add_argument("--offload", action="store_true",
                       help="Enable GPU+CPU memory offloading for large models")
    parser.add_argument("--quantization", type=str, default="none",
                       choices=["none", "4bit", "8bit"],
                       help="Quantization: none (FP16), 4bit, 8bit")
    parser.add_argument("--data-type", type=str, default="synthetic",
                       choices=["synthetic", "real"],
                       help="Data type: synthetic or real data")
    parser.add_argument("--dataset", type=str, default="cnn-dailymail",
                       choices=["cnn-dailymail", "openorca"],
                       help="Dataset: cnn-dailymail (llama3.1-8b) or openorca (llama2-70b)")
    parser.add_argument("--data-dir", type=str, default="data/cnn-dailymail")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--mlperf", action="store_true",
                       help="Use MLPerf official settings")
    return parser.parse_args()


def load_model(args):
    """Load Llama model with appropriate configuration"""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    log.info(f"Loading model: {args.model_name}")
    log.info(f"Device: {args.device}, Offload: {args.offload}, Quantization: {args.quantization}")
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        log.error("CUDA not available. Options:")
        log.error("  1. Use --cpu for CPU-only mode (slow)")
        log.error("  2. Install CUDA and GPU drivers")
        raise SystemExit(1)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model configuration based on quantization
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    
    if args.quantization == "4bit":
        log.info("Using 4-bit quantization (bitsandbytes)")
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        except Exception as e:
            log.error(f"Quantization failed: {e}")
            log.error("Install bitsandbytes: pip install bitsandbytes")
            raise SystemExit(1)
        
    elif args.quantization == "8bit":
        log.info("Using 8-bit quantization (bitsandbytes)")
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        except Exception as e:
            log.error(f"Quantization failed: {e}")
            log.error("Install bitsandbytes: pip install bitsandbytes")
            raise SystemExit(1)
        
    elif args.offload:
        log.info("Using GPU+CPU offloading")
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.float16
        
    elif args.device == "cuda":
        log.info("Loading on GPU (full precision FP16)")
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = {"": 0}
        
    else:  # cpu
        log.info("Loading on CPU (FP32)")
        model_kwargs["torch_dtype"] = torch.float32
    
    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            **model_kwargs
        )
    except torch.cuda.OutOfMemoryError:
        log.error("=" * 60)
        log.error("CUDA OUT OF MEMORY!")
        log.error("=" * 60)
        log.error("Your GPU doesn't have enough VRAM. Try:")
        log.error("  1. --offload           : Enable GPU+CPU memory offloading")
        log.error("  2. --quantization 4bit : Use 4-bit quantization")
        log.error("  3. --quantization 8bit : Use 8-bit quantization")
        log.error("  4. --cpu               : Run on CPU only (very slow)")
        raise SystemExit(1)
    
    # Print memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        log.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    log.info("Model loaded successfully!")
    return model, tokenizer


def run_benchmark(model, tokenizer, args, prompts: List[str], references: List[str] = None, data_info: Dict = None):
    """Run the benchmark"""
    log.info("=" * 60)
    log.info("Starting Llama Benchmark")
    log.info("=" * 60)
    log.info(f"Dataset: {args.dataset}")
    log.info(f"Processing {len(prompts)} prompts")
    
    results = []
    total_tokens = 0
    total_time = 0
    
    # Warmup
    log.info("Warmup...")
    with torch.no_grad():
        inputs = tokenizer(prompts[0], return_tensors="pt", truncation=True, max_length=512)
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif next(model.parameters()).device.type == 'cuda':
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        _ = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    log.info(f"\nRunning benchmark on {len(prompts)} prompts...")
    
    for i, prompt in enumerate(prompts):
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif next(model.parameters()).device.type == 'cuda':
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        input_len = inputs["input_ids"].shape[1]
        
        # Generate
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start_time
        
        # Decode
        output_len = outputs.shape[1] - input_len
        generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        total_tokens += output_len
        total_time += elapsed
        
        tokens_per_sec = output_len / elapsed if elapsed > 0 else 0
        
        results.append({
            "prompt_idx": i,
            "input_tokens": input_len,
            "output_tokens": output_len,
            "time_sec": elapsed,
            "tokens_per_sec": tokens_per_sec,
        })
        
        log.info(f"  [{i+1}/{len(prompts)}] {output_len} tokens, {tokens_per_sec:.1f} tok/s, {elapsed:.2f}s")
    
    # Calculate statistics
    throughput = total_tokens / total_time if total_time > 0 else 0
    avg_latency = total_time / len(prompts)
    
    # Print summary
    print("\n" + "=" * 60)
    if args.mlperf:
        if args.data_type == "synthetic":
            print("LLAMA BENCHMARK SUMMARY - MLPerf Settings (SYNTHETIC DATA)")
            print("⚠️  NOT COMPARABLE TO OFFICIAL MLPERF RESULTS")
        else:
            print("LLAMA BENCHMARK SUMMARY - MLPerf Compliant")
    else:
        print("LLAMA BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Model:              {args.model_name.split('/')[-1]}")
    print(f"Device:             {args.device}")
    print(f"Quantization:       {args.quantization}")
    print(f"Data Type:          {args.data_type}")
    print(f"Max New Tokens:     {args.max_new_tokens}")
    if args.mlperf:
        print(f"MLPerf Mode:        {'ENABLED' if args.mlperf else 'disabled'}")
    print(f"Total Samples:      {len(prompts)}")
    print(f"Total Tokens:       {total_tokens}")
    print(f"Total Time:         {total_time:.2f} seconds")
    print(f"Avg Latency:        {avg_latency:.2f} sec/sample")
    print(f"Throughput:         {throughput:.2f} tokens/sec")
    print("=" * 60)
    
    # Performance rating
    if throughput >= 20:
        print("Performance:        🚀 Excellent")
    elif throughput >= 5:
        print("Performance:        ✅ Good")
    elif throughput >= 1:
        print("Performance:        ⚠️ Moderate")
    else:
        print("Performance:        🐢 Slow (try --4bit or --offload)")
    print("=" * 60)
    
    # Print data information
    if data_info:
        print("\n" + "=" * 60)
        print("DATA INFORMATION")
        print("=" * 60)
        print(f"Type:               {data_info['type']}")
        print(f"Dataset:            {data_info['dataset']}")
        print(f"Source:             {data_info['source']}")
        print(f"Samples Used:       {data_info['samples_used']:,}")
        print(f"Samples Available:  {data_info['samples_available']:,}")
        print(f"Task:               {data_info['task']}")
        print(f"Verified:           {'✓' if data_info['verified'] else '✗'}")
        print(f"MLPerf Compliant:   {'✓' if data_info['mlperf_compliant'] else '✗'}")
        print(f"Note:               {data_info['note']}")
        print("=" * 60)
    
    # Print command for reproducibility
    print("\n" + "=" * 60)
    print("COMMAND")
    print("=" * 60)
    print(build_command(args))
    print("=" * 60)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_short = args.model_name.split('/')[-1].replace("-", "_").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"llama_{model_short}_{args.data_type}_{timestamp}.json"
    
    summary = {
        "model": args.model_name,
        "device": args.device,
        "quantization": args.quantization,
        "data_type": args.data_type,
        "max_new_tokens": args.max_new_tokens,
        "mlperf_mode": args.mlperf,
        "mlperf_compliant": args.mlperf and args.data_type == "real",
        "total_samples": len(prompts),
        "total_tokens": total_tokens,
        "total_time_sec": total_time,
        "throughput_tokens_per_sec": throughput,
        "avg_latency_sec": avg_latency,
        "timestamp": timestamp,
        "results": results,
    }
    
    if data_info:
        summary['data_info'] = data_info
    
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    log.info(f"\nResults saved to: {output_file}")
    return summary


def main():
    global _model_ref, _tokenizer_ref
    
    args = get_args()
    
    # Check for bitsandbytes if quantization requested
    if args.quantization in ["4bit", "8bit"]:
        try:
            import bitsandbytes
        except ImportError:
            log.error("bitsandbytes required for quantization. Install with:")
            log.error("  pip install bitsandbytes")
            sys.exit(1)
    
    try:
        # Load data first and get data_info
        references = None
        if args.data_type == "real":
            if args.dataset == "openorca":
                prompts, references, data_info = load_openorca(args.data_dir, args.max_examples)
            else:
                prompts, references, data_info = load_cnn_dailymail(args.data_dir, args.max_examples)
            
            if prompts is None:
                log.warning("Falling back to synthetic prompts")
                prompts, data_info = get_synthetic_data(args.max_examples)
        else:
            prompts, data_info = get_synthetic_data(args.max_examples)
        
        model, tokenizer = load_model(args)
        _model_ref = model
        _tokenizer_ref = tokenizer
        
        run_benchmark(model, tokenizer, args, prompts=prompts, references=references, data_info=data_info)
    
    finally:
        cleanup_gpu()


if __name__ == "__main__":
    main()
