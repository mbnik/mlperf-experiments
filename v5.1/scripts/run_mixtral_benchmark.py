#!/usr/bin/env python3
"""
MLPerf Benchmark Setup and Runner - Mixtral-8x7B Text Generation

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
import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("Mixtral-Benchmark")

# Synthetic prompts for testing (MLPerf uses OpenOrca, MBXP, GSM8K datasets)
SYNTHETIC_PROMPTS = [
    "[INST] Summarize the following article: The stock market experienced significant volatility today. [/INST]",
    "[INST] Write a brief summary of: Scientists have made a breakthrough in renewable energy research. [/INST]",
    "[INST] Summarize: The city announced new public transportation improvements. [/INST]",
    "[INST] Brief summary: A major tech company revealed their latest AI product. [/INST]",
    "[INST] Summarize this news: Climate scientists released new findings about global temperatures. [/INST]",
    "[INST] Create a summary: The sports team celebrated their championship victory. [/INST]",
    "[INST] Summarize: New health research suggests benefits of Mediterranean diet. [/INST]",
    "[INST] Write summary: The museum opened a new exhibition of modern art. [/INST]",
    "[INST] Summarize article: Economic reports indicate growth in the tech sector. [/INST]",
    "[INST] Create brief summary: Education reforms were announced by the government. [/INST]",
]


def load_mixtral_dataset(data_dir: str, max_examples: int) -> Tuple[Optional[List], Optional[List], Dict]:
    """Load MLPerf combined dataset (OpenOrca + GSM8k + MBXP)"""
    test_path = Path(data_dir) / "test.json"
    
    if not test_path.exists():
        log.warning(f"Test data not found at {test_path}")
        return None, None, {}
    
    with open(test_path) as f:
        data = json.load(f)
    
    total_available = len(data)
    
    # Count sources
    sources = {}
    for item in data:
        src = item.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    log.info(f"Loaded {len(data)} examples from MLPerf combined dataset")
    log.info(f"  Source distribution: {sources}")
    
    prompts = []
    references = []
    
    for item in data[:max_examples]:
        question = item.get("question", item.get("input", item.get("instruction", "")))[:1500]
        prompt = f"[INST] {question} [/INST]"
        prompts.append(prompt)
        references.append(item.get("response", item.get("output", "")))
    
    data_info = {
        'type': 'real',
        'dataset': 'MLPerf Combined (OpenOrca + GSM8k + MBXP)',
        'source': str(data_dir),
        'samples_used': len(prompts),
        'samples_available': total_available,
        'source_distribution': sources,
        'task': 'instruction_following',
        'verified': True,
        'mlperf_compliant': True,
        'note': 'MLPerf official dataset for Mixtral benchmark'
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
        'task': 'text_summarization',
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
    cmd_parts.extend(["--max-examples", str(args.max_examples)])
    cmd_parts.extend(["--data-type", args.data_type])
    cmd_parts.extend(["--data-dir", args.data_dir])
    cmd_parts.extend(["--output-dir", args.output_dir])
    cmd_parts.extend(["--max-new-tokens", str(args.max_new_tokens)])
    
    if args.offload:
        cmd_parts.append("--offload")
    if args.use_4bit:
        cmd_parts.append("--4bit")
    if args.use_8bit:
        cmd_parts.append("--8bit")
    if args.mlperf:
        cmd_parts.append("--mlperf")
    
    return " ".join(cmd_parts)


def get_args():
    parser = argparse.ArgumentParser(description="Mixtral-8x7B Benchmark")
    parser.add_argument("--model-name", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device: cuda (GPU) or cpu")
    parser.add_argument("--offload", action="store_true",
                       help="Enable CPU offloading for limited VRAM (required for Mixtral)")
    parser.add_argument("--4bit", dest="use_4bit", action="store_true",
                       help="Use 4-bit quantization (~24GB VRAM)")
    parser.add_argument("--8bit", dest="use_8bit", action="store_true",
                       help="Use 8-bit quantization (~48GB VRAM, or ~16GB with --offload)")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--data-type", type=str, default="synthetic",
                       choices=["synthetic", "real"])
    parser.add_argument("--data-dir", type=str, default="data/mixtral")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--mlperf", action="store_true",
                       help="Use MLPerf official settings")
    return parser.parse_args()


def load_model(args):
    """Load Mixtral model with appropriate device configuration"""
    log.info(f"Loading model: {args.model_name}")
    log.info(f"Device: {args.device}, Offload: {args.offload}")
    log.info("")
    log.info("NOTE: Mixtral-8x7B is a large MoE model (~93GB parameters)")
    log.info("      Recommended: --4bit --offload for consumer GPUs")
    log.info("")
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        log.error("CUDA not available. Options:")
        log.error("  1. Use --cpu for CPU-only mode (very slow)")
        log.error("  2. Install CUDA and GPU drivers")
        raise SystemExit(1)
    
    # Determine quantization
    quantization_config = None
    use_8bit_offload = args.use_8bit and args.offload
    use_4bit_offload = args.use_4bit and args.offload
    
    if use_8bit_offload:
        # 8-bit with CPU offloading workaround
        try:
            from transformers import BitsAndBytesConfig
            log.info("Using 8-bit quantization with CPU offloading")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
        except ImportError:
            log.error("bitsandbytes required for quantization. Install with:")
            log.error("  pip install bitsandbytes")
            raise SystemExit(1)
    elif use_4bit_offload:
        # 4-bit with CPU offloading - experimental, may not work on all GPUs
        try:
            from transformers import BitsAndBytesConfig
            log.info("Using 4-bit quantization with CPU offloading (experimental)")
            log.warning("Note: 4-bit + offload may fail on GPUs with limited VRAM")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        except ImportError:
            log.error("bitsandbytes required for quantization. Install with:")
            log.error("  pip install bitsandbytes")
            raise SystemExit(1)
    elif args.use_4bit or args.use_8bit:
        try:
            from transformers import BitsAndBytesConfig
            if args.use_4bit:
                log.info("Using 4-bit quantization (~24GB VRAM)")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                log.info("Using 8-bit quantization (~48GB VRAM)")
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        except ImportError:
            log.error("bitsandbytes required for quantization. Install with:")
            log.error("  pip install bitsandbytes")
            raise SystemExit(1)
    
    if args.device == "cpu":
        if quantization_config:
            log.warning("Quantization only works on GPU, ignoring")
            quantization_config = None
        device_map = None
        torch_dtype = torch.float32
        log.info("Loading on CPU (this will be VERY slow for Mixtral)...")
    elif use_8bit_offload or use_4bit_offload:
        # Quantization with CPU offloading
        device_map = "auto"
        torch_dtype = torch.float16
        log.info("Loading with quantization + CPU offloading...")
    elif args.offload:
        # FP16 with CPU offloading (no quantization)
        device_map = "auto"
        torch_dtype = torch.float16
        log.info("Loading with GPU+CPU offloading (FP16)...")
    else:
        device_map = {"": 0} if not quantization_config else "auto"
        torch_dtype = torch.float16
        log.info("Loading on GPU...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )
    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
        print_oom_error(args)
        raise SystemExit(1)
    except ValueError as e:
        if "CPU or the disk" in str(e) or "enough GPU RAM" in str(e):
            print_oom_error(args)
            raise SystemExit(1)
        raise
    
    log.info("Model loaded successfully!")
    return model, tokenizer


def print_oom_error(args):
    """Print helpful OOM error message with suggestions"""
    log.error("=" * 60)
    log.error("CUDA OUT OF MEMORY!")
    log.error("=" * 60)
    log.error("Mixtral-8x7B requires significant VRAM.")
    log.error("")
    log.error("Current settings:")
    log.error(f"  --4bit: {args.use_4bit}, --8bit: {args.use_8bit}, --offload: {args.offload}")
    log.error("")
    log.error("Try one of these configurations (ordered by VRAM needed):")
    log.error("")
    log.error("  1. --offload              : FP16 with full CPU offload (slowest, ~8GB VRAM)")
    log.error("  2. --8bit --offload       : 8-bit + CPU offload (~16GB VRAM)")
    log.error("  3. --4bit --offload       : 4-bit + CPU offload (~12GB VRAM, experimental)")
    log.error("  4. --4bit                 : 4-bit quantization only (~24GB VRAM)")
    log.error("  5. --8bit                 : 8-bit quantization only (~48GB VRAM)")
    log.error("  6. --cpu                  : Run on CPU only (extremely slow)")
    log.error("")
    log.error("Example:")
    log.error("  ./scripts/run_mixtral.sh --offload --mlperf")
    log.error("")


def calculate_rouge_l(reference, hypothesis):
    """Simple ROUGE-L F1 calculation"""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if not ref_words or not hyp_words:
        return 0.0
    
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_len = dp[m][n]
    precision = lcs_len / n if n > 0 else 0
    recall = lcs_len / m if m > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_benchmark(model, tokenizer, args, prompts: List[str], references: List[str] = None, data_info: Dict = None):
    """Run the benchmark"""
    log.info("=" * 60)
    log.info("Starting Mixtral-8x7B Benchmark")
    log.info("=" * 60)
    log.info(f"Data type: {args.data_type}")
    
    num_examples = len(prompts)
    log.info(f"Processing {num_examples} prompts")
    
    results = []
    total_time = 0
    total_tokens = 0
    total_rouge = 0
    rouge_count = 0
    
    # Warmup
    log.info("Warmup...")
    inputs = tokenizer(prompts[0], return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    log.info("\nRunning benchmark...")
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        total_time += elapsed
        
        output_len = outputs.shape[1] - input_len
        total_tokens += output_len
        
        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        # Calculate ROUGE-L if we have references
        rouge = None
        if references and i < len(references):
            rouge = calculate_rouge_l(references[i], generated)
            total_rouge += rouge
            rouge_count += 1
        
        tokens_per_sec = output_len / elapsed if elapsed > 0 else 0
        
        if (i + 1) % 2 == 0 or i == num_examples - 1:
            rouge_str = f", ROUGE-L: {rouge:.3f}" if rouge is not None else ""
            log.info(f"  [{i+1}/{num_examples}] {output_len} tokens, {tokens_per_sec:.1f} tok/s{rouge_str}")
        
        results.append({
            "sample_id": i,
            "generated": generated[:200],
            "tokens_generated": output_len,
            "latency_sec": elapsed,
            "tokens_per_sec": tokens_per_sec,
            "rouge_l": rouge,
        })
    
    # Summary
    avg_latency = total_time / num_examples
    avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    avg_rouge = total_rouge / rouge_count if rouge_count > 0 else None
    
    summary = {
        "model": args.model_name,
        "device": args.device,
        "data_type": args.data_type,
        "max_new_tokens": args.max_new_tokens,
        "mlperf_mode": args.mlperf,
        "mlperf_compliant": args.mlperf and args.data_type == "real",
        "num_samples": num_examples,
        "total_tokens": total_tokens,
        "total_time_sec": total_time,
        "avg_latency_sec": avg_latency,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "avg_rouge_l": avg_rouge,
        "timestamp": datetime.now().isoformat(),
    }
    
    print("\n" + "=" * 60)
    if args.mlperf:
        if args.data_type == "synthetic":
            print("MIXTRAL-8x7B BENCHMARK SUMMARY - MLPerf Settings (SYNTHETIC DATA)")
            print("⚠️  NOT COMPARABLE TO OFFICIAL MLPERF RESULTS")
        else:
            print("MIXTRAL-8x7B BENCHMARK SUMMARY - MLPerf Compliant")
    else:
        print("MIXTRAL-8x7B BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Device:             {args.device}")
    print(f"Data Type:          {args.data_type}")
    print(f"Max New Tokens:     {args.max_new_tokens}")
    if args.mlperf:
        print(f"MLPerf Mode:        ENABLED")
    print(f"Total Samples:      {num_examples}")
    print(f"Total Tokens:       {total_tokens}")
    print(f"Total Time:         {total_time:.2f} seconds")
    print(f"Avg Latency:        {avg_latency:.2f} sec/sample")
    print(f"Throughput:         {avg_tokens_per_sec:.2f} tokens/sec")
    if avg_rouge is not None:
        print(f"Average ROUGE-L:    {avg_rouge:.3f}")
    print("=" * 60)
    
    if avg_tokens_per_sec > 10:
        print("Performance:        🚀 Excellent")
    elif avg_tokens_per_sec > 2:
        print("Performance:        ✅ Good")
    else:
        print("Performance:        ⚠️ Slow (try --4bit --offload)")
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
        
        summary['data_info'] = data_info
    
    # Print command for reproducibility
    print("\n" + "=" * 60)
    print("COMMAND")
    print("=" * 60)
    print(build_command(args))
    print("=" * 60)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"mixtral_benchmark_{args.data_type}_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    
    log.info(f"\nResults saved to: {output_file}")
    return summary


def main():
    args = get_args()
    
    # Load data first
    references = None
    if args.data_type == "real":
        prompts, references, data_info = load_mixtral_dataset(args.data_dir, args.max_examples)
        if prompts is None:
            log.warning("Falling back to synthetic data")
            prompts, data_info = get_synthetic_data(args.max_examples)
    else:
        prompts, data_info = get_synthetic_data(args.max_examples)
    
    model, tokenizer = load_model(args)
    try:
        run_benchmark(model, tokenizer, args, prompts=prompts, references=references, data_info=data_info)
    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
        log.error("")
        log.error("=" * 60)
        log.error("CUDA OUT OF MEMORY DURING INFERENCE!")
        log.error("=" * 60)
        log.error("Model loaded but ran out of memory during generation.")
        log.error("")
        log.error("Current settings:")
        log.error(f"  --4bit: {args.use_4bit}, --8bit: {args.use_8bit}, --offload: {args.offload}")
        log.error("")
        log.error("Your GPU doesn't have enough VRAM for this configuration.")
        log.error("Try a configuration with lower VRAM requirements:")
        log.error("")
        log.error("  1. --offload              : FP16 with full CPU offload (slowest, ~8GB VRAM)")
        log.error("  2. --cpu                  : Run on CPU only (extremely slow)")
        log.error("")
        log.error("Example:")
        log.error("  ./scripts/run_mixtral.sh --offload --mlperf")
        log.error("")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
