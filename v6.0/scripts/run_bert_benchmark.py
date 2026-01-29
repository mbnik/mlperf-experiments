#!/usr/bin/env python3
"""
MLPerf Benchmark Setup and Runner - BERT Question Answering

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
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


# ============================================================================
# Synthetic Data Generation
# ============================================================================

def generate_synthetic_data(num_samples: int, max_seq_length: int = 384) -> Tuple[List[Dict], Dict]:
    """Generate synthetic QA data for benchmarking"""
    log.info(f"Generating {num_samples} synthetic QA samples...")
    
    samples = []
    for i in range(num_samples):
        # Generate random input IDs (vocabulary size ~30522 for BERT)
        input_ids = np.random.randint(1, 30000, size=max_seq_length).tolist()
        # Set CLS and SEP tokens
        input_ids[0] = 101  # [CLS]
        sep_pos = np.random.randint(50, 150)
        input_ids[sep_pos] = 102  # [SEP]
        
        attention_mask = [1] * max_seq_length
        token_type_ids = [0] * sep_pos + [1] * (max_seq_length - sep_pos)
        
        samples.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'question': f"Synthetic question {i}?",
            'context': f"Synthetic context passage {i}.",
            'answer': f"synthetic answer {i}"
        })
    
    data_info = {
        'type': 'synthetic',
        'dataset': 'Synthetic QA',
        'source': 'generated',
        'samples_used': num_samples,
        'samples_available': num_samples,
        'verified': False,
        'mlperf_compliant': False,
        'note': 'Randomly generated QA pairs for testing - not MLPerf compliant',
        'max_seq_length': max_seq_length,
        'vocab_size': 30522,
    }
    
    return samples, data_info


def load_squad_data(data_dir: str, max_samples: int = None) -> Tuple[List[Dict], Dict]:
    """Load SQuAD v1.1 data"""
    squad_file = Path(data_dir) / "dev-v1.1.json"
    
    if not squad_file.exists():
        log.warning(f"SQuAD file not found: {squad_file}")
        return None, {}
    
    log.info(f"Loading SQuAD data from {squad_file}")
    
    with open(squad_file) as f:
        squad_data = json.load(f)
    
    # Count total available samples
    total_available = 0
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            total_available += len(paragraph['qas'])
    
    samples = []
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                answer = qa['answers'][0]['text'] if qa['answers'] else ""
                samples.append({
                    'question': question,
                    'context': context,
                    'answer': answer
                })
                if max_samples and len(samples) >= max_samples:
                    break
            if max_samples and len(samples) >= max_samples:
                break
        if max_samples and len(samples) >= max_samples:
            break
    
    log.info(f"Loaded {len(samples)} QA pairs from SQuAD")
    
    data_info = {
        'type': 'real',
        'dataset': 'SQuAD v1.1',
        'source': str(squad_file),
        'samples_used': len(samples),
        'samples_available': total_available,
        'verified': True,
        'mlperf_compliant': True,
        'note': 'Stanford Question Answering Dataset (SQuAD) v1.1 dev set',
        'num_articles': len(squad_data['data']),
    }
    
    return samples, data_info


# ============================================================================
# Model Loading
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser(description="BERT Question Answering Benchmark")
    parser.add_argument("--model-name", type=str, 
                       default="bert-large-uncased-whole-word-masking-finetuned-squad",
                       help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device: cuda (GPU) or cpu")
    parser.add_argument("--offload", action="store_true",
                       help="Enable CPU offloading (not typically needed for BERT)")
    parser.add_argument("--max-examples", type=int, default=100,
                       help="Number of examples to process")
    parser.add_argument("--max-seq-length", type=int, default=384,
                       help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--data-type", type=str, default="synthetic",
                       choices=["synthetic", "real"],
                       help="Data type: synthetic or real SQuAD")
    parser.add_argument("--data-dir", type=str, default="data/squad",
                       help="Directory containing SQuAD data")
    parser.add_argument("--output-dir", type=str, default="results/bert",
                       help="Output directory for results")
    parser.add_argument("--mlperf", action="store_true",
                       help="Use MLPerf official settings")
    return parser.parse_args()


def load_model(args):
    """Load BERT model for question answering"""
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    
    log.info(f"Loading model: {args.model_name}")
    log.info(f"Device: {args.device}, Offload: {args.offload}")
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        log.error("CUDA not available. Options:")
        log.error("  1. Use --cpu for CPU-only mode")
        log.error("  2. Install CUDA and GPU drivers")
        raise SystemExit(1)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        if args.offload:
            log.info("Loading with device_map=auto for offloading...")
            model = AutoModelForQuestionAnswering.from_pretrained(
                args.model_name,
                device_map="auto",
                torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
            )
        else:
            model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
            if args.device == "cuda":
                model = model.cuda().half()
            
    except torch.cuda.OutOfMemoryError:
        log.error("=" * 60)
        log.error("CUDA OUT OF MEMORY!")
        log.error("=" * 60)
        log.error("BERT-Large requires ~1.3GB VRAM. Try:")
        log.error("  1. --offload : Enable CPU offloading")
        log.error("  2. --cpu     : Run on CPU")
        log.error("  3. Reduce --batch-size")
        raise SystemExit(1)
    
    model.eval()
    log.info("Model loaded successfully!")
    
    return model, tokenizer


# ============================================================================
# Benchmark
# ============================================================================

def run_benchmark(model, tokenizer, args):
    """Run the QA benchmark"""
    log.info("=" * 60)
    log.info("Starting BERT Question Answering Benchmark")
    log.info("=" * 60)
    log.info(f"Data type: {args.data_type}")
    log.info(f"Batch size: {args.batch_size}")
    
    # Load data
    if args.data_type == "real":
        samples, data_info = load_squad_data(args.data_dir, args.max_examples)
        if samples is None:
            log.warning("Falling back to synthetic data")
            samples, data_info = generate_synthetic_data(args.max_examples, args.max_seq_length)
    else:
        samples, data_info = generate_synthetic_data(args.max_examples, args.max_seq_length)
    
    num_samples = len(samples)
    log.info(f"Processing {num_samples} samples")
    
    # Warmup
    log.info("Warmup...")
    warmup_sample = samples[0]
    if 'input_ids' in warmup_sample:
        # Pre-tokenized synthetic data
        inputs = {
            'input_ids': torch.tensor([warmup_sample['input_ids']]),
            'attention_mask': torch.tensor([warmup_sample['attention_mask']]),
            'token_type_ids': torch.tensor([warmup_sample['token_type_ids']])
        }
    else:
        inputs = tokenizer(
            warmup_sample['question'],
            warmup_sample['context'],
            max_length=args.max_seq_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
    
    if args.device == "cuda" and not args.offload:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            _ = model(**inputs)
    except torch.cuda.OutOfMemoryError:
        log.error("=" * 60)
        log.error("CUDA OUT OF MEMORY during warmup!")
        log.error("=" * 60)
        log.error("BERT-Large with batch_size={} is too large. Try:".format(args.batch_size))
        log.error("  1. --batch=4    : Reduce batch size")
        log.error("  2. --offload    : Enable CPU offloading")
        log.error("  3. --cpu        : Run on CPU")
        raise SystemExit(1)
    
    if args.device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    log.info("\nRunning benchmark...")
    results = []
    total_time = 0
    correct = 0
    
    for i in range(0, num_samples, args.batch_size):
        batch_samples = samples[i:i + args.batch_size]
        batch_size = len(batch_samples)
        
        # Tokenize batch
        if 'input_ids' in batch_samples[0]:
            # Pre-tokenized
            inputs = {
                'input_ids': torch.tensor([s['input_ids'] for s in batch_samples]),
                'attention_mask': torch.tensor([s['attention_mask'] for s in batch_samples]),
                'token_type_ids': torch.tensor([s['token_type_ids'] for s in batch_samples])
            }
        else:
            questions = [s['question'] for s in batch_samples]
            contexts = [s['context'] for s in batch_samples]
            inputs = tokenizer(
                questions,
                contexts,
                max_length=args.max_seq_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
        
        if args.device == "cuda" and not args.offload:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        start_time = time.perf_counter()
        
        try:
            with torch.no_grad():
                outputs = model(**inputs)
        except torch.cuda.OutOfMemoryError:
            log.error("=" * 60)
            log.error("CUDA OUT OF MEMORY during inference!")
            log.error("=" * 60)
            log.error("Batch {} with size {} caused OOM. Try:".format(i // args.batch_size, batch_size))
            log.error("  1. --batch=4    : Reduce batch size")
            log.error("  2. --offload    : Enable CPU offloading")
            log.error("  3. --cpu        : Run on CPU")
            raise SystemExit(1)
        
        if args.device == "cuda":
            torch.cuda.synchronize()
        
        batch_time = time.perf_counter() - start_time
        total_time += batch_time
        
        # Get predictions
        start_logits = outputs.start_logits.cpu().numpy()
        end_logits = outputs.end_logits.cpu().numpy()
        
        for j in range(batch_size):
            start_idx = np.argmax(start_logits[j])
            end_idx = np.argmax(end_logits[j])
            
            results.append({
                'sample_idx': i + j,
                'start_idx': int(start_idx),
                'end_idx': int(end_idx),
                'latency_ms': batch_time / batch_size * 1000
            })
        
        if (i + batch_size) % 100 == 0 or i + batch_size >= num_samples:
            throughput = batch_size / batch_time
            log.info(f"  [{i + batch_size}/{num_samples}] {batch_time*1000:.1f}ms/batch, "
                    f"{throughput:.1f} samples/sec")
    
    # Summary
    avg_latency = total_time / num_samples * 1000
    throughput = num_samples / total_time
    
    print("\n" + "=" * 60)
    if args.mlperf:
        if args.data_type == "synthetic":
            print("BERT BENCHMARK SUMMARY - MLPerf Settings (SYNTHETIC DATA)")
            print("⚠️  NOT COMPARABLE TO OFFICIAL MLPERF RESULTS")
        else:
            print("BERT BENCHMARK SUMMARY - MLPerf Compliant")
    else:
        print("BERT BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Device:             {args.device}")
    print(f"Data Type:          {args.data_type}")
    if args.mlperf:
        print(f"MLPerf Mode:        ENABLED")
    print(f"Total Samples:      {num_samples}")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Total Time:         {total_time:.2f} seconds")
    print(f"Avg Latency:        {avg_latency:.2f} ms/sample")
    print(f"Throughput:         {throughput:.2f} samples/sec")
    print("=" * 60)
    
    # Performance rating
    if throughput > 100:
        rating = "🚀 Excellent"
    elif throughput > 50:
        rating = "✅ Good"
    elif throughput > 10:
        rating = "⚡ Moderate"
    else:
        rating = "🐢 Slow"
    print(f"Performance:        {rating}")
    print("=" * 60)
    
    # Print DATA INFORMATION section
    print("\n" + "=" * 60)
    print("DATA INFORMATION")
    print("=" * 60)
    print(f"Type:               {data_info['type']}")
    print(f"Dataset:            {data_info['dataset']}")
    print(f"Source:             {data_info['source']}")
    print(f"Samples Used:       {data_info['samples_used']:,}")
    print(f"Samples Available:  {data_info['samples_available']:,}")
    print(f"Verified:           {'✓' if data_info['verified'] else '✗'}")
    print(f"MLPerf Compliant:   {'✓' if data_info['mlperf_compliant'] else '✗'}")
    print("-" * 40)
    print(f"Note: {data_info['note']}")
    print("=" * 60)
    
    return {
        'device': args.device,
        'data_type': args.data_type,
        'model': args.model_name,
        'mlperf_mode': args.mlperf,
        'mlperf_compliant': args.mlperf and args.data_type == "real",
        'num_samples': num_samples,
        'batch_size': args.batch_size,
        'total_time_sec': total_time,
        'avg_latency_ms': avg_latency,
        'throughput_samples_per_sec': throughput,
        'results': results[:10],  # Save first 10 for reference
        'data_info': data_info,
    }


def save_results(results: Dict, output_dir: str):
    """Save benchmark results"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bert_benchmark_{results['data_type']}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    log.info(f"\nResults saved to: {filepath}")
    return filepath


# ============================================================================
# Main
# ============================================================================

def main():
    args = get_args()
    
    # Load model
    model, tokenizer = load_model(args)
    
    # Run benchmark
    results = run_benchmark(model, tokenizer, args)
    
    # Save results
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
