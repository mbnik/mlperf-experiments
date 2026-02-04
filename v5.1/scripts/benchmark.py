#!/usr/bin/env python3
"""
MLPerf Benchmark Runner - Self-Contained Unified Entry Point

Central entry point for running all MLPerf benchmarks with LoadGen.
This file is self-contained and does not call external run_*_benchmark.py scripts.

Author: Mehdi Nik
Created: Jan 2026

Directory Structure:
    data/{benchmark}/{dataset_name}/
        - metadata.json (created by data_prepare.py)
        - data files...

Usage:
    python benchmark.py --benchmark bert --dataset squad --mlperf-quick
    python benchmark.py --benchmark resnet50 --dataset imagenet --mlperf-quick
    python benchmark.py --list
"""

import argparse
import array
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# MLPerf LoadGen
try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    print("Warning: mlperf_loadgen not installed. Install with: pip install mlperf_loadgen")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

BENCHMARKS = {
    'bert': {
        'name': 'BERT Question Answering',
        'task': 'question-answering',
        'default_dataset': 'squad',
        'model_name': 'bert-large-uncased-whole-word-masking-finetuned-squad',
        'performance_sample_count': 10833,
    },
    'resnet50': {
        'name': 'ResNet50 Image Classification',
        'task': 'image-classification',
        'default_dataset': 'imagenet',
        'model_name': 'torchvision.resnet50',
        'performance_sample_count': 1024,
    },
    'retinanet': {
        'name': 'RetinaNet Object Detection',
        'task': 'object-detection',
        'default_dataset': 'openimages',
        'model_name': 'retinanet_resnet50_fpn_v2',
        'performance_sample_count': 64,
    },
    '3dunet': {
        'name': '3D-UNet Medical Segmentation',
        'task': 'medical-segmentation',
        'default_dataset': 'kits19',
        'model_name': '3d-unet',
        'performance_sample_count': 16,
    },
    'whisper': {
        'name': 'Whisper Speech Recognition',
        'task': 'speech-recognition',
        'default_dataset': 'librispeech',
        'model_name': 'openai/whisper-large-v3',
        'performance_sample_count': 2513,
    },
    'sdxl': {
        'name': 'Stable Diffusion XL',
        'task': 'text-to-image',
        'default_dataset': 'coco-2014',
        'model_name': 'stabilityai/stable-diffusion-xl-base-1.0',
        'performance_sample_count': 5000,
    },
    'gptj': {
        'name': 'GPT-J Text Summarization',
        'task': 'text-summarization',
        'default_dataset': 'cnn-dailymail',
        'model_name': 'EleutherAI/gpt-j-6B',
        'performance_sample_count': 13368,
    },
    'llama': {
        'name': 'Llama2 Text Generation',
        'task': 'text-generation',
        'default_dataset': 'openorca',
        'model_name': 'meta-llama/Llama-2-70b-chat-hf',
        'performance_sample_count': 24576,
    },
    'mixtral': {
        'name': 'Mixtral-8x7B Text Generation',
        'task': 'text-generation',
        'default_dataset': 'mixtral-15k',
        'model_name': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'performance_sample_count': 15000,
    },
    'dlrm': {
        'name': 'DLRM Recommendation',
        'task': 'recommendation',
        'default_dataset': 'criteo',
        'model_name': 'dlrm',
        'performance_sample_count': 204800,
    },
}

SCENARIO_MAP = {
    'Offline': lg.TestScenario.Offline if LOADGEN_AVAILABLE else None,
    'SingleStream': lg.TestScenario.SingleStream if LOADGEN_AVAILABLE else None,
    'Server': lg.TestScenario.Server if LOADGEN_AVAILABLE else None,
    'MultiStream': lg.TestScenario.MultiStream if LOADGEN_AVAILABLE else None,
}


# ============================================================================
# Utilities
# ============================================================================

def get_project_dir() -> Path:
    """Get project directory."""
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    """Get data directory."""
    return get_project_dir() / 'data'


def load_metadata(data_dir: Path) -> Optional[Dict]:
    """Load metadata.json from a dataset directory."""
    metadata_file = data_dir / 'metadata.json'
    if metadata_file.exists():
        with open(metadata_file) as f:
            return json.load(f)
    return None


def discover_datasets(benchmark: str) -> Dict[str, Dict]:
    """Discover all datasets for a benchmark."""
    benchmark_dir = get_data_dir() / benchmark
    datasets = {}
    
    if not benchmark_dir.exists():
        return datasets
    
    for dataset_dir in benchmark_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        metadata = load_metadata(dataset_dir)
        
        if metadata:
            datasets[dataset_name] = {
                'path': str(dataset_dir),
                'metadata': metadata,
            }
        else:
            datasets[dataset_name] = {
                'path': str(dataset_dir),
                'metadata': {'type': 'unknown', 'samples': 0, 'mlperf_compliant': False},
            }
    
    return datasets


def list_all_datasets(verbose: bool = True) -> Dict[str, Dict]:
    """List all datasets for all benchmarks."""
    all_datasets = {}
    
    if verbose:
        print("\n" + "=" * 80)
        print("AVAILABLE DATASETS")
        print("=" * 80)
        print(f"  {'Benchmark':<12} {'Dataset':<20} {'Type':<12} {'Samples':>10} {'MLPerf':>8}")
        print("-" * 80)
    
    for benchmark in BENCHMARKS:
        datasets = discover_datasets(benchmark)
        all_datasets[benchmark] = datasets
        
        if verbose:
            if not datasets:
                print(f"  {benchmark:<12} {'(no datasets)':<20} {'---':<12} {'---':>10} {'---':>8}")
            else:
                for name, info in datasets.items():
                    meta = info.get('metadata', {})
                    dtype = meta.get('type', 'unknown')
                    samples = meta.get('samples', 0)
                    mlperf = meta.get('mlperf_compliant', False)
                    samples_str = f"{samples:,}" if samples else "---"
                    mlperf_str = "✓" if mlperf else "✗"
                    print(f"  {benchmark:<12} {name:<20} {dtype:<12} {samples_str:>10} {mlperf_str:>8}")
    
    if verbose:
        print("=" * 80)
        total_benchmarks = len(BENCHMARKS)
        benchmarks_with_data = sum(1 for d in all_datasets.values() if d)
        total_datasets = sum(len(d) for d in all_datasets.values())
        mlperf_datasets = sum(
            1 for d in all_datasets.values()
            for info in d.values()
            if info.get('metadata', {}).get('mlperf_compliant')
        )
        print(f"\nSummary: {benchmarks_with_data}/{total_benchmarks} benchmarks have data")
        print(f"         {total_datasets} total datasets, {mlperf_datasets} MLPerf compliant")
    
    return all_datasets


# ============================================================================
# BERT Benchmark
# ============================================================================

def run_bert(dataset_path: str, args: argparse.Namespace, metadata: Optional[Dict] = None) -> int:
    """Run BERT Question Answering benchmark with LoadGen."""
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    
    data_path = Path(dataset_path)
    device = 'cuda' if args.device == 'gpu' else args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    scenario = args.scenario if args.scenario else 'Offline'
    mode = args.mode if args.mode else 'performance'
    batch_size = args.batch_size if args.batch_size else 8
    
    # Timing settings
    if args.mlperf_quick:
        min_duration_ms = 60000
    elif args.min_duration:
        min_duration_ms = args.min_duration
    else:
        min_duration_ms = 600000
    
    # Get number of samples: user override > quick default (50) > full default (100)
    if args.max_samples:
        num_samples = args.max_samples
    elif args.mlperf_quick:
        num_samples = 50
    else:
        num_samples = 100
    
    # Calculate target_qps from samples and duration if not specified
    # LoadGen generates: samples_per_query = target_qps × min_duration × 1.1
    if args.target_qps:
        target_qps = args.target_qps
    else:
        min_duration_sec = min_duration_ms / 1000
        target_qps = num_samples / (min_duration_sec * 1.1)
    
    output_dir = Path(args.output_dir) if args.output_dir else (get_project_dir() / 'results' / 'bert')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Display configuration
    _print_config('BERT QUESTION ANSWERING', 'bert', dataset_path, device, 
                  scenario, mode, min_duration_ms, target_qps, batch_size, output_dir, 
                  metadata, args)
    
    # Load model and tokenizer
    print("\n" + "=" * 70)
    print("LOADING MODEL AND DATA")
    print("=" * 70)
    
    model_name = BENCHMARKS['bert']['model_name']
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading model: {model_name}")
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.to(device)
    model.eval()
    if device == 'cuda':
        model = model.half()
    print("Model loaded and ready")
    
    # Load data
    squad_file = data_path / 'dev-v1.1.json'
    if squad_file.exists():
        print(f"Loading SQuAD data from {squad_file}")
        with open(squad_file) as f:
            squad_data = json.load(f)
        
        samples = []
        for article in squad_data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    samples.append({
                        'question': qa['question'],
                        'context': context,
                        'id': qa['id'],
                    })
                    if args.max_samples and len(samples) >= args.max_samples:
                        break
                if args.max_samples and len(samples) >= args.max_samples:
                    break
            if args.max_samples and len(samples) >= args.max_samples:
                break
    else:
        print(f"Error: SQuAD data not found at {squad_file}")
        return 1
    
    num_samples = len(samples)
    print(f"Loaded {num_samples} QA samples")
    
    # Pre-tokenize samples
    print("Pre-tokenizing samples...")
    features = []
    for sample in samples:
        encoding = tokenizer(
            sample['question'],
            sample['context'],
            max_length=384,
            truncation=True,
            padding='max_length',
            return_tensors='np'  # Use numpy to avoid HuggingFace warnings
        )
        features.append({
            'input_ids': torch.from_numpy(encoding['input_ids'].squeeze()),
            'attention_mask': torch.from_numpy(encoding['attention_mask'].squeeze()),
            'token_type_ids': torch.from_numpy(encoding['token_type_ids'].squeeze()),
        })
    print("Tokenization complete")
    
    # Metrics tracking
    total_inference_time = [0.0]  # Pure model inference time
    total_samples_processed = [0]
    query_start_time = [None]  # Wall-clock start time
    query_end_time = [None]    # Wall-clock end time
    predictions = []  # Store predictions for F1 calculation
    
    # F1 score calculation for QA
    def calculate_f1(prediction: str, ground_truth: str) -> float:
        """Calculate F1 score between prediction and ground truth."""
        pred_tokens = prediction.lower().split()
        truth_tokens = ground_truth.lower().split()
        
        if not pred_tokens or not truth_tokens:
            return 0.0
        
        common = set(pred_tokens) & set(truth_tokens)
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        
        return 2 * precision * recall / (precision + recall)
    
    # LoadGen callbacks
    perf_count = min(num_samples, BENCHMARKS['bert']['performance_sample_count'])
    
    def issue_queries(query_samples):
        """Process queries from LoadGen."""
        import time as time_module
        
        # Track wall-clock time
        if query_start_time[0] is None:
            query_start_time[0] = time_module.perf_counter()
        
        if scenario == 'Offline' and len(query_samples) > 1:
            # Batch processing for Offline
            for i in range(0, len(query_samples), batch_size):
                batch_samples = query_samples[i:i + batch_size]
                batch_indices = [s.index for s in batch_samples]
                
                input_ids = torch.stack([features[idx]['input_ids'] for idx in batch_indices]).to(device)
                attention_mask = torch.stack([features[idx]['attention_mask'] for idx in batch_indices]).to(device)
                token_type_ids = torch.stack([features[idx]['token_type_ids'] for idx in batch_indices]).to(device)
                
                start_time = time_module.perf_counter()
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                elapsed = time_module.perf_counter() - start_time
                total_inference_time[0] += elapsed
                total_samples_processed[0] += len(batch_samples)
                
                for j, sample in enumerate(batch_samples):
                    # Get predicted answer span
                    start_idx = outputs.start_logits[j].argmax().item()
                    end_idx = outputs.end_logits[j].argmax().item()
                    input_ids_sample = features[batch_indices[j]]['input_ids']
                    if end_idx >= start_idx:
                        pred_tokens = input_ids_sample[start_idx:end_idx+1]
                        pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                    else:
                        pred_text = ""
                    predictions.append((batch_indices[j], pred_text))
                    
                    output = np.stack([
                        outputs.start_logits[j].cpu().numpy(),
                        outputs.end_logits[j].cpu().numpy()
                    ], axis=-1).astype(np.float32)
                    response = lg.QuerySampleResponse(sample.id, output.ctypes.data, output.nbytes)
                    lg.QuerySamplesComplete([response])
        else:
            # Single sample processing
            for sample in query_samples:
                idx = sample.index
                input_ids = features[idx]['input_ids'].unsqueeze(0).to(device)
                attention_mask = features[idx]['attention_mask'].unsqueeze(0).to(device)
                token_type_ids = features[idx]['token_type_ids'].unsqueeze(0).to(device)
                
                start_time = time_module.perf_counter()
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                elapsed = time_module.perf_counter() - start_time
                total_inference_time[0] += elapsed
                total_samples_processed[0] += 1
                
                # Get predicted answer span
                start_idx = outputs.start_logits[0].argmax().item()
                end_idx = outputs.end_logits[0].argmax().item()
                if end_idx >= start_idx:
                    pred_tokens = features[idx]['input_ids'][start_idx:end_idx+1]
                    pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                else:
                    pred_text = ""
                predictions.append((idx, pred_text))
                
                output = np.stack([
                    outputs.start_logits[0].cpu().numpy(),
                    outputs.end_logits[0].cpu().numpy()
                ], axis=-1).astype(np.float32)
                response = lg.QuerySampleResponse(sample.id, output.ctypes.data, output.nbytes)
                lg.QuerySamplesComplete([response])
        
        # Update wall-clock end time after all queries processed
        query_end_time[0] = time_module.perf_counter()
    
    def flush_queries():
        pass
    
    def load_samples(sample_list):
        pass
    
    def unload_samples(sample_list):
        pass
    
    # Create QSL and SUT
    qsl = lg.ConstructQSL(num_samples, perf_count, load_samples, unload_samples)
    sut = lg.ConstructSUT(issue_queries, flush_queries)
    
    # Configure and run LoadGen
    result = _run_loadgen_test(sut, qsl, scenario, mode, min_duration_ms, target_qps, output_dir, num_samples, args)
    
    # Print detailed BERT metrics after LoadGen completes
    if total_samples_processed[0] > 0:
        samples_completed = total_samples_processed[0]
        inference_time = total_inference_time[0]
        
        # Wall-clock time matches LoadGen's measurement
        wall_clock_time = query_end_time[0] - query_start_time[0] if query_end_time[0] > 0 else inference_time
        
        # Throughput based on wall-clock time (matches LoadGen)
        samples_per_sec = samples_completed / wall_clock_time if wall_clock_time > 0 else 0
        
        # Pure inference throughput (excludes overhead)
        pure_inference_throughput = samples_completed / inference_time if inference_time > 0 else 0
        
        # Average latency per sample
        avg_latency = wall_clock_time / samples_completed
        
        print("\n" + "=" * 70)
        print("BERT QA BENCHMARK DETAILED METRICS")
        print("=" * 70)
        print(f"  Samples Processed:    {samples_completed}")
        print(f"  Wall-Clock Time:      {wall_clock_time:.2f} seconds")
        print(f"  Pure Inference Time:  {inference_time:.2f} seconds")
        print(f"  Avg Latency:          {avg_latency*1000:.2f} ms/sample")
        print(f"  Throughput:           {samples_per_sec:.2f} samples/sec")
        print(f"  (Pure GPU):           {pure_inference_throughput:.2f} samples/sec")
        print("=" * 70)
        
        # Performance rating based on wall-clock throughput
        if samples_per_sec > 100:
            print("  Performance:          🚀 Excellent")
        elif samples_per_sec > 50:
            print("  Performance:          ✅ Good")
        elif samples_per_sec > 10:
            print("  Performance:          ⚠️  Moderate")
        else:
            print("  Performance:          🐢 Slow")
        print("=" * 70)
    
    return result


# ============================================================================
# ResNet50 Benchmark
# ============================================================================

def run_resnet50(dataset_path: str, args: argparse.Namespace, metadata: Optional[Dict] = None) -> int:
    """Run ResNet50 Image Classification benchmark with LoadGen."""
    from torchvision import models, transforms
    from PIL import Image
    
    data_path = Path(dataset_path)
    device = 'cuda' if args.device == 'gpu' else args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    scenario = args.scenario if args.scenario else 'Offline'
    mode = args.mode if args.mode else 'performance'
    batch_size = args.batch_size if args.batch_size else 32
    
    if args.mlperf_quick:
        min_duration_ms = 60000
    elif args.min_duration:
        min_duration_ms = args.min_duration
    else:
        min_duration_ms = 600000
    
    # Get number of samples: user override > quick default (50) > full default (100)
    if args.max_samples:
        num_samples = args.max_samples
    elif args.mlperf_quick:
        num_samples = 50
    else:
        num_samples = 100
    
    # Calculate target_qps from samples and duration if not specified
    if args.target_qps:
        target_qps = args.target_qps
    else:
        min_duration_sec = min_duration_ms / 1000
        target_qps = num_samples / (min_duration_sec * 1.1)
    
    output_dir = Path(args.output_dir) if args.output_dir else (get_project_dir() / 'results' / 'resnet50')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    _print_config('RESNET50 IMAGE CLASSIFICATION', 'resnet50', dataset_path, device,
                  scenario, mode, min_duration_ms, target_qps, batch_size, output_dir,
                  metadata, args)
    
    print("\n" + "=" * 70)
    print("LOADING MODEL AND DATA")
    print("=" * 70)
    
    # Load model
    print("Loading ResNet50 model...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.to(device)
    model.eval()
    if device == 'cuda':
        model = model.half()
    print("Model loaded and ready")
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load images - ImageNet is organized as val/<class_id>/<images>
    images = []
    val_dir = data_path / 'val'
    if not val_dir.exists():
        val_dir = data_path  # Try root directory
    
    # Get all class directories
    class_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    
    if class_dirs:
        # ImageNet structure: val/<class_id>/<images>
        image_files = []
        for class_dir in class_dirs:
            image_files.extend(list(class_dir.glob('*.JPEG')))
            image_files.extend(list(class_dir.glob('*.jpg')))
            image_files.extend(list(class_dir.glob('*.png')))
    else:
        # Flat structure: images directly in directory
        image_files = list(val_dir.glob('*.JPEG')) + list(val_dir.glob('*.jpg')) + list(val_dir.glob('*.png'))
    
    if args.max_samples:
        image_files = image_files[:args.max_samples]
    
    print(f"Loading {len(image_files)} images...")
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img)
            images.append(img_tensor)
        except Exception as e:
            log.warning(f"Failed to load {img_path}: {e}")
    
    num_samples = len(images)
    print(f"Loaded {num_samples} images")
    
    if num_samples == 0:
        print("Error: No images found")
        return 1
    
    # Stack all images into a tensor
    images_tensor = torch.stack(images)
    
    # Metrics tracking
    total_inference_time = [0.0]
    total_samples_processed = [0]
    query_start_time = [None]
    query_end_time = [None]
    
    perf_count = min(num_samples, BENCHMARKS['resnet50']['performance_sample_count'])
    
    def issue_queries(query_samples):
        import time as time_module
        
        # Track wall-clock time
        if query_start_time[0] is None:
            query_start_time[0] = time_module.perf_counter()
        
        indices = [s.index for s in query_samples]
        response_ids = [s.id for s in query_samples]
        
        all_predictions = []
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = torch.stack([images_tensor[idx] for idx in batch_indices])
            
            if device == 'cuda':
                batch = batch.cuda().half()
            
            start_time = time_module.perf_counter()
            with torch.no_grad():
                outputs = model(batch)
                predictions = torch.argmax(outputs, dim=1)
            elapsed = time_module.perf_counter() - start_time
            total_inference_time[0] += elapsed
            total_samples_processed[0] += len(batch_indices)
            
            all_predictions.extend(predictions.cpu().tolist())
        
        responses = []
        for pred, response_id in zip(all_predictions, response_ids):
            response_data = array.array('i', [pred])
            bi = response_data.buffer_info()
            responses.append(lg.QuerySampleResponse(response_id, bi[0], bi[1] * response_data.itemsize))
        
        lg.QuerySamplesComplete(responses)
        
        # Update wall-clock end time
        query_end_time[0] = time_module.perf_counter()
    
    def flush_queries():
        pass
    
    def load_samples(sample_list):
        pass
    
    def unload_samples(sample_list):
        pass
    
    qsl = lg.ConstructQSL(num_samples, perf_count, load_samples, unload_samples)
    sut = lg.ConstructSUT(issue_queries, flush_queries)
    
    result = _run_loadgen_test(sut, qsl, scenario, mode, min_duration_ms, target_qps, output_dir, num_samples, args)
    
    # Print detailed ResNet50 metrics after LoadGen completes
    if total_samples_processed[0] > 0:
        samples_completed = total_samples_processed[0]
        inference_time = total_inference_time[0]
        wall_clock_time = query_end_time[0] - query_start_time[0] if query_end_time[0] else inference_time
        
        samples_per_sec = samples_completed / wall_clock_time if wall_clock_time > 0 else 0
        pure_samples_per_sec = samples_completed / inference_time if inference_time > 0 else 0
        avg_latency = wall_clock_time / samples_completed * 1000  # ms
        
        print("\n" + "=" * 70)
        print("RESNET50 BENCHMARK DETAILED METRICS")
        print("=" * 70)
        print(f"  Samples Processed:    {samples_completed}")
        print(f"  Wall-Clock Time:      {wall_clock_time:.2f} seconds")
        print(f"  Pure Inference Time:  {inference_time:.2f} seconds")
        print(f"  Avg Latency:          {avg_latency:.2f} ms/sample")
        print(f"  Throughput:           {samples_per_sec:.2f} samples/sec")
        print(f"  (Pure GPU):           {pure_samples_per_sec:.2f} samples/sec")
        print("=" * 70)
        
        if samples_per_sec > 500:
            print("  Performance:          🚀 Excellent")
        elif samples_per_sec > 200:
            print("  Performance:          ✅ Good")
        elif samples_per_sec > 50:
            print("  Performance:          ⚠️  Moderate")
        else:
            print("  Performance:          🐢 Slow")
        print("=" * 70)
    
    return result


# ============================================================================
# RetinaNet Benchmark
# ============================================================================

def run_retinanet(dataset_path: str, args: argparse.Namespace, metadata: Optional[Dict] = None) -> int:
    """Run RetinaNet Object Detection benchmark with LoadGen."""
    from torchvision import models, transforms
    from PIL import Image
    
    data_path = Path(dataset_path)
    device = 'cuda' if args.device == 'gpu' else args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    scenario = args.scenario if args.scenario else 'Offline'
    mode = args.mode if args.mode else 'performance'
    batch_size = args.batch_size if args.batch_size else 1
    
    if args.mlperf_quick:
        min_duration_ms = 60000
    elif args.min_duration:
        min_duration_ms = args.min_duration
    else:
        min_duration_ms = 600000
    
    # Get number of samples: user override > quick default (50) > full default (100)
    if args.max_samples:
        num_samples = args.max_samples
    elif args.mlperf_quick:
        num_samples = 50
    else:
        num_samples = 100
    
    # Calculate target_qps from samples and duration if not specified
    if args.target_qps:
        target_qps = args.target_qps
    else:
        min_duration_sec = min_duration_ms / 1000
        target_qps = num_samples / (min_duration_sec * 1.1)
    
    output_dir = Path(args.output_dir) if args.output_dir else (get_project_dir() / 'results' / 'retinanet')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    _print_config('RETINANET OBJECT DETECTION', 'retinanet', dataset_path, device,
                  scenario, mode, min_duration_ms, target_qps, batch_size, output_dir,
                  metadata, args)
    
    print("\n" + "=" * 70)
    print("LOADING MODEL AND DATA")
    print("=" * 70)
    
    # Load model
    print("Loading RetinaNet model...")
    model = models.detection.retinanet_resnet50_fpn_v2(weights=models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
    model.to(device)
    model.eval()
    print("Model loaded and ready")
    
    # Load images - check multiple possible locations
    images = []
    image_dir = None
    
    # Check for OpenImages structure: validation/data/
    if (data_path / 'validation' / 'data').exists():
        image_dir = data_path / 'validation' / 'data'
    # Check for synthetic structure: images/
    elif (data_path / 'images').exists():
        image_dir = data_path / 'images'
    # Fall back to root directory
    else:
        image_dir = data_path
    
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')) + list(image_dir.glob('*.JPEG'))
    
    if args.max_samples:
        image_files = image_files[:args.max_samples]
    
    print(f"Loading {len(image_files)} images...")
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img)
            images.append(img_tensor)
        except Exception as e:
            log.warning(f"Failed to load {img_path}: {e}")
    
    num_samples = len(images)
    print(f"Loaded {num_samples} images")
    
    if num_samples == 0:
        print("Error: No images found")
        return 1
    
    # Metrics tracking
    total_inference_time = [0.0]
    total_samples_processed = [0]
    total_detections = [0]
    query_start_time = [None]
    query_end_time = [None]
    
    perf_count = min(num_samples, BENCHMARKS['retinanet']['performance_sample_count'])
    
    def issue_queries(query_samples):
        import time as time_module
        responses = []
        
        # Track wall-clock time
        if query_start_time[0] is None:
            query_start_time[0] = time_module.perf_counter()
        
        for sample in query_samples:
            image = images[sample.index]
            
            if device == 'cuda':
                image = image.cuda()
            
            start_time = time_module.perf_counter()
            with torch.no_grad():
                outputs = model([image])
            elapsed = time_module.perf_counter() - start_time
            total_inference_time[0] += elapsed
            total_samples_processed[0] += 1
            
            num_detections = len(outputs[0]['boxes'])
            total_detections[0] += num_detections
            response_data = np.array([num_detections], dtype=np.int32)
            response_array = array.array('B', response_data.tobytes())
            bi = response_array.buffer_info()
            responses.append(lg.QuerySampleResponse(sample.id, bi[0], bi[1]))
        
        lg.QuerySamplesComplete(responses)
        
        # Update wall-clock end time
        query_end_time[0] = time_module.perf_counter()
    
    def flush_queries():
        pass
    
    def load_samples(sample_list):
        pass
    
    def unload_samples(sample_list):
        pass
    
    qsl = lg.ConstructQSL(num_samples, perf_count, load_samples, unload_samples)
    sut = lg.ConstructSUT(issue_queries, flush_queries)
    
    result = _run_loadgen_test(sut, qsl, scenario, mode, min_duration_ms, target_qps, output_dir, num_samples, args)
    
    # Print detailed RetinaNet metrics after LoadGen completes
    if total_samples_processed[0] > 0:
        samples_completed = total_samples_processed[0]
        inference_time = total_inference_time[0]
        wall_clock_time = query_end_time[0] - query_start_time[0] if query_end_time[0] else inference_time
        
        samples_per_sec = samples_completed / wall_clock_time if wall_clock_time > 0 else 0
        pure_samples_per_sec = samples_completed / inference_time if inference_time > 0 else 0
        avg_latency = wall_clock_time / samples_completed * 1000  # ms
        avg_detections = total_detections[0] / samples_completed
        
        print("\n" + "=" * 70)
        print("RETINANET BENCHMARK DETAILED METRICS")
        print("=" * 70)
        print(f"  Samples Processed:    {samples_completed}")
        print(f"  Total Detections:     {total_detections[0]}")
        print(f"  Avg Detections/Image: {avg_detections:.1f}")
        print(f"  Wall-Clock Time:      {wall_clock_time:.2f} seconds")
        print(f"  Pure Inference Time:  {inference_time:.2f} seconds")
        print(f"  Avg Latency:          {avg_latency:.2f} ms/sample")
        print(f"  Throughput:           {samples_per_sec:.2f} samples/sec")
        print(f"  (Pure GPU):           {pure_samples_per_sec:.2f} samples/sec")
        print("=" * 70)
        
        if samples_per_sec > 20:
            print("  Performance:          🚀 Excellent")
        elif samples_per_sec > 10:
            print("  Performance:          ✅ Good")
        elif samples_per_sec > 5:
            print("  Performance:          ⚠️  Moderate")
        else:
            print("  Performance:          🐢 Slow")
        print("=" * 70)
    
    return result


# ============================================================================
# 3D-UNet Benchmark
# ============================================================================

def run_3dunet(dataset_path: str, args: argparse.Namespace, metadata: Optional[Dict] = None) -> int:
    """Run 3D-UNet Medical Segmentation benchmark with LoadGen."""
    
    data_path = Path(dataset_path)
    device = 'cuda' if args.device == 'gpu' else args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    scenario = args.scenario if args.scenario else 'Offline'
    mode = args.mode if args.mode else 'performance'
    batch_size = args.batch_size if args.batch_size else 1
    
    if args.mlperf_quick:
        min_duration_ms = 60000
    elif args.min_duration:
        min_duration_ms = args.min_duration
    else:
        min_duration_ms = 600000
    
    # Get number of samples: user override > quick default (50) > full default (100)
    if args.max_samples:
        num_samples = args.max_samples
    elif args.mlperf_quick:
        num_samples = 50
    else:
        num_samples = 100
    
    # Calculate target_qps from samples and duration if not specified
    if args.target_qps:
        target_qps = args.target_qps
    else:
        min_duration_sec = min_duration_ms / 1000
        target_qps = num_samples / (min_duration_sec * 1.1)
    
    output_dir = Path(args.output_dir) if args.output_dir else (get_project_dir() / 'results' / '3dunet')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    _print_config('3D-UNET MEDICAL SEGMENTATION', '3dunet', dataset_path, device,
                  scenario, mode, min_duration_ms, target_qps, batch_size, output_dir,
                  metadata, args)
    
    print("\n" + "=" * 70)
    print("LOADING MODEL AND DATA")
    print("=" * 70)
    
    # Load data
    volumes_file = data_path / 'volumes.npy'
    if not volumes_file.exists():
        print(f"Error: volumes.npy not found in {dataset_path}")
        return 1
    
    print(f"Loading volumes from {volumes_file}...")
    volumes = np.load(volumes_file)
    num_samples = len(volumes)
    print(f"Loaded {num_samples} volumes, shape: {volumes.shape}")
    
    if args.max_samples:
        num_samples = min(num_samples, args.max_samples)
    
    # Create simple 3D UNet model
    print("Creating 3D-UNet model...")
    model = _create_3dunet_model()
    model.to(device)
    model.eval()
    print("Model loaded and ready")
    
    # Metrics tracking
    total_inference_time = [0.0]
    total_samples_processed = [0]
    query_start_time = [None]
    query_end_time = [None]
    
    perf_count = min(num_samples, BENCHMARKS['3dunet']['performance_sample_count'])
    
    def issue_queries(query_samples):
        import time as time_module
        
        # Track wall-clock time
        if query_start_time[0] is None:
            query_start_time[0] = time_module.perf_counter()
        
        for sample in query_samples:
            idx = sample.index % num_samples
            volume = torch.from_numpy(volumes[idx:idx+1]).float().to(device)
            
            start_time = time_module.perf_counter()
            with torch.no_grad():
                output = model(volume)
            elapsed = time_module.perf_counter() - start_time
            total_inference_time[0] += elapsed
            total_samples_processed[0] += 1
            
            response_array = np.array([idx], dtype=np.int32)
            response = lg.QuerySampleResponse(sample.id, response_array.ctypes.data, response_array.nbytes)
            lg.QuerySamplesComplete([response])
        
        # Update wall-clock end time
        query_end_time[0] = time_module.perf_counter()
    
    def flush_queries():
        pass
    
    def load_samples(sample_list):
        pass
    
    def unload_samples(sample_list):
        pass
    
    qsl = lg.ConstructQSL(num_samples, perf_count, load_samples, unload_samples)
    sut = lg.ConstructSUT(issue_queries, flush_queries)
    
    result = _run_loadgen_test(sut, qsl, scenario, mode, min_duration_ms, target_qps, output_dir, num_samples, args)
    
    # Print detailed 3D-UNet metrics after LoadGen completes
    if total_samples_processed[0] > 0:
        samples_completed = total_samples_processed[0]
        inference_time = total_inference_time[0]
        wall_clock_time = query_end_time[0] - query_start_time[0] if query_end_time[0] else inference_time
        
        samples_per_sec = samples_completed / wall_clock_time if wall_clock_time > 0 else 0
        pure_samples_per_sec = samples_completed / inference_time if inference_time > 0 else 0
        avg_latency = wall_clock_time / samples_completed * 1000  # ms
        
        print("\n" + "=" * 70)
        print("3D-UNET BENCHMARK DETAILED METRICS")
        print("=" * 70)
        print(f"  Volumes Processed:    {samples_completed}")
        print(f"  Wall-Clock Time:      {wall_clock_time:.2f} seconds")
        print(f"  Pure Inference Time:  {inference_time:.2f} seconds")
        print(f"  Avg Latency:          {avg_latency:.2f} ms/volume")
        print(f"  Throughput:           {samples_per_sec:.2f} volumes/sec")
        print(f"  (Pure GPU):           {pure_samples_per_sec:.2f} volumes/sec")
        print("=" * 70)
        
        if samples_per_sec > 5:
            print("  Performance:          🚀 Excellent")
        elif samples_per_sec > 2:
            print("  Performance:          ✅ Good")
        elif samples_per_sec > 1:
            print("  Performance:          ⚠️  Moderate")
        else:
            print("  Performance:          🐢 Slow")
        print("=" * 70)
    
    return result


def _create_3dunet_model():
    """Create a simple 3D UNet model for benchmarking."""
    
    class Simple3DUNet(nn.Module):
        def __init__(self, in_channels=1, out_channels=3, features=32):
            super().__init__()
            
            self.enc1 = self._block(in_channels, features)
            self.pool1 = nn.MaxPool3d(2, 2)
            self.enc2 = self._block(features, features * 2)
            self.pool2 = nn.MaxPool3d(2, 2)
            
            self.bottleneck = self._block(features * 2, features * 4)
            
            self.up2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
            self.dec2 = self._block(features * 4, features * 2)
            self.up1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
            self.dec1 = self._block(features * 2, features)
            
            self.out = nn.Conv3d(features, out_channels, kernel_size=1)
        
        def _block(self, in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            )
        
        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            b = self.bottleneck(self.pool2(e2))
            d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out(d1)
    
    return Simple3DUNet()


# ============================================================================
# Whisper Benchmark
# ============================================================================

def run_whisper(dataset_path: str, args: argparse.Namespace, metadata: Optional[Dict] = None) -> int:
    """Run Whisper Speech Recognition benchmark with LoadGen."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    
    data_path = Path(dataset_path)
    device = 'cuda' if args.device == 'gpu' else args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    scenario = args.scenario if args.scenario else 'Offline'
    mode = args.mode if args.mode else 'performance'
    batch_size = args.batch_size if args.batch_size else 1
    
    if args.mlperf_quick:
        min_duration_ms = 60000
    elif args.min_duration:
        min_duration_ms = args.min_duration
    else:
        min_duration_ms = 600000
    
    # Get number of samples: user override > quick default (50) > full default (100)
    if args.max_samples:
        num_samples = args.max_samples
    elif args.mlperf_quick:
        num_samples = 50
    else:
        num_samples = 100
    
    # Calculate target_qps from samples and duration if not specified
    if args.target_qps:
        target_qps = args.target_qps
    else:
        min_duration_sec = min_duration_ms / 1000
        target_qps = num_samples / (min_duration_sec * 1.1)
    
    output_dir = Path(args.output_dir) if args.output_dir else (get_project_dir() / 'results' / 'whisper')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    _print_config('WHISPER SPEECH RECOGNITION', 'whisper', dataset_path, device,
                  scenario, mode, min_duration_ms, target_qps, batch_size, output_dir,
                  metadata, args)
    
    print("\n" + "=" * 70)
    print("LOADING MODEL AND DATA")
    print("=" * 70)
    
    # Load audio data
    audio_samples = None
    transcripts = None
    
    # Check for numpy format (synthetic)
    audio_file = data_path / "audio_samples.npy"
    manifest_file = data_path / "manifest.json"
    
    if audio_file.exists():
        print(f"  Loading audio from: {audio_file}")
        audio_samples = np.load(audio_file)
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
                transcripts = [m.get('text', '') for m in manifest]
        print(f"  Loaded {len(audio_samples)} audio samples")
    elif manifest_file.exists():
        # LibriSpeech format - manifest with file paths
        print(f"  Loading from manifest: {manifest_file}")
        with open(manifest_file) as f:
            manifest = json.load(f)
        
        import soundfile as sf
        audio_samples_list = []
        transcripts = []
        
        max_samples = args.max_samples if args.max_samples else len(manifest)
        for entry in manifest[:max_samples]:
            try:
                audio_path = entry.get('audio_path', entry.get('audio_filepath'))
                if audio_path:
                    audio, sr = sf.read(audio_path)
                    if sr != 16000:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    audio_samples_list.append(audio.astype(np.float32))
                    transcripts.append(entry.get('text', ''))
            except Exception as e:
                print(f"  Warning: Failed to load {audio_path}: {e}")
        
        audio_samples = audio_samples_list if audio_samples_list else None
        print(f"  Loaded {len(audio_samples) if audio_samples else 0} audio samples")
    else:
        print(f"  Error: No audio data found in {data_path}")
        return 1
    
    if audio_samples is None or len(audio_samples) == 0:
        print("  Error: No audio samples loaded")
        return 1
    
    # Convert to list if numpy array
    if isinstance(audio_samples, np.ndarray):
        audio_samples = [audio_samples[i] for i in range(len(audio_samples))]
    
    # Apply max_samples limit
    if args.max_samples and args.max_samples < len(audio_samples):
        audio_samples = audio_samples[:args.max_samples]
        if transcripts:
            transcripts = transcripts[:args.max_samples]
        print(f"  Limited to {len(audio_samples)} samples")
    
    num_samples = len(audio_samples)
    
    # Load model - bypass pipeline API to avoid deprecation warnings
    model_name = BENCHMARKS['whisper']['model_name']
    print(f"\n  Loading model: {model_name}")
    
    torch_dtype = torch.float16 if device == 'cuda' else torch.float32
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model = model.to(device)
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Get forced decoder IDs for English transcription
    # This avoids the "language detection" warning by explicitly setting language
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    print(f"  Model loaded on {device}")
    print(f"  Audio samples: {num_samples}")
    
    # Pre-process all audio samples to avoid repeated processing
    print("  Pre-processing audio features...")
    processed_features = []
    for i, audio in enumerate(audio_samples):
        # Process audio to get input_features (not 'inputs')
        inputs = processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt",
            return_attention_mask=True,  # Explicitly request attention mask
        )
        processed_features.append({
            'input_features': inputs.input_features.to(device, dtype=torch_dtype),
            'attention_mask': inputs.attention_mask.to(device) if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None else None,
        })
    print(f"  Pre-processed {len(processed_features)} samples")
    
    # Metrics tracking
    total_inference_time = [0.0]
    total_samples_processed = [0]
    total_tokens_generated = [0]
    transcriptions = []  # Store for WER calculation if needed
    query_start_time = [None]
    query_end_time = [None]
    
    # LoadGen callbacks
    perf_count = min(num_samples, BENCHMARKS['whisper']['performance_sample_count'])
    
    def issue_queries(query_samples):
        """Process queries from LoadGen."""
        import time as time_module
        
        # Track wall-clock time
        if query_start_time[0] is None:
            query_start_time[0] = time_module.perf_counter()
        
        for sample in query_samples:
            idx = sample.index
            features = processed_features[idx]
            
            # Run inference directly on model (bypassing pipeline)
            start_time = time_module.perf_counter()
            with torch.no_grad():
                # Generate with explicit parameters to avoid warnings
                generated_ids = model.generate(
                    input_features=features['input_features'],
                    attention_mask=features['attention_mask'],
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=256,
                    use_cache=True,
                )
            elapsed = time_module.perf_counter() - start_time
            total_inference_time[0] += elapsed
            total_samples_processed[0] += 1
            total_tokens_generated[0] += generated_ids.shape[1]
            
            # Decode transcription
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            transcriptions.append((idx, transcription))
            
            # Create response
            response_data = np.array([0], dtype=np.int64)
            response = lg.QuerySampleResponse(sample.id, response_data.ctypes.data, response_data.nbytes)
            lg.QuerySamplesComplete([response])
        
        # Update wall-clock end time
        query_end_time[0] = time_module.perf_counter()
    
    def flush_queries():
        pass
    
    def load_samples(sample_list):
        pass
    
    def unload_samples(sample_list):
        pass
    
    # Create QSL and SUT
    qsl = lg.ConstructQSL(num_samples, perf_count, load_samples, unload_samples)
    sut = lg.ConstructSUT(issue_queries, flush_queries)
    
    result = _run_loadgen_test(sut, qsl, scenario, mode, min_duration_ms, target_qps, output_dir, num_samples, args)
    
    # Print detailed Whisper metrics after LoadGen completes
    if total_samples_processed[0] > 0:
        samples_completed = total_samples_processed[0]
        inference_time = total_inference_time[0]
        wall_clock_time = query_end_time[0] - query_start_time[0] if query_end_time[0] else inference_time
        
        samples_per_sec = samples_completed / wall_clock_time if wall_clock_time > 0 else 0
        pure_samples_per_sec = samples_completed / inference_time if inference_time > 0 else 0
        avg_latency = wall_clock_time / samples_completed * 1000  # ms
        tokens_per_sec = total_tokens_generated[0] / wall_clock_time if wall_clock_time > 0 else 0
        
        print("\n" + "=" * 70)
        print("WHISPER BENCHMARK DETAILED METRICS")
        print("=" * 70)
        print(f"  Audio Samples:        {samples_completed}")
        print(f"  Total Tokens:         {total_tokens_generated[0]}")
        print(f"  Wall-Clock Time:      {wall_clock_time:.2f} seconds")
        print(f"  Pure Inference Time:  {inference_time:.2f} seconds")
        print(f"  Avg Latency:          {avg_latency:.2f} ms/sample")
        print(f"  Throughput:           {samples_per_sec:.2f} samples/sec")
        print(f"  Token Throughput:     {tokens_per_sec:.2f} tokens/sec")
        print(f"  (Pure GPU):           {pure_samples_per_sec:.2f} samples/sec")
        print("=" * 70)
        
        if samples_per_sec > 5:
            print("  Performance:          🚀 Excellent")
        elif samples_per_sec > 2:
            print("  Performance:          ✅ Good")
        elif samples_per_sec > 1:
            print("  Performance:          ⚠️  Moderate")
        else:
            print("  Performance:          🐢 Slow")
        print("=" * 70)
    
    return result


# ============================================================================
# SDXL Benchmark (placeholder)
# ============================================================================

def run_sdxl(dataset_path: str, args: argparse.Namespace, metadata: Optional[Dict] = None) -> int:
    """Run Stable Diffusion XL benchmark with LoadGen."""
    from diffusers import DiffusionPipeline
    
    data_path = Path(dataset_path)
    device = 'cuda' if args.device == 'gpu' else args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    scenario = args.scenario if args.scenario else 'Offline'
    mode = args.mode if args.mode else 'performance'
    batch_size = args.batch_size if args.batch_size else 1
    
    if args.mlperf_quick:
        min_duration_ms = 60000
    elif args.min_duration:
        min_duration_ms = args.min_duration
    else:
        min_duration_ms = 600000
    
    # Get number of samples: user override > quick default (50) > full default (100)
    if args.max_samples:
        num_samples = args.max_samples
    elif args.mlperf_quick:
        num_samples = 50
    else:
        num_samples = 100
    
    # Calculate target_qps from samples and duration if not specified
    if args.target_qps:
        target_qps = args.target_qps
    else:
        min_duration_sec = min_duration_ms / 1000
        target_qps = num_samples / (min_duration_sec * 1.1)
    
    output_dir = Path(args.output_dir) if args.output_dir else (get_project_dir() / 'results' / 'sdxl')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # SDXL-specific options
    num_steps = getattr(args, 'num_steps', 20)
    offload = getattr(args, 'offload', False)
    
    _print_config('STABLE DIFFUSION XL', 'sdxl', dataset_path, device,
                  scenario, mode, min_duration_ms, target_qps, batch_size, output_dir,
                  metadata, args)
    
    # Print SDXL-specific config
    print(f"  Diffusion Steps:      {num_steps}")
    print(f"  CPU Offload:          {'enabled' if offload else 'disabled'}")
    
    print("\n" + "=" * 70)
    print("LOADING MODEL AND DATA")
    print("=" * 70)
    
    # Load captions/prompts from captions.json
    captions_file = data_path / "captions.json"
    
    if not captions_file.exists():
        print(f"  Error: No captions.json found in {data_path}")
        return 1
    
    print(f"  Loading prompts from: {captions_file}")
    with open(captions_file) as f:
        prompts = json.load(f)
    
    print(f"  Loaded {len(prompts)} prompts")
    
    # Apply max_samples limit
    if args.max_samples and args.max_samples < len(prompts):
        prompts = prompts[:args.max_samples]
        print(f"  Limited to {len(prompts)} samples")
    
    num_samples = len(prompts)
    
    if num_samples == 0:
        print("Error: No prompts found")
        return 1
    
    # Load SDXL model
    model_name = BENCHMARKS['sdxl']['model_name']
    print(f"\n  Loading model: {model_name}")
    
    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        print("  Error: CUDA not available. Use --device cpu (very slow)")
        return 1
    
    try:
        if device == "cpu":
            torch_dtype = torch.float32
            print("  Loading on CPU (this will be very slow)...")
            pipe = DiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        elif offload:
            torch_dtype = torch.float16
            print("  Loading with model CPU offloading...")
            pipe = DiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                variant="fp16",
            )
            pipe.enable_model_cpu_offload()
        else:
            torch_dtype = torch.float16
            print("  Loading on GPU...")
            pipe = DiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                variant="fp16",
            )
            pipe = pipe.to("cuda")
    except torch.cuda.OutOfMemoryError:
        print("\n" + "=" * 60)
        print("CUDA OUT OF MEMORY!")
        print("=" * 60)
        print("SDXL requires ~6.5GB VRAM. Try:")
        print("  1. --offload : Enable CPU offloading (~3GB VRAM)")
        print("  2. --device cpu : Run on CPU only (very slow)")
        return 1
    
    print("  Model loaded successfully!")
    
    # Disable progress bar for cleaner output during LoadGen test
    pipe.set_progress_bar_config(disable=True)
    
    # Warmup
    print("\n  Warmup run...")
    try:
        _ = pipe(prompts[0], num_inference_steps=5, output_type="latent")
    except torch.cuda.OutOfMemoryError:
        print("\n" + "=" * 60)
        print("CUDA OUT OF MEMORY during warmup!")
        print("=" * 60)
        print("SDXL requires ~6.5GB VRAM. Try:")
        print("  1. --offload : Enable CPU offloading (~3GB VRAM)")
        print("  2. --device cpu : Run on CPU only (very slow)")
        return 1
    print("  Warmup complete")
    
    # Metrics tracking
    total_inference_time = [0.0]
    total_samples_processed = [0]
    total_steps = [0]
    query_start_time = [None]
    query_end_time = [None]
    
    perf_count = min(num_samples, BENCHMARKS['sdxl']['performance_sample_count'])
    
    def issue_queries(query_samples):
        import time as time_module
        responses = []
        
        # Track wall-clock time
        if query_start_time[0] is None:
            query_start_time[0] = time_module.perf_counter()
        
        batch_start = time_module.perf_counter()
        batch_size = len(query_samples)
        
        for i, sample in enumerate(query_samples):
            prompt = prompts[sample.index]
            
            start_time = time_module.perf_counter()
            try:
                image = pipe(
                    prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=7.5,
                ).images[0]
            except torch.cuda.OutOfMemoryError:
                print(f"\n  CUDA OOM at sample {sample.index}! Use --offload")
                response_data = np.array([0], dtype=np.int32)
                response_array = array.array('B', response_data.tobytes())
                bi = response_array.buffer_info()
                responses.append(lg.QuerySampleResponse(sample.id, bi[0], bi[1]))
                continue
            
            elapsed = time_module.perf_counter() - start_time
            total_inference_time[0] += elapsed
            total_samples_processed[0] += 1
            total_steps[0] += num_steps
            
            # Response: just indicate success
            response_data = np.array([1], dtype=np.int32)
            response_array = array.array('B', response_data.tobytes())
            bi = response_array.buffer_info()
            responses.append(lg.QuerySampleResponse(sample.id, bi[0], bi[1]))
        
        lg.QuerySamplesComplete(responses)
        
        # Update wall-clock end time and print progress
        query_end_time[0] = time_module.perf_counter()
        batch_elapsed = query_end_time[0] - batch_start
        total_done = total_samples_processed[0]
        rate = batch_size / batch_elapsed if batch_elapsed > 0 else 0
        print(f"  Processed {total_done} images ({rate:.2f} img/sec)", end='\r')
    
    def flush_queries():
        pass
    
    def load_samples(sample_list):
        pass
    
    def unload_samples(sample_list):
        pass
    
    qsl = lg.ConstructQSL(num_samples, perf_count, load_samples, unload_samples)
    sut = lg.ConstructSUT(issue_queries, flush_queries)
    
    result = _run_loadgen_test(sut, qsl, scenario, mode, min_duration_ms, target_qps, output_dir, num_samples, args)
    
    # Print detailed SDXL metrics after LoadGen completes
    if total_samples_processed[0] > 0:
        samples_completed = total_samples_processed[0]
        inference_time = total_inference_time[0]
        wall_clock_time = query_end_time[0] - query_start_time[0] if query_end_time[0] else inference_time
        
        images_per_sec = samples_completed / wall_clock_time if wall_clock_time > 0 else 0
        pure_images_per_sec = samples_completed / inference_time if inference_time > 0 else 0
        avg_latency = wall_clock_time / samples_completed
        steps_per_sec = total_steps[0] / wall_clock_time if wall_clock_time > 0 else 0
        images_per_min = images_per_sec * 60
        
        print("\n" + "=" * 70)
        print("SDXL BENCHMARK DETAILED METRICS")
        print("=" * 70)
        print(f"  Images Generated:     {samples_completed}")
        print(f"  Diffusion Steps:      {num_steps} per image")
        print(f"  Total Steps:          {total_steps[0]}")
        print(f"  Wall-Clock Time:      {wall_clock_time:.2f} seconds")
        print(f"  Pure Inference Time:  {inference_time:.2f} seconds")
        print(f"  Avg Latency:          {avg_latency:.2f} sec/image")
        print(f"  Throughput:           {steps_per_sec:.2f} steps/sec")
        print(f"  Images/sec:           {images_per_sec:.3f}")
        print(f"  Images/min:           {images_per_min:.1f}")
        print(f"  (Pure GPU):           {pure_images_per_sec:.3f} images/sec")
        print("=" * 70)
        
        if steps_per_sec > 10:
            print("  Performance:          🚀 Excellent")
        elif steps_per_sec > 3:
            print("  Performance:          ✅ Good")
        elif steps_per_sec > 1:
            print("  Performance:          ⚠️  Moderate")
        else:
            print("  Performance:          🐢 Slow")
        print("=" * 70)
    
    # Cleanup
    del pipe
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result


# ============================================================================
# GPT-J Benchmark
# ============================================================================

def run_gptj(dataset_path: str, args: argparse.Namespace, metadata: Optional[Dict] = None) -> int:
    """Run GPT-J Text Summarization benchmark with LoadGen."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    data_path = Path(dataset_path)
    device = 'cuda' if args.device == 'gpu' else args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    scenario = args.scenario if args.scenario else 'Offline'
    mode = args.mode if args.mode else 'performance'
    batch_size = args.batch_size if args.batch_size else 1
    
    if args.mlperf_quick:
        min_duration_ms = 60000
    elif args.min_duration:
        min_duration_ms = args.min_duration
    else:
        min_duration_ms = 600000
    
    # Get number of samples: user override > quick default (50) > full default (100)
    if args.max_samples:
        num_samples = args.max_samples
    elif args.mlperf_quick:
        num_samples = 50
    else:
        num_samples = 100
    
    # Calculate target_qps from samples and duration if not specified
    if args.target_qps:
        target_qps = args.target_qps
    else:
        min_duration_sec = min_duration_ms / 1000
        target_qps = num_samples / (min_duration_sec * 1.1)
    
    output_dir = Path(args.output_dir) if args.output_dir else (get_project_dir() / 'results' / 'gptj')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    _print_config('GPT-J TEXT SUMMARIZATION', 'gptj', dataset_path, device,
                  scenario, mode, min_duration_ms, target_qps, batch_size, output_dir,
                  metadata, args)
    
    print("\n" + "=" * 70)
    print("LOADING MODEL AND DATA")
    print("=" * 70)
    
    # Load text data - expects test.json with article/highlights format
    test_file = data_path / "test.json"
    
    if not test_file.exists():
        print(f"  Error: No test.json found in {data_path}")
        return 1
    
    print(f"  Loading data from: {test_file}")
    with open(test_file) as f:
        data = json.load(f)
    
    # Prepare prompts
    prompts = []
    references = []
    
    for item in data:
        article = item.get('article', '')
        # Truncate article to reasonable length for prompt
        article = article[:1500]
        prompt = f"Summarize the following article:\n\n{article}\n\nSummary:"
        prompts.append(prompt)
        references.append(item.get('highlights', ''))
    
    print(f"  Loaded {len(prompts)} articles")
    
    # Apply max_samples limit
    if args.max_samples and args.max_samples < len(prompts):
        prompts = prompts[:args.max_samples]
        references = references[:args.max_samples]
        print(f"  Limited to {len(prompts)} samples")
    
    num_samples = len(prompts)
    
    # Load model with quantization support
    model_name = BENCHMARKS['gptj']['model_name']
    print(f"\n  Loading model: {model_name}")
    
    # Check quantization options
    use_4bit = getattr(args, 'use_4bit', False)
    use_8bit = getattr(args, 'use_8bit', False)
    
    if use_4bit:
        print("  Using 4-bit quantization (~6GB VRAM)")
    elif use_8bit:
        print("  Using 8-bit quantization (~8GB VRAM)")
    else:
        print("  Note: GPT-J 6B requires ~12GB GPU memory (fp16) or ~24GB (fp32)")
        print("        Use --4bit or --8bit for lower memory usage")
    
    torch_dtype = torch.float16 if device == 'cuda' else torch.float32
    
    try:
        # Setup quantization config if requested
        quantization_config = None
        device_map = None
        max_memory = None
        
        if use_4bit and device == 'cuda':
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                device_map = "auto"
                print("  Quantization: 4-bit (bitsandbytes)")
            except ImportError:
                print("  Warning: bitsandbytes not installed")
                print("           Install with: pip install bitsandbytes")
                return 1
                
        elif use_8bit and device == 'cuda':
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                device_map = "auto"
                # Set max_memory to use most of GPU (leave 10% buffer)
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory
                    max_memory = {0: f"{int(gpu_mem * 0.9 / 1e9)}GB", "cpu": "32GB"}
                print("  Quantization: 8-bit (bitsandbytes)")
            except ImportError:
                print("  Warning: bitsandbytes not installed")
                print("           Install with: pip install bitsandbytes")
                return 1
                
        elif device == 'cuda':
            device_map = {"": 0}
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "device_map": device_map,
            "quantization_config": quantization_config,
        }
        if max_memory:
            model_kwargs["max_memory"] = max_memory
            
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        if device_map is None and device != 'cuda':
            model = model.to(device)
        
        model.eval()
    except Exception as e:
        print(f"\n  Error loading model: {e}")
        print("\n  Tips for OOM errors:")
        print("    - Use --4bit for ~6GB VRAM (smallest)")
        print("    - Use --8bit for ~8GB VRAM")
        print("    - Or use --device cpu (very slow)")
        return 1
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    print(f"  Model loaded on {device}")
    print(f"  Prompts: {num_samples}")
    
    # Pre-tokenize all prompts to avoid repeated tokenization during benchmark
    print("  Pre-tokenizing prompts...")
    tokenized_prompts = []
    max_input_length = 1024  # Limit input length for memory efficiency
    max_new_tokens = 128  # Generated tokens
    
    for prompt in prompts:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding=False,
        )
        tokenized_prompts.append({
            'input_ids': inputs.input_ids.to(device),
            'attention_mask': inputs.attention_mask.to(device),
            'input_length': inputs.input_ids.shape[1],
        })
    print(f"  Pre-tokenized {len(tokenized_prompts)} prompts")
    
    # Warmup
    print("  Running warmup inference...")
    with torch.no_grad():
        # MLPerf GPT-J uses beam search (num_beams=4)
        _ = model.generate(
            input_ids=tokenized_prompts[0]['input_ids'],
            attention_mask=tokenized_prompts[0]['attention_mask'],
            max_new_tokens=32,
            min_new_tokens=1,
            num_beams=4,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    if device == 'cuda':
        torch.cuda.synchronize()
    print("  Warmup complete")
    
    # Metrics tracking
    total_tokens_generated = [0]  # Use list to allow modification in nested function
    sample_latencies = []  # Pure inference time per sample
    generated_texts = []
    query_start_time = [None]  # Wall-clock start time
    query_end_time = [None]    # Wall-clock end time
    
    # Simple ROUGE-L calculation
    def calculate_rouge_l(reference: str, hypothesis: str) -> float:
        """Calculate ROUGE-L F1 score."""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        if not ref_words or not hyp_words:
            return 0.0
        
        # LCS length using dynamic programming
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
    
    # LoadGen callbacks
    perf_count = min(num_samples, BENCHMARKS['gptj']['performance_sample_count'])
    
    def issue_queries(query_samples):
        """Process queries from LoadGen."""
        import time as time_module
        
        # Track wall-clock time
        if query_start_time[0] is None:
            query_start_time[0] = time_module.perf_counter()
        
        for sample in query_samples:
            idx = sample.index
            inputs = tokenized_prompts[idx]
            
            start_time = time_module.perf_counter()
            with torch.no_grad():
                # MLPerf GPT-J uses beam search (num_beams=4)
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=30,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            elapsed = time_module.perf_counter() - start_time
            
            # Track metrics
            tokens_generated = outputs.shape[1] - inputs['input_length']
            total_tokens_generated[0] += tokens_generated
            sample_latencies.append(elapsed)
            
            # Decode for ROUGE-L calculation
            generated_text = tokenizer.decode(outputs[0][inputs['input_length']:], skip_special_tokens=True)
            generated_texts.append((idx, generated_text))
            
            # Create response
            response_data = np.array([tokens_generated], dtype=np.int64)
            response = lg.QuerySampleResponse(sample.id, response_data.ctypes.data, response_data.nbytes)
            lg.QuerySamplesComplete([response])
        
        # Update wall-clock end time after all queries processed
        query_end_time[0] = time_module.perf_counter()
    
    def flush_queries():
        pass
    
    def load_samples(sample_list):
        pass
    
    def unload_samples(sample_list):
        pass
    
    # Create QSL and SUT
    qsl = lg.ConstructQSL(num_samples, perf_count, load_samples, unload_samples)
    sut = lg.ConstructSUT(issue_queries, flush_queries)
    
    result = _run_loadgen_test(sut, qsl, scenario, mode, min_duration_ms, target_qps, output_dir, num_samples, args)
    
    # Print detailed GPT-J metrics after LoadGen completes
    if sample_latencies:
        inference_time = sum(sample_latencies)  # Pure GPU time
        samples_completed = len(sample_latencies)
        
        # Wall-clock time matches LoadGen's measurement
        wall_clock_time = query_end_time[0] - query_start_time[0] if query_end_time[0] else inference_time
        
        # Throughput based on wall-clock time (matches LoadGen)
        samples_per_sec = samples_completed / wall_clock_time if wall_clock_time > 0 else 0
        tokens_per_sec = total_tokens_generated[0] / wall_clock_time if wall_clock_time > 0 else 0
        
        # Pure inference throughput (excludes overhead)
        pure_samples_per_sec = samples_completed / inference_time if inference_time > 0 else 0
        pure_tokens_per_sec = total_tokens_generated[0] / inference_time if inference_time > 0 else 0
        
        # Average latency per sample
        avg_latency = wall_clock_time / samples_completed
        
        # Calculate ROUGE-L if we have references
        rouge_scores = []
        for idx, gen_text in generated_texts:
            if idx < len(references) and references[idx]:
                score = calculate_rouge_l(references[idx], gen_text)
                rouge_scores.append(score)
        
        avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else None
        
        print("\n" + "=" * 70)
        print("GPT-J BENCHMARK DETAILED METRICS")
        print("=" * 70)
        print(f"  Samples Processed:    {samples_completed}")
        print(f"  Total Tokens:         {total_tokens_generated[0]}")
        print(f"  Wall-Clock Time:      {wall_clock_time:.2f} seconds")
        print(f"  Pure Inference Time:  {inference_time:.2f} seconds")
        print(f"  Avg Latency:          {avg_latency:.2f} sec/sample")
        print(f"  Throughput:           {samples_per_sec:.3f} samples/sec")
        print(f"  Token Throughput:     {tokens_per_sec:.2f} tokens/sec")
        print(f"  (Pure GPU):           {pure_tokens_per_sec:.2f} tokens/sec")
        if avg_rouge is not None:
            print(f"  Average ROUGE-L:      {avg_rouge:.3f}")
        print("=" * 70)
        
        # Performance rating based on wall-clock throughput
        if tokens_per_sec > 20:
            print("  Performance:          🚀 Excellent")
        elif tokens_per_sec > 10:
            print("  Performance:          ✅ Good")
        elif tokens_per_sec > 5:
            print("  Performance:          ⚠️  Moderate")
        else:
            print("  Performance:          🐢 Slow (consider --4bit)")
        print("=" * 70)
    
    return result


# ============================================================================
# Llama Benchmark
# ============================================================================

# Supported Llama model variants
LLAMA_MODELS = {
    'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2-70b': 'meta-llama/Llama-2-70b-chat-hf',
    'llama3-8b': 'meta-llama/Llama-3.1-8B-Instruct',
    'llama3-70b': 'meta-llama/Llama-3.1-70B-Instruct',
}

def run_llama(dataset_path: str, args: argparse.Namespace, metadata: Optional[Dict] = None) -> int:
    """Run Llama Text Generation benchmark with LoadGen.
    
    Supports multiple Llama variants with quantization and offloading.
    Datasets: OpenOrca (instruction following) or CNN-DailyMail (summarization)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    data_path = Path(dataset_path)
    device = 'cuda' if args.device == 'gpu' else args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    scenario = args.scenario if args.scenario else 'Offline'
    mode = args.mode if args.mode else 'performance'
    batch_size = args.batch_size if args.batch_size else 1
    
    # Determine min_duration
    if args.mlperf_quick:
        min_duration_ms = 60000
    elif args.min_duration:
        min_duration_ms = args.min_duration
    else:
        min_duration_ms = 600000
    
    # Get number of samples: user override > quick default (50) > full default (100)
    if args.max_samples:
        num_samples = args.max_samples
    elif args.mlperf_quick:
        num_samples = 50
    else:
        num_samples = 100
    
    # Calculate target_qps if not specified by user
    # Formula: LoadGen generates samples_per_query = target_qps × min_duration × 1.1
    # So: target_qps = num_samples / (min_duration_sec × 1.1)
    if args.target_qps:
        target_qps = args.target_qps
    else:
        min_duration_sec = min_duration_ms / 1000
        target_qps = num_samples / (min_duration_sec * 1.1)
    
    output_dir = Path(args.output_dir) if args.output_dir else (get_project_dir() / 'results' / 'llama')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for model variant argument
    model_variant = getattr(args, 'model_variant', None) or 'llama2-70b'
    if model_variant in LLAMA_MODELS:
        model_name = LLAMA_MODELS[model_variant]
    else:
        # Allow direct HuggingFace model name
        model_name = model_variant if '/' in model_variant else LLAMA_MODELS.get('llama2-70b')
    
    # Update BENCHMARKS for display
    actual_model_name = model_name.split('/')[-1]
    
    _print_config('LLAMA TEXT GENERATION', 'llama', dataset_path, device,
                  scenario, mode, min_duration_ms, target_qps, batch_size, output_dir,
                  metadata, args)
    
    print("\n" + "=" * 70)
    print("LOADING MODEL AND DATA")
    print("=" * 70)
    
    # =========================================================================
    # Load Data
    # =========================================================================
    test_file = data_path / "test.json"
    
    if not test_file.exists():
        print(f"  Error: No test.json found in {data_path}")
        print(f"  Run: python data_prepare.py llama --mode mlperf")
        return 1
    
    print(f"  Loading data from: {test_file}")
    with open(test_file) as f:
        data = json.load(f)
    
    # Detect data format and prepare prompts
    prompts = []
    references = []
    
    # Check format: OpenOrca vs CNN-DailyMail
    first_item = data[0] if data else {}
    is_openorca = 'question' in first_item and 'response' in first_item
    is_cnn = 'article' in first_item and 'highlights' in first_item
    
    if is_openorca:
        print(f"  Detected format: OpenOrca (instruction following)")
        for item in data:
            # OpenOrca format - already formatted prompts or raw questions
            question = item.get('question', '')
            
            # Check if already in Llama format
            if '[INST]' in question or '<|begin_of_text|>' in question:
                prompt = question
            else:
                # Format for Llama chat
                system = item.get('system_prompt', '')
                if system:
                    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{question} [/INST]"
                else:
                    prompt = f"<s>[INST] {question} [/INST]"
            
            prompts.append(prompt)
            references.append(item.get('response', ''))
            
    elif is_cnn:
        print(f"  Detected format: CNN-DailyMail (summarization)")
        for item in data:
            article = item.get('article', '')[:2000]  # Truncate for context window
            prompt = f"<s>[INST] Summarize the following article:\n\n{article}\n\nSummary: [/INST]"
            prompts.append(prompt)
            references.append(item.get('highlights', ''))
    else:
        print(f"  Error: Unknown data format. Expected 'question/response' or 'article/highlights'")
        return 1
    
    print(f"  Loaded {len(prompts)} samples")
    
    # Apply max_samples limit
    if args.max_samples and args.max_samples < len(prompts):
        prompts = prompts[:args.max_samples]
        references = references[:args.max_samples]
        print(f"  Limited to {len(prompts)} samples")
    
    num_samples = len(prompts)
    
    # =========================================================================
    # Load Model
    # =========================================================================
    print(f"\n  Loading model: {model_name}")
    
    # Check quantization options
    use_4bit = getattr(args, 'use_4bit', False)
    use_8bit = getattr(args, 'use_8bit', False)
    use_offload = getattr(args, 'offload', False)
    
    # Estimate memory requirements
    model_size_map = {
        '7b': 14, '8b': 16, '13b': 26, '70b': 140,
    }
    model_size_gb = 140  # default
    for key, size in model_size_map.items():
        if key in model_name.lower():
            model_size_gb = size
            break
    
    if use_4bit:
        print(f"  Using 4-bit quantization (~{model_size_gb//4}GB VRAM)")
    elif use_8bit:
        print(f"  Using 8-bit quantization (~{model_size_gb//2}GB VRAM)")
    elif use_offload:
        print(f"  Using CPU offloading (model: ~{model_size_gb}GB)")
    else:
        print(f"  Note: {actual_model_name} requires ~{model_size_gb}GB GPU memory (fp16)")
        if model_size_gb > 20:
            print(f"        Use --4bit, --8bit, or --offload for large models")
    
    torch_dtype = torch.float16 if device == 'cuda' else torch.float32
    
    try:
        # Setup quantization config if requested
        quantization_config = None
        device_map = None
        max_memory = None
        
        if use_4bit and device == 'cuda':
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                device_map = "auto"
                print("  Quantization: 4-bit (bitsandbytes)")
            except ImportError:
                print("  Warning: bitsandbytes not installed")
                print("           Install with: pip install bitsandbytes")
                return 1
                
        elif use_8bit and device == 'cuda':
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                device_map = "auto"
                # Set max_memory to use most of GPU (leave 1GB buffer)
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory
                    max_memory = {0: f"{int(gpu_mem * 0.9 / 1e9)}GB", "cpu": "32GB"}
                print("  Quantization: 8-bit (bitsandbytes)")
            except ImportError:
                print("  Warning: bitsandbytes not installed")
                return 1
                
        elif use_offload:
            device_map = "auto"  # Will automatically offload to CPU as needed
            print("  Mode: GPU + CPU offloading")
            
        elif device == 'cuda':
            device_map = {"": 0}
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "device_map": device_map,
            "quantization_config": quantization_config,
            "trust_remote_code": True,
        }
        if max_memory:
            model_kwargs["max_memory"] = max_memory
            
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        model.eval()
        
    except torch.cuda.OutOfMemoryError:
        print(f"\n" + "=" * 70)
        print("❌ OUT OF MEMORY: Cannot fit model on GPU")
        print("=" * 70)
        print(f"  Model: {actual_model_name} (~{model_size_gb}GB)")
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU memory: {gpu_mem_gb:.1f} GB")
        print(f"\n  💡 SOLUTIONS:")
        print(f"\n  1. Use 4-bit quantization (recommended):")
        print(f"     python benchmark_new.py -b llama --mlperf --4bit")
        print(f"\n  2. Use 8-bit quantization:")
        print(f"     python benchmark_new.py -b llama --mlperf --8bit")
        print(f"\n  3. Use CPU offloading (slower):")
        print(f"     python benchmark_new.py -b llama --mlperf --offload")
        print(f"\n  4. Use smaller model variant:")
        print(f"     python benchmark_new.py -b llama --mlperf --4bit --model llama3-8b")
        print("=" * 70)
        return 1
        
    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "403" in error_msg or "gated" in error_msg:
            print(f"\n" + "=" * 70)
            print("❌ ACCESS DENIED: Model requires authentication")
            print("=" * 70)
            print(f"\n  Model: {model_name}")
            print(f"\n  Steps to gain access:")
            print(f"  1. Create HuggingFace account: https://huggingface.co/join")
            print(f"  2. Request model access: https://huggingface.co/{model_name}")
            print(f"  3. Login with: huggingface-cli login")
            print("=" * 70)
            return 1
        else:
            print(f"\n  Error loading model: {e}")
            return 1
    
    # Print memory usage
    if device == 'cuda' and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    print(f"  Model loaded successfully")
    
    # =========================================================================
    # Pre-tokenize prompts
    # =========================================================================
    print("  Pre-tokenizing prompts...")
    tokenized_prompts = []
    max_input_length = 2048  # Llama context window
    
    # Max new tokens - use command line override or defaults
    if hasattr(args, 'max_tokens') and args.max_tokens:
        max_new_tokens = args.max_tokens
    elif args.mlperf:
        max_new_tokens = 1024  # MLPerf official requirement
    elif args.mlperf_quick:
        max_new_tokens = 128  # Quick mode - faster iteration
    else:
        max_new_tokens = 128  # Default for testing
    
    for prompt in prompts:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding=False,
        )
        
        # Determine target device
        if hasattr(model, 'device'):
            target_device = model.device
        elif hasattr(model, 'hf_device_map'):
            # For device_map="auto", use the first device
            target_device = 'cuda' if 'cuda' in str(model.hf_device_map) else 'cpu'
        else:
            target_device = device
        
        tokenized_prompts.append({
            'input_ids': inputs.input_ids.to(target_device),
            'attention_mask': inputs.attention_mask.to(target_device),
            'input_length': inputs.input_ids.shape[1],
        })
    print(f"  Pre-tokenized {len(tokenized_prompts)} prompts")
    print(f"  Max new tokens: {max_new_tokens}")
    
    # =========================================================================
    # Warmup
    # =========================================================================
    print("  Running warmup inference...")
    with torch.no_grad():
        # MLPerf Llama official settings:
        # early_stopping=True, max_new_tokens=1024, min_new_tokens=1
        # num_beams=1, do_sample=False
        _ = model.generate(
            input_ids=tokenized_prompts[0]['input_ids'],
            attention_mask=tokenized_prompts[0]['attention_mask'],
            max_new_tokens=32,
            min_new_tokens=1,
            num_beams=1,
            early_stopping=True,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    if device == 'cuda':
        torch.cuda.synchronize()
    print("  Warmup complete")
    
    # =========================================================================
    # Metrics tracking
    # =========================================================================
    total_tokens_generated = [0]
    sample_latencies = []
    generated_texts = []
    query_start_time = [None]
    query_end_time = [None]
    
    perf_count = min(num_samples, BENCHMARKS['llama']['performance_sample_count'])
    
    # Calculate expected queries from LoadGen (Offline mode: target_qps * min_duration)
    expected_queries = int(target_qps * (min_duration_ms / 1000))
    if expected_queries > num_samples:
        print(f"\n  Note: LoadGen will send {expected_queries} queries using {num_samples} unique samples")
        print(f"        (samples will be reused to meet {min_duration_ms/1000:.0f}s minimum duration)")
    
    # =========================================================================
    # LoadGen callbacks
    # =========================================================================
    query_count = [0]
    
    def issue_queries(query_samples):
        import time as time_module
        
        if query_start_time[0] is None:
            query_start_time[0] = time_module.perf_counter()
        
        for sample in query_samples:
            idx = sample.index
            inputs = tokenized_prompts[idx]
            query_count[0] += 1
            
            start_time = time_module.perf_counter()
            with torch.no_grad():
                # MLPerf Llama official settings:
                # early_stopping=True, max_new_tokens=1024, min_new_tokens=1
                # num_beams=1, do_sample=False
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
                    num_beams=1,
                    early_stopping=True,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time_module.perf_counter() - start_time
            
            # Track metrics
            tokens_generated = outputs.shape[1] - inputs['input_length']
            total_tokens_generated[0] += tokens_generated
            sample_latencies.append(elapsed)
            
            # Progress output - match old run_llama_benchmark.py format
            tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0
            print(f"  [{query_count[0]}/{num_samples}] {tokens_generated} tokens, {tokens_per_sec:.1f} tok/s, {elapsed:.2f}s")
            
            # Decode for accuracy
            generated_text = tokenizer.decode(outputs[0][inputs['input_length']:], skip_special_tokens=True)
            generated_texts.append((idx, generated_text))
            
            # Create response
            response_data = np.array([tokens_generated], dtype=np.int64)
            response = lg.QuerySampleResponse(sample.id, response_data.ctypes.data, response_data.nbytes)
            lg.QuerySamplesComplete([response])
        
        query_end_time[0] = time_module.perf_counter()
    
    def flush_queries():
        pass
    
    def load_samples(sample_list):
        pass
    
    def unload_samples(sample_list):
        pass
    
    # =========================================================================
    # Run LoadGen
    # =========================================================================
    qsl = lg.ConstructQSL(num_samples, perf_count, load_samples, unload_samples)
    sut = lg.ConstructSUT(issue_queries, flush_queries)
    
    result = _run_loadgen_test(sut, qsl, scenario, mode, min_duration_ms, target_qps, output_dir, num_samples, args)
    
    # =========================================================================
    # Print detailed metrics
    # =========================================================================
    if sample_latencies:
        inference_time = sum(sample_latencies)
        samples_completed = len(sample_latencies)
        wall_clock_time = query_end_time[0] - query_start_time[0] if query_end_time[0] else inference_time
        
        samples_per_sec = samples_completed / wall_clock_time if wall_clock_time > 0 else 0
        tokens_per_sec = total_tokens_generated[0] / wall_clock_time if wall_clock_time > 0 else 0
        pure_tokens_per_sec = total_tokens_generated[0] / inference_time if inference_time > 0 else 0
        avg_latency = wall_clock_time / samples_completed
        avg_tokens_per_sample = total_tokens_generated[0] / samples_completed
        
        print("\n" + "=" * 70)
        print("LLAMA BENCHMARK DETAILED METRICS")
        print("=" * 70)
        print(f"  Model:                {actual_model_name}")
        print(f"  Samples Processed:    {samples_completed}")
        print(f"  Total Tokens:         {total_tokens_generated[0]}")
        print(f"  Avg Tokens/Sample:    {avg_tokens_per_sample:.1f}")
        print(f"  Wall-Clock Time:      {wall_clock_time:.2f} seconds")
        print(f"  Pure Inference Time:  {inference_time:.2f} seconds")
        print(f"  Avg Latency:          {avg_latency:.2f} sec/sample")
        print(f"  Throughput:           {samples_per_sec:.3f} samples/sec")
        print(f"  Token Throughput:     {tokens_per_sec:.2f} tokens/sec")
        print(f"  (Pure GPU):           {pure_tokens_per_sec:.2f} tokens/sec")
        print("=" * 70)
        
        # Performance rating
        if tokens_per_sec > 50:
            print("  Performance:          🚀 Excellent")
        elif tokens_per_sec > 20:
            print("  Performance:          ✅ Good")
        elif tokens_per_sec > 10:
            print("  Performance:          ⚠️  Moderate")
        else:
            print("  Performance:          🐢 Slow")
        print("=" * 70)
    
    return result


# ============================================================================
# Mixtral Benchmark
# ============================================================================

def _print_mixtral_oom_error(use_4bit: bool, use_8bit: bool, use_offload: bool):
    """Print helpful OOM error message with suggestions for Mixtral."""
    print(f"\n" + "=" * 70)
    print("❌ OUT OF MEMORY: Cannot fit Mixtral-8x7B on GPU")
    print("=" * 70)
    
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU memory: {gpu_mem_gb:.1f} GB")
    
    print(f"\n  Current settings: --4bit={use_4bit}, --8bit={use_8bit}, --offload={use_offload}")
    print(f"\n  💡 SOLUTIONS (ordered by VRAM needed):")
    
    # Give contextual advice based on current options
    if not use_offload:
        print(f"\n  Try adding --offload to enable CPU offloading:")
        if use_4bit:
            print(f"     python benchmark_new.py -b mixtral --4bit --offload ...")
        elif use_8bit:
            print(f"     python benchmark_new.py -b mixtral --8bit --offload ...")
        else:
            print(f"     python benchmark_new.py -b mixtral --offload ...")
    else:
        print(f"\n  Already using offload. Try reducing quantization precision:")
        if use_8bit:
            print(f"     python benchmark_new.py -b mixtral --4bit --offload ...")
        print(f"     python benchmark_new.py -b mixtral --offload ...  (FP16, slowest)")
    
    print(f"\n  VRAM requirements for Mixtral-8x7B:")
    print(f"     --offload           : ~8GB  VRAM (FP16, slowest)")
    print(f"     --4bit --offload    : ~12GB VRAM (experimental)")
    print(f"     --8bit --offload    : ~16GB VRAM")
    print(f"     --4bit              : ~24GB VRAM")
    print(f"     --8bit              : ~48GB VRAM")
    print(f"     (no options)        : ~90GB VRAM")
    print("=" * 70)


def run_mixtral(dataset_path: str, args: argparse.Namespace, metadata: Optional[Dict] = None) -> int:
    """Run Mixtral-8x7B Text Generation benchmark with LoadGen.
    
    Mixtral-8x7B is a large Mixture-of-Experts model (~93GB parameters).
    Uses combined dataset: OpenOrca + GSM8K + MBXP (15K samples).
    
    VRAM requirements (approximate):
    - Full precision (fp16): ~90GB
    - 8-bit quantization: ~48GB
    - 4-bit quantization: ~24GB
    - 4-bit + offload: ~12GB
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
    import time as time_module
    
    # Custom streamer for progress feedback during generation
    class ProgressStreamer(TextStreamer):
        """Streamer that shows token generation progress."""
        def __init__(self, tokenizer, prefix="", show_tokens=True, **kwargs):
            super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True, **kwargs)
            self.prefix = prefix
            self.show_tokens = show_tokens
            self.token_count = 0
            self.start_time = time_module.perf_counter()
            self.last_update = self.start_time
            
        def on_finalized_text(self, text: str, stream_end: bool = False):
            self.token_count += 1
            now = time_module.perf_counter()
            # Update every 0.5 seconds to avoid too much output
            if now - self.last_update >= 0.5 or stream_end:
                elapsed = now - self.start_time
                tok_per_sec = self.token_count / elapsed if elapsed > 0 else 0
                if stream_end:
                    print(f"\r  {self.prefix}{self.token_count} tokens, {tok_per_sec:.2f} tok/s, {elapsed:.1f}s")
                else:
                    print(f"\r  {self.prefix}{self.token_count} tokens, {tok_per_sec:.2f} tok/s, {elapsed:.1f}s...", end="", flush=True)
                self.last_update = now
    
    data_path = Path(dataset_path)
    device = 'cuda' if args.device == 'gpu' else args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    scenario = args.scenario if args.scenario else 'Offline'
    mode = args.mode if args.mode else 'performance'
    batch_size = args.batch_size if args.batch_size else 1
    
    # Model name - Mixtral 8x7B Instruct
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    # Determine min_duration
    if args.mlperf_quick:
        min_duration_ms = 60000
    elif args.min_duration:
        min_duration_ms = args.min_duration
    else:
        min_duration_ms = 600000
    
    # Get number of samples: 
    # For Mixtral with quick mode, override to 1 sample (very slow with offload)
    # Otherwise: user override > quick default (50) > full default (100)
    if args.mlperf_quick and not args.max_samples:
        # This shouldn't happen since main() sets max_samples=50 for quick mode
        num_samples = 1
    elif args.mlperf_quick and args.max_samples == 50:
        # Quick mode default of 50 is too slow for Mixtral, override to 1
        num_samples = 1
    elif args.max_samples:
        num_samples = args.max_samples
    else:
        num_samples = 100
    
    # Calculate target_qps if not specified
    if args.target_qps:
        target_qps = args.target_qps
    elif args.mlperf_quick:
        target_qps = 0.01  # Low QPS for Mixtral quick test (model is very slow with offload)
    else:
        min_duration_sec = min_duration_ms / 1000
        target_qps = num_samples / (min_duration_sec * 1.1)
    
    # Set warmup tokens for quick mode if not specified
    warmup_tokens = getattr(args, 'warmup_tokens', 32)
    if args.mlperf_quick and warmup_tokens == 32:
        args.warmup_tokens = 4  # Reduced warmup for Mixtral quick test
    
    output_dir = Path(args.output_dir) if args.output_dir else (get_project_dir() / 'results' / 'mixtral-8x7b')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    _print_config('MIXTRAL-8x7B TEXT GENERATION', 'mixtral', dataset_path, device,
                  scenario, mode, min_duration_ms, target_qps, batch_size, output_dir,
                  metadata, args)
    
    # Show quick mode optimizations for Mixtral
    if args.mlperf_quick:
        print("\n" + "-" * 70)
        print("QUICK MODE OPTIMIZATIONS (Mixtral is very slow with CPU offload)")
        print("-" * 70)
        print(f"  Samples:       {num_samples} (default: 100)")
        print(f"  Target QPS:    {target_qps} (default: auto-calculated)")
        print(f"  Warmup tokens: {getattr(args, 'warmup_tokens', 32)} (default: 32)")
        print(f"  These settings reduce test time from ~hours to ~minutes")
    
    print("\n" + "=" * 70)
    print("LOADING MODEL AND DATA")
    print("=" * 70)
    
    # =========================================================================
    # Load Data
    # =========================================================================
    test_file = data_path / "test.json"
    
    if not test_file.exists():
        print(f"  Error: No test.json found in {data_path}")
        print(f"  Run: python data_prepare.py mixtral --mode mlperf")
        return 1
    
    print(f"  Loading data from: {test_file}")
    with open(test_file) as f:
        data = json.load(f)
    
    # Prepare prompts - MLPerf Mixtral data is already pre-formatted
    prompts = []
    references = []
    
    # Check data format
    first_item = data[0] if data else {}
    is_preformatted = 'question' in first_item and ('[INST]' in first_item.get('question', '') or '<s>' in first_item.get('question', ''))
    
    if is_preformatted:
        print(f"  Detected format: MLPerf pre-formatted (GSM8K/OpenOrca/MBXP)")
        for item in data:
            question = item.get('question', '')
            prompts.append(question)
            references.append(item.get('response', ''))
    else:
        # Legacy format - wrap in Mixtral instruction format
        print(f"  Detected format: Raw questions (wrapping in [INST] format)")
        for item in data:
            question = item.get('question', item.get('input', ''))[:1500]
            prompt = f"[INST] {question} [/INST]"
            prompts.append(prompt)
            references.append(item.get('response', item.get('output', '')))
    
    print(f"  Loaded {len(prompts)} samples")
    
    # Apply max_samples limit
    if args.max_samples and args.max_samples < len(prompts):
        prompts = prompts[:args.max_samples]
        references = references[:args.max_samples]
        print(f"  Limited to {len(prompts)} samples")
    
    num_samples = len(prompts)
    
    # =========================================================================
    # Load Model
    # =========================================================================
    print(f"\n  Loading model: {model_name}")
    print(f"  NOTE: Mixtral-8x7B is a large MoE model (~93GB parameters)")
    
    # Check quantization options
    use_4bit = getattr(args, 'use_4bit', False)
    use_8bit = getattr(args, 'use_8bit', False)
    use_offload = getattr(args, 'offload', False)
    
    # Print VRAM guidance
    if use_4bit and use_offload:
        print(f"  Using 4-bit quantization + CPU offload (experimental, ~12GB VRAM)")
        print(f"  WARNING: 4-bit + offload may fail on GPUs with limited VRAM")
    elif use_4bit:
        print(f"  Using 4-bit quantization (~24GB VRAM)")
    elif use_8bit and use_offload:
        print(f"  Using 8-bit quantization + CPU offload (~16GB VRAM)")
    elif use_8bit:
        print(f"  Using 8-bit quantization (~48GB VRAM)")
    elif use_offload:
        print(f"  Using FP16 + CPU offload (~8GB VRAM, slower)")
    else:
        print(f"  WARNING: Full precision requires ~90GB VRAM!")
        print(f"           Recommend: --4bit, --8bit, or --offload")
    
    torch_dtype = torch.float16 if device == 'cuda' else torch.float32
    
    try:
        # Setup quantization config
        quantization_config = None
        device_map = None
        max_memory = None
        
        # Calculate max_memory to leave headroom for inference
        # Mixtral MoE needs extra memory during forward pass
        if device == 'cuda' and torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            # Leave 2GB headroom for inference (MoE experts activation)
            max_gpu_mem = int((gpu_mem - 2 * 1024**3) * 0.9)
            if max_gpu_mem > 0:
                max_memory = {0: f"{max_gpu_mem // (1024**3)}GB", "cpu": "64GB"}
        
        if use_4bit and use_offload and device == 'cuda':
            # 4-bit with CPU offloading - experimental
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                device_map = "auto"
                print("  Quantization: 4-bit + offload (bitsandbytes)")
            except ImportError:
                print("  Error: bitsandbytes not installed")
                print("         Install with: pip install bitsandbytes")
                return 1
                
        elif use_4bit and device == 'cuda':
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                device_map = "auto"
                print("  Quantization: 4-bit (bitsandbytes)")
            except ImportError:
                print("  Error: bitsandbytes not installed")
                print("         Install with: pip install bitsandbytes")
                return 1
                
        elif use_8bit and device == 'cuda':
            try:
                from transformers import BitsAndBytesConfig
                if use_offload:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True
                    )
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                device_map = "auto"
                print("  Quantization: 8-bit (bitsandbytes)")
            except ImportError:
                print("  Error: bitsandbytes not installed")
                return 1
                
        elif use_offload:
            device_map = "auto"
            print("  Mode: GPU + CPU offloading (FP16)")
            
        elif device == 'cuda':
            device_map = {"": 0}
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "device_map": device_map,
            "trust_remote_code": True,
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        if max_memory and use_offload:
            model_kwargs["max_memory"] = max_memory
            
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()
        
    except torch.cuda.OutOfMemoryError:
        _print_mixtral_oom_error(use_4bit, use_8bit, use_offload)
        return 1
        
    except ValueError as e:
        error_msg = str(e)
        if "CPU or the disk" in error_msg or "enough GPU RAM" in error_msg:
            _print_mixtral_oom_error(use_4bit, use_8bit, use_offload)
            return 1
        else:
            print(f"\n  Error loading model: {e}")
            return 1
        
    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "403" in error_msg or "gated" in error_msg:
            print(f"\n" + "=" * 70)
            print("❌ ACCESS DENIED: Model requires authentication")
            print("=" * 70)
            print(f"\n  Model: {model_name}")
            print(f"\n  Steps to gain access:")
            print(f"  1. Create HuggingFace account: https://huggingface.co/join")
            print(f"  2. Request model access: https://huggingface.co/{model_name}")
            print(f"  3. Login with: huggingface-cli login")
            print("=" * 70)
            return 1
        else:
            print(f"\n  Error loading model: {e}")
            return 1
    
    # Print memory usage
    if device == 'cuda' and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    print(f"  Model loaded successfully")
    
    # =========================================================================
    # Pre-tokenize prompts
    # =========================================================================
    print("  Pre-tokenizing prompts...")
    tokenized_prompts = []
    max_input_length = 2048  # Mixtral context window
    
    # Max new tokens - use command line override or defaults
    if hasattr(args, 'max_tokens') and args.max_tokens:
        max_new_tokens = args.max_tokens
    elif args.mlperf:
        max_new_tokens = 1024  # MLPerf official requirement
    elif args.mlperf_quick:
        max_new_tokens = 128  # Quick mode - faster iteration
    else:
        max_new_tokens = 128  # Default for testing
    
    for prompt in prompts:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding=False,
        )
        
        # Determine target device
        if hasattr(model, 'device'):
            target_device = model.device
        elif hasattr(model, 'hf_device_map'):
            target_device = 'cuda' if 'cuda' in str(model.hf_device_map) else 'cpu'
        else:
            target_device = device
        
        tokenized_prompts.append({
            'input_ids': inputs.input_ids.to(target_device),
            'attention_mask': inputs.attention_mask.to(target_device),
            'input_length': inputs.input_ids.shape[1],
        })
    print(f"  Pre-tokenized {len(tokenized_prompts)} prompts")
    print(f"  Max new tokens: {max_new_tokens}")
    
    # =========================================================================
    # Warmup
    # =========================================================================
    warmup_tokens = getattr(args, 'warmup_tokens', 32)
    print(f"  Running warmup inference ({warmup_tokens} tokens)...")
    if use_offload:
        print(f"  NOTE: With CPU offload, warmup takes ~{warmup_tokens * 40}s ({warmup_tokens} tokens × ~40s/token)")
    try:
        warmup_streamer = ProgressStreamer(tokenizer, prefix="Warmup: ")
        with torch.no_grad():
            # MLPerf Mixtral official settings:
            # min_new_tokens=2, max_new_tokens=1024, do_sample=False
            # temperature=None, top_p=None (no num_beams, defaults to greedy)
            _ = model.generate(
                input_ids=tokenized_prompts[0]['input_ids'],
                attention_mask=tokenized_prompts[0]['attention_mask'],
                max_new_tokens=warmup_tokens,
                min_new_tokens=min(2, warmup_tokens),
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
                streamer=warmup_streamer,
            )
        if device == 'cuda':
            torch.cuda.synchronize()
        print("  Warmup complete")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        error_msg = str(e).lower()
        if 'out of memory' in error_msg or 'cuda' in error_msg:
            _print_mixtral_oom_error(use_4bit, use_8bit, use_offload)
            print(f"\n  OOM occurred during warmup inference")
            return 1
        raise  # Re-raise if it's a different error
    
    # =========================================================================
    # Metrics tracking
    # =========================================================================
    total_tokens_generated = [0]
    sample_latencies = []
    generated_texts = []
    query_start_time = [None]
    query_end_time = [None]
    
    perf_count = min(num_samples, BENCHMARKS['mixtral']['performance_sample_count'])
    
    # Calculate expected queries
    expected_queries = int(target_qps * (min_duration_ms / 1000))
    if expected_queries > num_samples:
        print(f"\n  Note: LoadGen will send {expected_queries} queries using {num_samples} unique samples")
        print(f"        (samples will be reused to meet {min_duration_ms/1000:.0f}s minimum duration)")
    
    # =========================================================================
    # LoadGen callbacks
    # =========================================================================
    query_count = [0]
    oom_error_occurred = [False]  # Track OOM across callback invocations
    
    def issue_queries(query_samples):
        import time as time_module
        
        # If OOM already occurred, just complete queries without processing
        if oom_error_occurred[0]:
            for sample in query_samples:
                response_data = np.array([0], dtype=np.int64)
                response = lg.QuerySampleResponse(sample.id, response_data.ctypes.data, response_data.nbytes)
                lg.QuerySamplesComplete([response])
            return
        
        if query_start_time[0] is None:
            query_start_time[0] = time_module.perf_counter()
        
        for sample in query_samples:
            idx = sample.index
            inputs = tokenized_prompts[idx]
            query_count[0] += 1
            
            try:
                start_time = time_module.perf_counter()
                with torch.no_grad():
                    # MLPerf Mixtral official settings:
                    # min_new_tokens=2, max_new_tokens=1024, do_sample=False
                    # temperature=None, top_p=None (no num_beams, defaults to greedy)
                    outputs = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=2,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                if device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time_module.perf_counter() - start_time
                
                # Track metrics
                tokens_generated = outputs.shape[1] - inputs['input_length']
                total_tokens_generated[0] += tokens_generated
                sample_latencies.append(elapsed)
                
                # Progress output
                tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0
                print(f"  [{query_count[0]}/{num_samples}] {tokens_generated} tokens, {tokens_per_sec:.1f} tok/s, {elapsed:.2f}s")
                
                # Decode for accuracy
                generated_text = tokenizer.decode(outputs[0][inputs['input_length']:], skip_special_tokens=True)
                generated_texts.append((idx, generated_text))
                
                # Create response
                response_data = np.array([tokens_generated], dtype=np.int64)
                response = lg.QuerySampleResponse(sample.id, response_data.ctypes.data, response_data.nbytes)
                lg.QuerySamplesComplete([response])
                
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                error_msg = str(e).lower()
                if 'out of memory' in error_msg or 'cuda' in error_msg:
                    oom_error_occurred[0] = True
                    print(f"\n  ❌ OOM during inference at query {query_count[0]}")
                    # Complete this query with dummy response
                    response_data = np.array([0], dtype=np.int64)
                    response = lg.QuerySampleResponse(sample.id, response_data.ctypes.data, response_data.nbytes)
                    lg.QuerySamplesComplete([response])
                    return  # Exit the callback, OOM will be handled after LoadGen
                raise  # Re-raise if it's a different error
        
        query_end_time[0] = time_module.perf_counter()
    
    def flush_queries():
        pass
    
    def load_samples(sample_list):
        pass
    
    def unload_samples(sample_list):
        pass
    
    # =========================================================================
    # Run LoadGen
    # =========================================================================
    qsl = lg.ConstructQSL(num_samples, perf_count, load_samples, unload_samples)
    sut = lg.ConstructSUT(issue_queries, flush_queries)
    
    result = _run_loadgen_test(sut, qsl, scenario, mode, min_duration_ms, target_qps, output_dir, num_samples, args)
    
    # Check if OOM occurred during benchmark
    if oom_error_occurred[0]:
        _print_mixtral_oom_error(use_4bit, use_8bit, use_offload)
        print(f"\n  ❌ Benchmark failed due to OOM during inference")
        print(f"     Completed {len(sample_latencies)} queries before OOM")
        lg.DestroyQSL(qsl)
        lg.DestroySUT(sut)
        return 1
    
    # =========================================================================
    # Print detailed metrics
    # =========================================================================
    if sample_latencies:
        inference_time = sum(sample_latencies)
        samples_completed = len(sample_latencies)
        wall_clock_time = query_end_time[0] - query_start_time[0] if query_end_time[0] else inference_time
        
        samples_per_sec = samples_completed / wall_clock_time if wall_clock_time > 0 else 0
        tokens_per_sec = total_tokens_generated[0] / wall_clock_time if wall_clock_time > 0 else 0
        pure_tokens_per_sec = total_tokens_generated[0] / inference_time if inference_time > 0 else 0
        avg_latency = wall_clock_time / samples_completed
        avg_tokens_per_sample = total_tokens_generated[0] / samples_completed
        
        print("\n" + "=" * 70)
        print("MIXTRAL-8x7B BENCHMARK DETAILED METRICS")
        print("=" * 70)
        print(f"  Model:                {model_name.split('/')[-1]}")
        print(f"  Samples Processed:    {samples_completed}")
        print(f"  Total Tokens:         {total_tokens_generated[0]}")
        print(f"  Avg Tokens/Sample:    {avg_tokens_per_sample:.1f}")
        print(f"  Wall-Clock Time:      {wall_clock_time:.2f} seconds")
        print(f"  Pure Inference Time:  {inference_time:.2f} seconds")
        print(f"  Avg Latency:          {avg_latency:.2f} sec/sample")
        print(f"  Throughput:           {samples_per_sec:.3f} samples/sec")
        print(f"  Token Throughput:     {tokens_per_sec:.2f} tokens/sec")
        print(f"  (Pure GPU):           {pure_tokens_per_sec:.2f} tokens/sec")
        print("=" * 70)
        
        # Performance rating
        if tokens_per_sec > 20:
            print("  Performance:          🚀 Excellent")
        elif tokens_per_sec > 10:
            print("  Performance:          ✅ Good")
        elif tokens_per_sec > 5:
            print("  Performance:          ⚠️  Moderate")
        else:
            print("  Performance:          🐢 Slow (try --4bit or reduce samples)")
        print("=" * 70)
    
    return result


# ============================================================================
# DLRM Benchmark
# ============================================================================

def run_dlrm(dataset_path: str, args: argparse.Namespace, metadata: Optional[Dict] = None) -> int:
    """Run DLRM Recommendation benchmark with LoadGen.
    
    DLRM (Deep Learning Recommendation Model) for click-through rate prediction.
    Uses preprocessed Criteo Terabyte dataset in MLPerf format.
    """
    import torch.nn as nn
    
    data_path = Path(dataset_path)
    device = 'cuda' if args.device == 'gpu' else args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    scenario = args.scenario if args.scenario else 'Offline'
    mode = args.mode if args.mode else 'performance'
    batch_size = args.batch_size if args.batch_size else 2048
    
    if args.mlperf_quick:
        min_duration_ms = 60000
    elif args.min_duration:
        min_duration_ms = args.min_duration
    else:
        min_duration_ms = 600000
    
    # Get number of samples: user override > quick default (50) > full default (100)
    if args.max_samples:
        num_samples = args.max_samples
    elif args.mlperf_quick:
        num_samples = 50
    else:
        num_samples = 100
    
    # Calculate target_qps from samples and duration if not specified
    if args.target_qps:
        target_qps = args.target_qps
    else:
        min_duration_sec = min_duration_ms / 1000
        target_qps = num_samples / (min_duration_sec * 1.1)
    
    output_dir = Path(args.output_dir) if args.output_dir else (get_project_dir() / 'results' / 'dlrm')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for offload flag
    use_offload = getattr(args, 'offload', False)
    
    _print_config('DLRM RECOMMENDATION', 'dlrm', dataset_path, device,
                  scenario, mode, min_duration_ms, target_qps, batch_size, output_dir,
                  metadata, args)
    
    print("\n" + "=" * 70)
    print("LOADING MODEL AND DATA")
    print("=" * 70)
    
    # =========================================================================
    # DLRM Model Definition
    # =========================================================================
    class DLRM(nn.Module):
        """Deep Learning Recommendation Model"""
        def __init__(
            self,
            embedding_dim: int = 128,
            num_dense_features: int = 13,
            embedding_sizes: list = None,
            bottom_mlp_dims: list = None,
            top_mlp_dims: list = None,
            multi_hot_widths: list = None,
        ):
            super().__init__()
            
            self.embedding_dim = embedding_dim
            self.num_dense_features = num_dense_features
            
            if embedding_sizes is None:
                embedding_sizes = [1000] * 26
            self.embedding_sizes = embedding_sizes
            self.num_sparse_features = len(embedding_sizes)
            
            # Multi-hot widths (how many values per sparse feature)
            self.multi_hot_widths = multi_hot_widths or [1] * len(embedding_sizes)
            
            if bottom_mlp_dims is None:
                bottom_mlp_dims = [num_dense_features, 512, 256, embedding_dim]
            if top_mlp_dims is None:
                interaction_dim = embedding_dim * (self.num_sparse_features + 1)
                top_mlp_dims = [interaction_dim, 512, 256, 1]
            
            # Bottom MLP (processes dense features)
            self.bottom_mlp = self._build_mlp(bottom_mlp_dims)
            
            # Embedding tables for sparse features
            self.embedding_tables = nn.ModuleList([
                nn.EmbeddingBag(size, embedding_dim, mode='mean')
                for size in embedding_sizes
            ])
            
            # Top MLP (processes interaction features)
            interaction_dim = embedding_dim * (self.num_sparse_features + 1)
            top_mlp_dims[0] = interaction_dim
            self.top_mlp = self._build_mlp(top_mlp_dims)
            
            self.sigmoid = nn.Sigmoid()
        
        def _build_mlp(self, dims: list) -> nn.Sequential:
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
            return nn.Sequential(*layers)
        
        def forward(self, dense_features: torch.Tensor, sparse_features_list: list) -> torch.Tensor:
            """
            Args:
                dense_features: [batch, 13] float tensor
                sparse_features_list: List of 26 tensors, each [batch, multi_hot_width] int tensor
            """
            # Process dense features through bottom MLP
            dense_out = self.bottom_mlp(dense_features)  # [batch, embedding_dim]
            
            # Process sparse features through embeddings
            sparse_outs = []
            for i, (emb_table, sparse_feat) in enumerate(zip(self.embedding_tables, sparse_features_list)):
                # Clamp indices to valid range
                max_idx = self.embedding_sizes[i] - 1
                indices = sparse_feat.long().clamp(0, max_idx)
                
                # EmbeddingBag expects flattened indices and offsets
                batch_size = indices.shape[0]
                flat_indices = indices.reshape(-1)
                offsets = torch.arange(0, batch_size * indices.shape[1], indices.shape[1], device=indices.device)
                
                sparse_out = emb_table(flat_indices, offsets)  # [batch, embedding_dim]
                sparse_outs.append(sparse_out)
            
            sparse_out = torch.cat(sparse_outs, dim=1)  # [batch, num_sparse * embedding_dim]
            
            # Move sparse output to same device as dense output (for offload mode)
            if sparse_out.device != dense_out.device:
                sparse_out = sparse_out.to(dense_out.device)
            
            # Concatenate dense and sparse
            interaction = torch.cat([dense_out, sparse_out], dim=1)
            
            # Top MLP
            out = self.top_mlp(interaction)
            return self.sigmoid(out).squeeze(-1)
    
    # =========================================================================
    # Load Data (MLPerf preprocessed format)
    # =========================================================================
    print("Loading DLRM data...")
    
    # Check for MLPerf preprocessed format
    dense_file = data_path / 'day_23_dense.npy'
    sparse_file = data_path / 'day_23_sparse_multi_hot.npz'
    
    if dense_file.exists() and sparse_file.exists():
        print(f"  Loading MLPerf preprocessed format...")
        dense_features = np.load(dense_file)
        print(f"  Dense features: {dense_features.shape}")
        
        # Load sparse features (26 separate arrays)
        sparse_data = np.load(sparse_file, allow_pickle=True)
        sparse_keys = sorted(sparse_data.keys(), key=lambda x: int(x))
        
        # Get multi-hot widths from actual data
        multi_hot_widths = [sparse_data[k].shape[1] for k in sparse_keys]
        print(f"  Sparse features: {len(sparse_keys)} tables, multi-hot widths: {multi_hot_widths[:5]}...{multi_hot_widths[-3:]}")
        
        # Build list of sparse feature arrays
        sparse_features_list = [sparse_data[k] for k in sparse_keys]
        
        num_samples = len(dense_features)
    else:
        print(f"Error: MLPerf preprocessed data not found in {data_path}")
        print(f"  Expected: day_23_dense.npy, day_23_sparse_multi_hot.npz")
        return 1
    
    # Limit samples if requested
    if args.max_samples and args.max_samples < num_samples:
        num_samples = args.max_samples
        dense_features = dense_features[:num_samples]
        sparse_features_list = [sf[:num_samples] for sf in sparse_features_list]
    
    print(f"  Total samples: {num_samples:,}")
    
    # =========================================================================
    # Create Model
    # =========================================================================
    # MLPerf DLRM-v2 embedding sizes (26 tables)
    embedding_sizes = [
        40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63,
        40000000, 3067956, 405282, 10, 2209, 11938, 155, 4,
        976, 14, 40000000, 40000000, 40000000, 590152, 12973, 108, 36
    ]
    
    # For synthetic/small data, use smaller embedding sizes
    if metadata and metadata.get('type') == 'synthetic':
        # Cap embedding sizes for synthetic data
        max_in_data = [sf.max() + 1 for sf in sparse_features_list]
        embedding_sizes = [min(e, max(1000, m)) for e, m in zip(embedding_sizes, max_in_data)]
        print(f"  Using reduced embedding sizes for synthetic data")
    
    print("Creating DLRM model...")
    model = DLRM(
        embedding_dim=128,
        num_dense_features=13,
        embedding_sizes=embedding_sizes,
        multi_hot_widths=multi_hot_widths,
    )
    
    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    emb_params = sum(p.numel() for p in model.embedding_tables.parameters())
    mlp_params = total_params - emb_params
    emb_size_gb = emb_params * 4 / 1e9  # float32
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Embedding parameters: {emb_params:,} ({emb_size_gb:.1f} GB)")
    print(f"  MLP parameters: {mlp_params:,}")
    
    # Move model to device
    if use_offload:
        print("  Using CPU offloading for embeddings...")
        # Keep embeddings on CPU, move MLPs to GPU
        model.bottom_mlp = model.bottom_mlp.to(device)
        model.top_mlp = model.top_mlp.to(device)
        model.sigmoid = model.sigmoid.to(device)
        # Embeddings stay on CPU
        model._use_offload = True
    else:
        if device == 'cuda':
            # Check GPU memory
            if torch.cuda.is_available():
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if emb_size_gb > gpu_mem_gb * 0.8:
                    print(f"\n" + "=" * 70)
                    print("⚠️  WARNING: Model may exceed GPU memory")
                    print("=" * 70)
                    print(f"  Model embeddings: {emb_size_gb:.1f} GB")
                    print(f"  GPU memory:       {gpu_mem_gb:.1f} GB")
                    print(f"\n  💡 SUGGESTION: Use --offload to keep embeddings on CPU:")
                    print(f"\n     python benchmark_new.py -b dlrm --dataset criteo --mlperf --offload")
                    print("=" * 70 + "\n")
        try:
            model = model.to(device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"\n" + "=" * 70)
                print("❌ OUT OF MEMORY: Cannot fit model on GPU")
                print("=" * 70)
                print(f"  Model embeddings: {emb_size_gb:.1f} GB")
                if torch.cuda.is_available():
                    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"  GPU memory:       {gpu_mem_gb:.1f} GB")
                print(f"\n  💡 SOLUTIONS:")
                print(f"\n  1. Use CPU offloading (recommended for DLRM):")
                print(f"     python benchmark_new.py -b dlrm --dataset criteo --mlperf --offload")
                print(f"\n  2. Use smaller batch size:")
                print(f"     python benchmark_new.py -b dlrm --dataset criteo --mlperf --batch-size 512")
                print(f"\n  3. Use synthetic data (smaller embeddings):")
                print(f"     python benchmark_new.py -b dlrm --dataset synthetic --mlperf")
                print("=" * 70)
                return 1
            raise
        model._use_offload = False
    
    model.eval()
    print("Model ready")
    
    # =========================================================================
    # Convert data to tensors
    # =========================================================================
    print("Preparing data tensors...")
    dense_tensor = torch.from_numpy(dense_features).float()
    sparse_tensors = [torch.from_numpy(sf).int() for sf in sparse_features_list]
    
    # Pre-move to GPU if not offloading (for speed)
    if device == 'cuda' and not use_offload:
        try:
            dense_tensor = dense_tensor.cuda()
            sparse_tensors = [st.cuda() for st in sparse_tensors]
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"\n" + "=" * 70)
                print("❌ OUT OF MEMORY: Cannot fit data tensors on GPU")
                print("=" * 70)
                print(f"  Data samples:     {num_samples:,}")
                if torch.cuda.is_available():
                    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                    free_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
                    print(f"  GPU memory:       {gpu_mem_gb:.1f} GB")
                    print(f"  Free after model: {free_mem / 1e9:.1f} GB")
                print(f"\n  💡 SOLUTIONS:")
                print(f"\n  1. Use CPU offloading:")
                print(f"     python benchmark_new.py -b dlrm --dataset criteo --mlperf --offload")
                print(f"\n  2. Limit samples:")
                print(f"     python benchmark_new.py -b dlrm --dataset criteo --mlperf -n 1000000")
                print("=" * 70)
                return 1
            raise
    
    # =========================================================================
    # Metrics tracking
    # =========================================================================
    total_inference_time = [0.0]
    total_samples_processed = [0]
    query_start_time = [None]
    query_end_time = [None]
    
    perf_count = min(num_samples, BENCHMARKS['dlrm']['performance_sample_count'])
    
    # =========================================================================
    # LoadGen callbacks
    # =========================================================================
    def issue_queries(query_samples):
        import time as time_module
        
        if query_start_time[0] is None:
            query_start_time[0] = time_module.perf_counter()
        
        indices = [s.index for s in query_samples]
        response_ids = [s.id for s in query_samples]
        
        all_predictions = []
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            # Get batch data
            batch_dense = torch.stack([dense_tensor[idx] for idx in batch_indices])
            batch_sparse = [torch.stack([sparse_tensors[j][idx] for idx in batch_indices]) for j in range(26)]
            
            # Move to device
            if use_offload:
                batch_dense = batch_dense.to(device)
                # Sparse stays on CPU, will be processed per-embedding
            elif device == 'cuda' and dense_tensor.device.type == 'cpu':
                batch_dense = batch_dense.cuda()
                batch_sparse = [bs.cuda() for bs in batch_sparse]
            
            start_time = time_module.perf_counter()
            with torch.no_grad():
                predictions = model(batch_dense, batch_sparse)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time_module.perf_counter() - start_time
            
            total_inference_time[0] += elapsed
            total_samples_processed[0] += len(batch_indices)
            
            # Convert predictions to binary (threshold 0.5)
            binary_preds = (predictions > 0.5).int()
            all_predictions.extend(binary_preds.cpu().tolist())
        
        responses = []
        for pred, response_id in zip(all_predictions, response_ids):
            response_data = array.array('i', [pred])
            bi = response_data.buffer_info()
            responses.append(lg.QuerySampleResponse(response_id, bi[0], bi[1] * response_data.itemsize))
        
        lg.QuerySamplesComplete(responses)
        query_end_time[0] = time_module.perf_counter()
    
    def flush_queries():
        pass
    
    def load_samples(sample_list):
        pass
    
    def unload_samples(sample_list):
        pass
    
    # =========================================================================
    # Run LoadGen
    # =========================================================================
    qsl = lg.ConstructQSL(num_samples, perf_count, load_samples, unload_samples)
    sut = lg.ConstructSUT(issue_queries, flush_queries)
    
    result = _run_loadgen_test(sut, qsl, scenario, mode, min_duration_ms, target_qps, output_dir, num_samples, args)
    
    # =========================================================================
    # Print detailed metrics
    # =========================================================================
    if total_samples_processed[0] > 0:
        samples_completed = total_samples_processed[0]
        inference_time = total_inference_time[0]
        wall_clock_time = query_end_time[0] - query_start_time[0] if query_end_time[0] else inference_time
        
        # Throughput based on wall-clock time (matches LoadGen)
        samples_per_sec = samples_completed / wall_clock_time if wall_clock_time > 0 else 0
        
        # Pure inference throughput (excludes overhead)
        pure_inference_throughput = samples_completed / inference_time if inference_time > 0 else 0
        
        # Average latency per sample
        avg_latency = wall_clock_time / samples_completed
        
        print("\n" + "=" * 70)
        print("DLRM RECOMMENDATION BENCHMARK DETAILED METRICS")
        print("=" * 70)
        print(f"  Samples Processed:    {samples_completed:,}")
        print(f"  Wall-Clock Time:      {wall_clock_time:.2f} seconds")
        print(f"  Pure Inference Time:  {inference_time:.2f} seconds")
        print(f"  Avg Latency:          {avg_latency*1000:.4f} ms/sample")
        print(f"  Throughput:           {samples_per_sec:,.2f} samples/sec")
        print(f"  (Pure GPU):           {pure_inference_throughput:,.2f} samples/sec")
        if use_offload:
            print(f"  Mode:                 GPU+CPU offload")
        print("=" * 70)
        
        # Performance rating based on wall-clock throughput
        if samples_per_sec >= 100000:
            print("  Performance:          🚀 Excellent (>100K samples/sec)")
        elif samples_per_sec >= 10000:
            print("  Performance:          ✅ Good (>10K samples/sec)")
        elif samples_per_sec >= 1000:
            print("  Performance:          ⚠️  Moderate (>1K samples/sec)")
        else:
            print("  Performance:          🐢 Slow (<1K samples/sec)")
        print("=" * 70)
    
    # Note: LoadGen cleanup (DestroyQSL/DestroySUT) is handled by _run_loadgen_test
    
    return result


# ============================================================================
# Common LoadGen Helper Functions
# ============================================================================

def _print_config(title: str, benchmark: str, dataset_path: str, device: str,
                  scenario: str, mode: str, min_duration_ms: int, target_qps: float,
                  batch_size: int, output_dir: Path, metadata: Optional[Dict],
                  args: argparse.Namespace):
    """Print benchmark configuration."""
    
    print("\n" + "=" * 70)
    print(f"{title} BENCHMARK")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("MODEL CONFIGURATION")
    print("-" * 70)
    print(f"  Model:              {BENCHMARKS[benchmark]['model_name']}")
    print(f"  Task:               {BENCHMARKS[benchmark]['task']}")
    print(f"  Device:             {device}")
    if device == 'cuda' and torch.cuda.is_available():
        print(f"  GPU:                {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version:       {torch.version.cuda}")
    
    print("\n" + "-" * 70)
    print("DATASET CONFIGURATION")
    print("-" * 70)
    print(f"  Dataset:            {Path(dataset_path).name}")
    print(f"  Data path:          {dataset_path}")
    if metadata:
        print(f"  Type:               {metadata.get('type', 'unknown')}")
        print(f"  Total samples:      {metadata.get('samples', 'unknown'):,}")
        print(f"  MLPerf compliant:   {'✓ Yes' if metadata.get('mlperf_compliant') else '✗ No'}")
    
    print("\n" + "-" * 70)
    print("MLPERF LOADGEN CONFIGURATION")
    print("-" * 70)
    print(f"  Scenario:           {scenario}")
    print(f"  Mode:               {mode}")
    print(f"  Min duration:       {min_duration_ms / 1000:.0f}s")
    print(f"  Target QPS:         {target_qps}")
    print(f"  Batch size:         {batch_size}")
    print(f"  Output directory:   {output_dir}")
    
    # Build and display command
    print("\n" + "-" * 70)
    print("COMMAND")
    print("-" * 70)
    cmd_parts = [
        "python benchmark.py",
        f"--benchmark {benchmark}",
        f"--dataset {Path(dataset_path).name}",
        "--mlperf-quick" if args.mlperf_quick else "--mlperf",
        f"--scenario {scenario}",
        f"--mode {mode}",
        f"--target-qps {target_qps}",
        f"--device {device}",
        f"--batch-size {batch_size}",
    ]
    if args.max_samples:
        cmd_parts.append(f"--max-samples {args.max_samples}")
    
    full_cmd = " \\\n    ".join(cmd_parts)
    print(f"  {full_cmd}")


def _run_loadgen_test(sut, qsl, scenario: str, mode: str, min_duration_ms: int,
                      target_qps: float, output_dir: Path, 
                      num_samples: int = 0, args: argparse.Namespace = None,
                      actual_throughput: float = None) -> int:
    """Run LoadGen test with common settings.
    
    Args:
        actual_throughput: If provided (from detailed metrics), use for suggestions
    """
    import time
    
    settings = lg.TestSettings()
    settings.scenario = SCENARIO_MAP[scenario]
    
    if mode == 'accuracy':
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly
    
    settings.min_duration_ms = min_duration_ms
    settings.offline_expected_qps = target_qps
    
    # Calculate expected samples for Offline mode: target_qps * min_duration + 10%
    # But respect user's sample count limit
    expected_samples = int(target_qps * (min_duration_ms / 1000) * 1.1)
    min_query_count = max(1, min(expected_samples, num_samples))
    settings.min_query_count = min_query_count
    
    log_settings = lg.LogSettings()
    log_settings.log_output.outdir = str(output_dir)
    log_settings.log_output.copy_summary_to_stdout = True
    log_settings.enable_trace = False
    
    print("\n" + "=" * 70)
    print("RUNNING MLPERF LOADGEN TEST")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(f"Test completed in {elapsed:.2f} seconds")
    print("=" * 70)
    
    # Calculate actual throughput from LoadGen results
    # LoadGen sends: target_qps * (min_duration_ms / 1000) samples
    samples_sent = int(target_qps * (min_duration_ms / 1000))
    measured_throughput = samples_sent / elapsed if elapsed > 0 else 0
    
    # Provide helpful guidance if test didn't meet duration
    min_duration_sec = min_duration_ms / 1000
    if elapsed < min_duration_sec * 0.99:  # Allow 1% tolerance
        print("\n" + "=" * 70)
        print("⚠️  INVALID RESULT - Test did not meet minimum duration")
        print("=" * 70)
        print(f"\n  Required Duration:    {min_duration_sec:.0f} seconds")
        print(f"  Actual Duration:      {elapsed:.1f} seconds")
        print(f"  Shortfall:            {min_duration_sec - elapsed:.1f} seconds")
        
        print(f"\n  Target QPS:           {target_qps:.1f}")
        print(f"  Measured Throughput:  {measured_throughput:.2f} samples/sec")
        print(f"  Samples Processed:    {samples_sent:,}")
        
        # Calculate recommended target_qps
        # To run for min_duration, we need: samples / throughput >= min_duration
        # samples = target_qps * min_duration
        # So: target_qps >= throughput
        # Add 10% buffer for safety
        recommended_qps = int(measured_throughput * 1.1) + 1
        
        # Also calculate how many samples would be generated
        recommended_samples = int(recommended_qps * min_duration_sec)
        
        print("\n" + "-" * 70)
        print("💡 SUGGESTIONS TO MAKE RESULT VALID:")
        print("-" * 70)
        
        # Get benchmark name from args if available
        benchmark_name = args.benchmark if args and hasattr(args, 'benchmark') else 'BENCHMARK'
        dataset_name = args.dataset if args and hasattr(args, 'dataset') else 'DATASET'
        
        print(f"\n  For Offline scenario, LoadGen generates:")
        print(f"    samples = target_qps × min_duration = {target_qps:.0f} × {min_duration_sec:.0f} = {samples_sent:,}")
        print(f"\n  Your system processes {measured_throughput:.1f} samples/sec, so it finishes")
        print(f"  {samples_sent:,} samples in only {elapsed:.1f}s instead of {min_duration_sec:.0f}s.")
        
        print(f"\n  ✅ RECOMMENDED: Increase --target-qps to match your throughput:")
        print(f"\n     python benchmark_new.py -b {benchmark_name} --dataset {dataset_name} \\")
        print(f"         --mlperf --target-qps {recommended_qps}")
        
        print(f"\n     This will generate {recommended_samples:,} samples, requiring ~{recommended_samples/measured_throughput:.0f}s")
        
        if args and hasattr(args, 'mlperf_quick') and not args.mlperf_quick:
            quick_qps = int(measured_throughput * 1.1) + 1
            print(f"\n  ⚡ ALTERNATIVE: Use quick mode (60s instead of 600s):")
            print(f"\n     python benchmark_new.py -b {benchmark_name} --dataset {dataset_name} \\")
            print(f"         --mlperf-quick --target-qps {quick_qps}")
        
        print("\n" + "=" * 70)
    
    # Cleanup - force garbage collection before LoadGen cleanup to avoid double-free
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)
    
    return 0


# ============================================================================
# Benchmark Dispatcher
# ============================================================================

BENCHMARK_RUNNERS = {
    'bert': run_bert,
    'resnet50': run_resnet50,
    'retinanet': run_retinanet,
    '3dunet': run_3dunet,
    'whisper': run_whisper,
    'sdxl': run_sdxl,
    'gptj': run_gptj,
    'llama': run_llama,
    'mixtral': run_mixtral,
    'dlrm': run_dlrm,
}


def run_benchmark(benchmark: str, dataset_name: str, args: argparse.Namespace) -> int:
    """Run a benchmark with the specified dataset."""
    
    if not LOADGEN_AVAILABLE:
        print("Error: mlperf_loadgen not installed")
        print("Install with: pip install mlperf_loadgen")
        return 1
    
    if benchmark not in BENCHMARKS:
        print(f"Error: Unknown benchmark '{benchmark}'")
        print(f"Available: {', '.join(BENCHMARKS.keys())}")
        return 1
    
    if not dataset_name:
        print(f"Error: --dataset is required")
        print(f"\nAvailable datasets for {benchmark}:")
        datasets = discover_datasets(benchmark)
        if datasets:
            for name, info in datasets.items():
                meta = info.get('metadata', {})
                dtype = meta.get('type', 'unknown')
                samples = meta.get('samples', 0)
                print(f"  - {name}: {dtype}, {samples:,} samples")
        else:
            print("  (none - run data_prepare.py first)")
        return 1
    
    # Determine dataset path
    if args.dataset_dir:
        # Use custom dataset directory: {dataset_dir}/{dataset}
        dataset_path = Path(args.dataset_dir) / dataset_name
    else:
        # Use default data/{benchmark}/{dataset} structure
        dataset_path = get_data_dir() / benchmark / dataset_name
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        if args.dataset_dir:
            print(f"Tip: Check that '{dataset_name}' exists in '{args.dataset_dir}'")
        return 1
    
    metadata = load_metadata(dataset_path)
    
    # Run the appropriate benchmark
    runner = BENCHMARK_RUNNERS.get(benchmark)
    if runner:
        return runner(str(dataset_path), args, metadata)
    else:
        print(f"Error: No runner for benchmark '{benchmark}'")
        return 1


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run MLPerf benchmarks (self-contained)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --benchmark bert --dataset squad --mlperf-quick
  %(prog)s --benchmark resnet50 --dataset imagenet --mlperf-quick --target-qps 2000
  %(prog)s --benchmark 3dunet --dataset kits19 --mlperf-quick --target-qps 20
  %(prog)s --benchmark bert --dataset-dir ../data/bert --dataset synthetic --mlperf-quick
  %(prog)s --list

Available benchmarks:
  bert, resnet50, retinanet, 3dunet, whisper, sdxl, gptj, llama, mixtral, dlrm
"""
    )
    
    parser.add_argument('--benchmark', '-b', type=str, help='Benchmark to run')
    parser.add_argument('--dataset', '-d', type=str, help='Dataset name (subdirectory in data folder)')
    parser.add_argument('--dataset-dir', type=str, help='Custom data directory (default: data/{benchmark})')
    
    parser.add_argument('--mlperf', action='store_true', help='MLPerf LoadGen mode (10min)')
    parser.add_argument('--mlperf-quick', action='store_true', help='MLPerf mode (60s)')
    parser.add_argument('--scenario', type=str, choices=['Offline', 'SingleStream', 'Server', 'MultiStream'],
                       default='Offline', help='MLPerf scenario')
    parser.add_argument('--mode', type=str, choices=['accuracy', 'performance'],
                       default='performance', help='MLPerf test mode')
    parser.add_argument('--target-qps', type=float, help='Target QPS')
    parser.add_argument('--min-duration', type=int, help='Minimum test duration in ms')
    
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'], help='Device (gpu or cpu)')
    parser.add_argument('--max-samples', '-n', type=int, help='Maximum samples')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--output-dir', '-o', type=str, help='Output directory')
    
    # Quantization options for large models (GPT-J, Llama, Mixtral)
    parser.add_argument('--4bit', dest='use_4bit', action='store_true',
                       help='Use 4-bit quantization (requires bitsandbytes)')
    parser.add_argument('--8bit', dest='use_8bit', action='store_true',
                       help='Use 8-bit quantization (requires bitsandbytes)')
    
    # Model-specific options
    parser.add_argument('--offload', action='store_true',
                       help='Enable CPU offloading for large models (SDXL, DLRM embeddings, Llama)')
    parser.add_argument('--num-steps', type=int, default=20,
                       help='Number of diffusion steps for SDXL (default: 20)')
    parser.add_argument('--model', dest='model_variant', type=str,
                       help='Model variant for Llama: llama2-7b, llama2-13b, llama2-70b, llama3-8b, llama3-70b')
    parser.add_argument('--max-tokens', type=int,
                       help='Max new tokens to generate for LLM benchmarks (overrides defaults)')
    parser.add_argument('--warmup-tokens', type=int, default=32,
                       help='Number of tokens to generate during warmup (default: 32, use 4 for quick testing)')
    
    parser.add_argument('--list', '-l', action='store_true', help='List datasets')
    parser.add_argument('--info', '-i', action='store_true', help='Show benchmark info')
    
    args = parser.parse_args()
    
    if args.list:
        list_all_datasets()
        return 0
    
    if args.info:
        if not args.benchmark:
            print("\nAvailable Benchmarks:")
            print("-" * 70)
            for name, config in BENCHMARKS.items():
                print(f"  {name:<12} {config['name']}")
                print(f"               Task: {config['task']}")
                print()
            return 0
        else:
            if args.benchmark not in BENCHMARKS:
                print(f"Error: Unknown benchmark '{args.benchmark}'")
                return 1
            config = BENCHMARKS[args.benchmark]
            print(f"\nBenchmark: {config['name']}")
            print("-" * 50)
            print(f"  Task:            {config['task']}")
            print(f"  Default model:   {config['model_name']}")
            print(f"  Default dataset: {config['default_dataset']}")
            datasets = discover_datasets(args.benchmark)
            print(f"\nAvailable datasets:")
            if datasets:
                for name, info in datasets.items():
                    meta = info.get('metadata', {})
                    print(f"  - {name}: {meta.get('type', 'unknown')}, {meta.get('samples', 0):,} samples")
            else:
                print("  (none)")
            return 0
    
    if not args.benchmark:
        parser.print_help()
        return 1
    
    if not (args.mlperf or args.mlperf_quick) and not args.min_duration:
        print("Note: Running in MLPerf quick mode (60s). Use --mlperf for full 10min test.")
        args.mlperf_quick = True
    
    # Set default max_samples if not provided
    if not args.max_samples:
        if args.mlperf_quick:
            args.max_samples = 50  # Quick mode default
        elif not args.mlperf:
            args.max_samples = 100  # Default for testing
        # For --mlperf, don't set default - use all available samples
    
    return run_benchmark(args.benchmark, args.dataset, args)


if __name__ == '__main__':
    sys.exit(main())
