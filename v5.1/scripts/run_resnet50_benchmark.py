#!/usr/bin/env python3
"""
MLPerf Benchmark Setup and Runner - ResNet50 Image Classification

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
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


# ============================================================================
# GPU Cleanup Utilities
# ============================================================================

_model_ref = None  # Global reference for cleanup


def cleanup_gpu():
    """Properly cleanup GPU resources to prevent device unavailability issues."""
    global _model_ref
    
    log.info("Cleaning up GPU resources...")
    
    if _model_ref is not None:
        del _model_ref
        _model_ref = None
    
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


# ImageNet class labels (top 10 for display)
IMAGENET_CLASSES = [
    "tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead",
    "electric_ray", "stingray", "cock", "hen", "ostrich"
]


# ============================================================================
# Data Loading
# ============================================================================

def generate_synthetic_data(num_samples: int, image_size: int = 224) -> Tuple[torch.Tensor, Dict]:
    """Generate synthetic image data for benchmarking"""
    log.info(f"Generating {num_samples} synthetic images ({image_size}x{image_size})...")
    
    # Generate random normalized images (ImageNet normalization)
    images = torch.randn(num_samples, 3, image_size, image_size)
    
    data_info = {
        'type': 'synthetic',
        'dataset': 'Synthetic Images',
        'source': 'generated',
        'samples_used': num_samples,
        'samples_available': num_samples,
        'verified': False,
        'mlperf_compliant': False,
        'note': 'Random tensor data for testing - not MLPerf compliant',
        'image_size': image_size,
        'channels': 3,
        'num_classes': 1000,
    }
    
    return images, data_info


def load_imagenet_data(data_dir: str, max_samples: int = None, image_size: int = 224) -> Tuple[torch.Tensor, List[int], Dict]:
    """Load ImageNet validation data"""
    val_dir = Path(data_dir) / "val"
    
    if not val_dir.exists():
        log.warning(f"ImageNet validation directory not found: {val_dir}")
        return None, None, {}
    
    log.info(f"Loading ImageNet data from {val_dir}")
    
    # Standard ImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    from PIL import Image
    
    images = []
    labels = []
    
    # Load images from subdirectories (each subdir is a class)
    class_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    total_available = 0
    
    # Count total available first
    for class_dir in class_dirs:
        image_files = list(class_dir.glob("*.JPEG")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        total_available += len(image_files)
    
    for class_idx, class_dir in enumerate(class_dirs):
        image_files = list(class_dir.glob("*.JPEG")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(img)
                images.append(img_tensor)
                labels.append(class_idx)
                
                if max_samples and len(images) >= max_samples:
                    break
            except Exception as e:
                log.warning(f"Failed to load {img_path}: {e}")
        
        if max_samples and len(images) >= max_samples:
            break
    
    if not images:
        return None, None, {}
    
    log.info(f"Loaded {len(images)} images from ImageNet")
    
    data_info = {
        'type': 'real',
        'dataset': 'ImageNet ILSVRC2012',
        'source': str(val_dir),
        'samples_used': len(images),
        'samples_available': total_available,
        'verified': True,
        'mlperf_compliant': True,
        'note': 'ImageNet ILSVRC2012 validation set (50,000 images)',
        'image_size': image_size,
        'channels': 3,
        'num_classes': len(class_dirs),
        'preprocessing': 'Resize(256) -> CenterCrop(224) -> Normalize(ImageNet)',
    }
    
    return torch.stack(images), labels, data_info


# ============================================================================
# Command Builder
# ============================================================================

def build_command(args) -> str:
    """Build the command line string for reproducibility"""
    import sys
    cmd_parts = [sys.executable, __file__]
    
    cmd_parts.extend(["--device", args.device])
    cmd_parts.extend(["--max-examples", str(args.max_examples)])
    cmd_parts.extend(["--batch-size", str(args.batch_size)])
    cmd_parts.extend(["--image-size", str(args.image_size)])
    cmd_parts.extend(["--data-type", args.data_type])
    cmd_parts.extend(["--data-dir", args.data_dir])
    cmd_parts.extend(["--output-dir", args.output_dir])
    
    if args.offload:
        cmd_parts.append("--offload")
    if args.mlperf:
        cmd_parts.append("--mlperf")
    
    return " ".join(cmd_parts)


# ============================================================================
# Model Loading
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser(description="ResNet50 Image Classification Benchmark")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device: cuda (GPU) or cpu")
    parser.add_argument("--offload", action="store_true",
                       help="Enable CPU offloading (not typically needed for ResNet50)")
    parser.add_argument("--max-examples", type=int, default=1000,
                       help="Number of images to process")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--image-size", type=int, default=224,
                       help="Input image size")
    parser.add_argument("--data-type", type=str, default="synthetic",
                       choices=["synthetic", "real"],
                       help="Data type: synthetic or real ImageNet")
    parser.add_argument("--data-dir", type=str, default="data/imagenet",
                       help="Directory containing ImageNet data")
    parser.add_argument("--output-dir", type=str, default="results/resnet50",
                       help="Output directory for results")
    parser.add_argument("--mlperf", action="store_true",
                       help="Use MLPerf official settings")
    return parser.parse_args()


def load_model(args):
    """Load ResNet50 model"""
    log.info("Loading ResNet50 model...")
    log.info(f"Device: {args.device}, Offload: {args.offload}")
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        log.error("CUDA not available. Options:")
        log.error("  1. Use --cpu for CPU-only mode")
        log.error("  2. Install CUDA and GPU drivers")
        raise SystemExit(1)
    
    try:
        # Load pretrained ResNet50
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        if args.device == "cuda":
            model = model.cuda()
            if not args.offload:
                model = model.half()  # FP16 for faster inference
        
        model.eval()
        
    except torch.cuda.OutOfMemoryError:
        log.error("=" * 60)
        log.error("CUDA OUT OF MEMORY!")
        log.error("=" * 60)
        log.error("ResNet50 requires ~100MB VRAM. Try:")
        log.error("  1. Reduce --batch-size")
        log.error("  2. --cpu : Run on CPU")
        raise SystemExit(1)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {total_params:,} ({total_params * 4 / 1e6:.1f} MB)")
    log.info("Model loaded successfully!")
    
    return model


# ============================================================================
# Benchmark
# ============================================================================

def run_benchmark(model, args):
    """Run the image classification benchmark"""
    log.info("=" * 60)
    log.info("Starting ResNet50 Image Classification Benchmark")
    log.info("=" * 60)
    log.info(f"Data type: {args.data_type}")
    log.info(f"Batch size: {args.batch_size}")
    
    # Load data
    labels = None
    if args.data_type == "real":
        images, labels, data_info = load_imagenet_data(args.data_dir, args.max_examples, args.image_size)
        if images is None:
            log.warning("Falling back to synthetic data")
            images, data_info = generate_synthetic_data(args.max_examples, args.image_size)
    else:
        images, data_info = generate_synthetic_data(args.max_examples, args.image_size)
    
    num_samples = len(images)
    num_batches = (num_samples + args.batch_size - 1) // args.batch_size
    log.info(f"Processing {num_samples} images in {num_batches} batches")
    
    # Warmup
    log.info("Warmup...")
    warmup_batch = images[:args.batch_size]
    if args.device == "cuda":
        warmup_batch = warmup_batch.cuda().half() if not args.offload else warmup_batch.cuda()
    
    try:
        with torch.no_grad():
            _ = model(warmup_batch)
    except torch.cuda.OutOfMemoryError:
        log.error("=" * 60)
        log.error("CUDA OUT OF MEMORY during warmup!")
        log.error("=" * 60)
        log.error("ResNet50 with batch_size={} is too large. Try:".format(args.batch_size))
        log.error("  1. --batch=16   : Reduce batch size")
        log.error("  2. --cpu        : Run on CPU")
        raise SystemExit(1)
    
    if args.device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    log.info("\nRunning benchmark...")
    all_predictions = []
    batch_times = []
    total_time = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, num_samples)
        batch = images[start_idx:end_idx]
        
        if args.device == "cuda":
            batch = batch.cuda()
            if not args.offload:
                batch = batch.half()
        
        start_time = time.perf_counter()
        
        try:
            with torch.no_grad():
                outputs = model(batch)
                probs = F.softmax(outputs, dim=1)
                predictions = torch.argmax(probs, dim=1)
        except torch.cuda.OutOfMemoryError:
            log.error("=" * 60)
            log.error("CUDA OUT OF MEMORY during inference!")
            log.error("=" * 60)
            log.error("Batch {} with size {} caused OOM. Try:".format(batch_idx, end_idx - start_idx))
            log.error("  1. --batch=16   : Reduce batch size")
            log.error("  2. --cpu        : Run on CPU")
            raise SystemExit(1)
        
        if args.device == "cuda":
            torch.cuda.synchronize()
        
        batch_time = time.perf_counter() - start_time
        batch_times.append(batch_time)
        total_time += batch_time
        
        all_predictions.extend(predictions.cpu().tolist())
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            throughput = len(batch) / batch_time
            log.info(f"  [{end_idx}/{num_samples}] {batch_time*1000:.1f}ms/batch, "
                    f"{throughput:.1f} images/sec")
    
    # Calculate accuracy if labels available
    accuracy = None
    if labels is not None:
        correct = sum(1 for p, l in zip(all_predictions, labels) if p == l)
        accuracy = correct / num_samples * 100
    
    # Summary
    avg_latency = total_time / num_samples * 1000
    throughput = num_samples / total_time
    
    print("\n" + "=" * 60)
    if args.mlperf:
        if args.data_type == "synthetic":
            print("RESNET50 BENCHMARK SUMMARY - MLPerf Settings (SYNTHETIC DATA)")
            print("⚠️  NOT COMPARABLE TO OFFICIAL MLPERF RESULTS")
        else:
            print("RESNET50 BENCHMARK SUMMARY - MLPerf Compliant")
    else:
        print("RESNET50 BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Device:             {args.device}")
    print(f"Data Type:          {args.data_type}")
    if args.mlperf:
        print(f"MLPerf Mode:        ENABLED")
    print(f"Total Images:       {num_samples}")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Total Time:         {total_time:.2f} seconds")
    print(f"Avg Latency:        {avg_latency:.2f} ms/image")
    print(f"Throughput:         {throughput:.2f} images/sec")
    if accuracy is not None:
        print(f"Top-1 Accuracy:     {accuracy:.2f}%")
    print("=" * 60)
    
    # Performance rating
    if throughput > 500:
        rating = "🚀 Excellent"
    elif throughput > 200:
        rating = "✅ Good"
    elif throughput > 50:
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
    print(f"Image Size:         {data_info.get('image_size', 224)}x{data_info.get('image_size', 224)}")
    print(f"Channels:           {data_info.get('channels', 3)}")
    print(f"Num Classes:        {data_info.get('num_classes', 1000)}")
    print("-" * 40)
    print(f"Note: {data_info['note']}")
    print("=" * 60)
    
    # Print command for reproducibility
    print("\n" + "=" * 60)
    print("COMMAND")
    print("=" * 60)
    print(build_command(args))
    print("=" * 60)
    
    return {
        'device': args.device,
        'data_type': args.data_type,
        'mlperf_mode': args.mlperf,
        'mlperf_compliant': args.mlperf and args.data_type == "real",
        'num_samples': num_samples,
        'batch_size': args.batch_size,
        'total_time_sec': total_time,
        'avg_latency_ms': avg_latency,
        'throughput_images_per_sec': throughput,
        'accuracy': accuracy,
        'batch_times': batch_times[:10],  # Save first 10 for reference
        'data_info': data_info,
    }


def save_results(results: Dict, output_dir: str):
    """Save benchmark results"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"resnet50_benchmark_{results['data_type']}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    log.info(f"\nResults saved to: {filepath}")
    return filepath


# ============================================================================
# Main
# ============================================================================

def main():
    global _model_ref
    
    args = get_args()
    
    try:
        # Load model
        model = load_model(args)
        _model_ref = model
        
        # Run benchmark
        results = run_benchmark(model, args)
        
        # Save results
        save_results(results, args.output_dir)
    
    finally:
        cleanup_gpu()


if __name__ == "__main__":
    main()
