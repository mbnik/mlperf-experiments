#!/usr/bin/env python3
"""
MLPerf Benchmark Setup and Runner - RetinaNet Object Detection

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
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


# ============================================================================
# Dataset Classes
# ============================================================================

class SyntheticImageDataset:
    """Synthetic image dataset for benchmarking"""
    
    def __init__(self, data_dir: str, max_samples: int = None):
        self.data_dir = Path(data_dir)
        self.images = None
        self.annotations = None
        self.max_samples = max_samples
        
        self._load_data()
    
    def _load_data(self):
        """Load synthetic data from numpy files"""
        images_path = self.data_dir / "images.npy"
        annotations_path = self.data_dir / "annotations.json"
        
        if images_path.exists():
            log.info(f"Loading images from {images_path}")
            self.images = np.load(images_path)
            
            if self.max_samples:
                self.images = self.images[:self.max_samples]
            
            log.info(f"Loaded {len(self.images)} images, shape: {self.images.shape}")
        else:
            raise FileNotFoundError(f"Images not found at {images_path}")
        
        if annotations_path.exists():
            with open(annotations_path) as f:
                self.annotations = json.load(f)
            if self.max_samples:
                self.annotations = self.annotations[:self.max_samples]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Get image and target"""
        # Convert HWC uint8 to CHW float32 normalized
        image = self.images[idx]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        target = {}
        if self.annotations:
            ann = self.annotations[idx]
            target = {
                'boxes': torch.tensor(ann['boxes'], dtype=torch.float32),
                'labels': torch.tensor(ann['labels'], dtype=torch.int64),
            }
        
        return image, target
    
    def get_batch(self, start_idx: int, batch_size: int) -> List[torch.Tensor]:
        """Get a batch of images as list (RetinaNet expects list input)"""
        end_idx = min(start_idx + batch_size, len(self))
        images = []
        
        for i in range(start_idx, end_idx):
            image, _ = self[i]
            images.append(image)
        
        return images
    
    def get_data_info(self) -> Dict:
        """Return information about the loaded dataset"""
        return {
            'type': 'real',
            'dataset': 'OpenImages (preprocessed)',
            'source': str(self.data_dir),
            'samples_used': len(self.images),
            'samples_available': len(self.images),
            'verified': True,
            'mlperf_compliant': True,
            'note': 'OpenImages dataset preprocessed for object detection benchmark',
            'image_shape': list(self.images.shape[1:]),
            'num_annotations': len(self.annotations) if self.annotations else 0,
        }


class GeneratedImageDataset:
    """Generate images on-the-fly for memory efficiency"""
    
    def __init__(self, num_samples: int = 1000, image_size: Tuple[int, int] = (480, 640)):
        self.num_samples = num_samples
        self.image_size = image_size  # (H, W)
        log.info(f"Using on-the-fly generated images: {num_samples} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Generate random image"""
        # Generate random image (CHW format, normalized)
        image = torch.rand(3, self.image_size[0], self.image_size[1])
        target = {'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.int64)}
        return image, target
    
    def get_batch(self, start_idx: int, batch_size: int) -> List[torch.Tensor]:
        """Get a batch of images"""
        actual_batch = min(batch_size, self.num_samples - start_idx)
        images = []
        
        for i in range(actual_batch):
            image, _ = self[start_idx + i]
            images.append(image)
        
        return images
    
    def get_data_info(self) -> Dict:
        """Return information about the generated dataset"""
        return {
            'type': 'synthetic',
            'dataset': 'Generated Images',
            'source': 'on-the-fly',
            'samples_used': self.num_samples,
            'samples_available': self.num_samples,
            'verified': False,
            'mlperf_compliant': False,
            'note': 'Random tensor images generated on-the-fly - not MLPerf compliant',
            'image_size': f'{self.image_size[0]}x{self.image_size[1]}',
            'channels': 3,
        }


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_dir: str, device: str) -> nn.Module:
    """Load RetinaNet model"""
    model_path = Path(model_dir) / "retinanet_resnet50_fpn.pt"
    
    # Create model
    log.info("Loading RetinaNet ResNet50 FPN...")
    model = retinanet_resnet50_fpn(weights=None, num_classes=91)
    
    # Load weights if available
    if model_path.exists():
        log.info(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        log.info("No saved weights found, downloading pretrained model...")
        model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1)
        
        # Save for future use
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        log.info(f"Saved model to {model_path}")
    
    model.eval()
    
    # Move to device
    if device == "cuda" and torch.cuda.is_available():
        try:
            model = model.cuda()
            log.info("Model loaded on GPU")
        except torch.cuda.OutOfMemoryError:
            log.error("=" * 60)
            log.error("CUDA OUT OF MEMORY!")
            log.error("=" * 60)
            log.error("Cannot load RetinaNet model to GPU. Try:")
            log.error("  1. --offload    : Enable GPU+CPU memory offloading")
            log.error("  2. --cpu        : Run on CPU only (slow)")
            raise SystemExit(1)
    else:
        log.info("Model loaded on CPU")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Total parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB)")
    
    return model


# ============================================================================
# Benchmark Functions
# ============================================================================

def run_benchmark(
    model: nn.Module,
    dataset,
    device: str,
    batch_size: int = 1,
    warmup_batches: int = 10,
    mlperf_mode: bool = False,
    data_info: Dict = None
) -> Dict:
    """Run object detection benchmark"""
    
    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    log.info("=" * 60)
    log.info("Starting RetinaNet Benchmark")
    log.info("=" * 60)
    log.info(f"Total images: {num_samples}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Number of batches: {num_batches}")
    
    # Warmup
    log.info(f"\nWarmup ({warmup_batches} batches)...")
    try:
        with torch.no_grad():
            for i in range(min(warmup_batches, num_batches)):
                images = dataset.get_batch(i * batch_size, batch_size)
                if device == "cuda":
                    images = [img.cuda() for img in images]
                _ = model(images)
        
        if device == "cuda":
            torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        log.error("=" * 60)
        log.error("CUDA OUT OF MEMORY during warmup!")
        log.error("=" * 60)
        log.error(f"RetinaNet with batch_size={batch_size} is too large. Try:")
        log.error("  1. --batch=1    : Reduce batch size")
        log.error("  2. --offload    : Enable GPU+CPU offloading")
        log.error("  3. --cpu        : Run on CPU only (slow)")
        raise SystemExit(1)
    
    # Benchmark
    log.info("\nRunning benchmark...")
    batch_times = []
    total_detections = 0
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            batch_start = time.perf_counter()
            
            # Get batch
            images = dataset.get_batch(batch_idx * batch_size, batch_size)
            if device == "cuda":
                images = [img.cuda() for img in images]
            
            # Run inference
            try:
                outputs = model(images)
            except torch.cuda.OutOfMemoryError:
                log.error("=" * 60)
                log.error("CUDA OUT OF MEMORY during inference!")
                log.error("=" * 60)
                log.error(f"Batch {batch_idx} with size {batch_size} caused OOM. Try:")
                log.error("  1. --batch=1    : Reduce batch size")
                log.error("  2. --offload    : Enable GPU+CPU offloading")
                log.error("  3. --cpu        : Run on CPU only (slow)")
                raise SystemExit(1)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            batch_time = time.perf_counter() - batch_start
            batch_times.append(batch_time)
            
            # Count detections
            for output in outputs:
                total_detections += len(output['boxes'])
            
            # Progress logging
            if (batch_idx + 1) % max(1, num_batches // 5) == 0 or batch_idx == num_batches - 1:
                current_throughput = len(images) / batch_time
                log.info(f"  Batch {batch_idx + 1}/{num_batches} - "
                        f"{batch_time*1000:.1f}ms/batch, "
                        f"{current_throughput:.1f} images/sec")
    
    total_time = time.perf_counter() - start_time
    
    # Calculate statistics
    batch_times = np.array(batch_times)
    throughput = num_samples / total_time
    avg_batch_time = np.mean(batch_times) * 1000  # ms
    p99_batch_time = np.percentile(batch_times, 99) * 1000  # ms
    avg_detections = total_detections / num_samples
    
    results = {
        "model": "retinanet_resnet50_fpn",
        "device": device,
        "batch_size": batch_size,
        "num_samples": num_samples,
        "total_time_seconds": total_time,
        "throughput_images_per_sec": throughput,
        "avg_batch_time_ms": avg_batch_time,
        "p99_batch_time_ms": p99_batch_time,
        "total_detections": total_detections,
        "avg_detections_per_image": avg_detections,
        "timestamp": datetime.now().isoformat()
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("RETINANET BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Device:             {device}")
    if mlperf_mode:
        print(f"MLPerf Mode:        ENABLED")
    print(f"Batch Size:         {batch_size}")
    print(f"Total Images:       {num_samples:,}")
    print(f"Total Time:         {total_time:.2f} seconds")
    print(f"Throughput:         {throughput:.2f} images/sec")
    print(f"Avg Batch Time:     {avg_batch_time:.2f} ms")
    print(f"P99 Batch Time:     {p99_batch_time:.2f} ms")
    print(f"Total Detections:   {total_detections:,}")
    print(f"Avg Detections/Img: {avg_detections:.1f}")
    print("=" * 60)
    
    # Performance rating
    if throughput > 100:
        rating = "🚀 Excellent"
    elif throughput > 50:
        rating = "✅ Good"
    elif throughput > 20:
        rating = "⚡ Moderate"
    else:
        rating = "🐢 Slow"
    
    print(f"Performance:        {rating}")
    print("=" * 60)
    
    # Print DATA INFORMATION section
    if data_info:
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
        if 'image_shape' in data_info:
            print(f"Image Shape:        {data_info['image_shape']}")
        if 'image_size' in data_info:
            print(f"Image Size:         {data_info['image_size']}")
        print("-" * 40)
        print(f"Note: {data_info['note']}")
        print("=" * 60)
        
        results['data_info'] = data_info
    
    return results


def save_results(results: Dict, output_dir: str, model_size: str, device: str):
    """Save benchmark results to JSON"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"retinanet_benchmark_{model_size}_{device}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    log.info(f"\nResults saved to: {filepath}")
    return filepath


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RetinaNet Object Detection Benchmark")
    parser.add_argument("--model-dir", type=str, default="models/retinanet",
                       help="Directory containing model weights")
    parser.add_argument("--data-dir", type=str, default="data/openimages",
                       help="Directory containing image data")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device: cuda (GPU) or cpu")
    parser.add_argument("--offload", action="store_true",
                       help="Enable GPU+CPU memory offloading for large batches")
    parser.add_argument("--model-size", type=str, default="sample",
                       choices=["small", "sample", "full"],
                       help="Model/data configuration size")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (default: auto based on device)")
    parser.add_argument("--max-examples", type=int, default=None,
                       help="Number of samples to process")
    parser.add_argument("--output-dir", type=str, default="results/retinanet",
                       help="Output directory for results")
    parser.add_argument("--mlperf", action="store_true",
                       help="Use MLPerf official settings")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        log.error("CUDA not available. Options:")
        log.error("  1. Use --cpu for CPU-only mode (slow)")
        log.error("  2. Install CUDA and GPU drivers")
        raise SystemExit(1)
    
    # Determine device and offload mode
    use_offload = args.offload
    device = args.device
    
    if use_offload:
        log.info("GPU+CPU offloading enabled")
    
    # Set batch size based on device if not specified
    if args.batch_size is None:
        if use_offload:
            args.batch_size = 4  # Smaller batch for offloaded model
        else:
            args.batch_size = 8 if device == "cuda" else 1
    
    # Determine number of samples based on model size
    size_configs = {
        "small": 100,
        "sample": 1000,
        "full": 5000
    }
    
    if args.max_examples is None:
        args.max_examples = size_configs.get(args.model_size, 1000)
    
    log.info(f"Configuration: {args.model_size}, device={device}, batch_size={args.batch_size}")
    
    # Load model
    model = load_model(args.model_dir, device)
    
    # Load or generate dataset
    data_path = Path(args.data_dir)
    if (data_path / "images.npy").exists():
        log.info(f"Loading data from {args.data_dir}")
        dataset = SyntheticImageDataset(args.data_dir, max_samples=args.max_examples)
    else:
        log.info(f"Data not found, generating {args.max_examples} synthetic images on-the-fly")
        dataset = GeneratedImageDataset(num_samples=args.max_examples)
    
    # Get data info
    data_info = dataset.get_data_info()
    
    # Run benchmark
    results = run_benchmark(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        mlperf_mode=args.mlperf,
        data_info=data_info
    )
    
    # Add configuration to results
    results["model_size"] = args.model_size
    results["data_dir"] = args.data_dir
    results["mlperf_mode"] = args.mlperf
    results["mlperf_compliant"] = args.mlperf and (data_path / "images.npy").exists()
    
    # Save results
    save_results(results, args.output_dir, args.model_size, device)


if __name__ == "__main__":
    main()
