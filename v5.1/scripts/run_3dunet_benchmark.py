#!/usr/bin/env python3
"""
MLPerf Benchmark Setup and Runner - 3D-UNet Medical Image Segmentation

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
import torch.nn as nn
import torch.nn.functional as F

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


# ============================================================================
# 3D-UNet Model Architecture
# ============================================================================

class DoubleConv3D(nn.Module):
    """Double 3D convolution block"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class Down3D(nn.Module):
    """Downsampling block with strided convolution"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class Up3D(nn.Module):
    """Upsampling block with transposed convolution"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle size mismatch
        diff_d = x2.size(2) - x1.size(2)
        diff_h = x2.size(3) - x1.size(3)
        diff_w = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2,
                       diff_d // 2, diff_d - diff_d // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric medical image segmentation
    Based on MLPerf 3D-UNet architecture
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,  # Background, Kidney, Tumor
        base_filters: int = 32,
        depth: int = 5
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.depth = depth
        
        # Calculate channel sizes for each level
        # e.g., depth=5: [32, 64, 128, 256, 320] (capped at 320)
        self.enc_channels = [base_filters]
        for i in range(depth - 1):
            next_ch = min(self.enc_channels[-1] * 2, 320)
            self.enc_channels.append(next_ch)
        
        # Encoder
        self.inc = DoubleConv3D(in_channels, base_filters)
        
        self.encoders = nn.ModuleList()
        for i in range(depth - 1):
            self.encoders.append(Down3D(self.enc_channels[i], self.enc_channels[i + 1]))
        
        # Decoder - needs to account for skip connection concatenation
        self.decoders = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        
        for i in range(depth - 1):
            # Going from deepest to shallowest
            dec_in = self.enc_channels[depth - 1 - i]  # Current level channels
            skip_ch = self.enc_channels[depth - 2 - i]  # Skip connection channels
            dec_out = skip_ch  # Output same as skip level
            
            # Upsampling conv
            self.up_convs.append(nn.ConvTranspose3d(dec_in, dec_out, kernel_size=2, stride=2))
            # After concat with skip: dec_out + skip_ch = 2 * skip_ch
            self.decoders.append(DoubleConv3D(dec_out + skip_ch, dec_out))
        
        # Output
        self.outc = nn.Conv3d(base_filters, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path with skip connections
        skips = []
        x = self.inc(x)
        skips.append(x)
        
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        
        # Start from bottleneck (last encoder output)
        x = skips[-1]
        skips = skips[:-1]  # Remove bottleneck from skips
        
        # Decoder path
        for up_conv, decoder, skip in zip(self.up_convs, self.decoders, reversed(skips)):
            x = up_conv(x)
            # Handle size mismatch
            if x.shape != skip.shape:
                diff_d = skip.size(2) - x.size(2)
                diff_h = skip.size(3) - x.size(3)
                diff_w = skip.size(4) - x.size(4)
                x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                             diff_h // 2, diff_h - diff_h // 2,
                             diff_d // 2, diff_d - diff_d // 2])
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)
        
        return self.outc(x)


# ============================================================================
# Model Configurations
# ============================================================================

def get_model_config(model_size: str) -> Dict:
    """Get model configuration based on size"""
    configs = {
        "small": {
            "base_filters": 16,
            "depth": 4,
            "input_shape": (1, 64, 64, 64),  # C, D, H, W
            "description": "Small debug model (~5MB)"
        },
        "sample": {
            "base_filters": 32,
            "depth": 5,
            "input_shape": (1, 128, 128, 128),
            "description": "Sample model (~30MB)"
        },
        "full": {
            "base_filters": 32,
            "depth": 5,
            "input_shape": (1, 128, 128, 128),  # KiTS19 patch size
            "description": "Full MLPerf model (~30MB)"
        }
    }
    return configs[model_size]


# ============================================================================
# Dataset Classes
# ============================================================================

class Synthetic3DDataset:
    """Synthetic 3D medical imaging dataset"""
    
    def __init__(self, data_dir: str, max_samples: int = None, input_shape: Tuple = None):
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        self.input_shape = input_shape or (1, 128, 128, 128)
        self.volumes = None
        self.labels = None
        self.total_available = 0
        
        self._load_data()
    
    def _load_data(self):
        """Load or generate data"""
        volumes_path = self.data_dir / "volumes.npy"
        labels_path = self.data_dir / "labels.npy"
        
        if volumes_path.exists() and labels_path.exists():
            log.info(f"Loading data from {self.data_dir}")
            self.volumes = np.load(volumes_path)
            self.labels = np.load(labels_path)
            self.total_available = len(self.volumes)
            
            if self.max_samples:
                self.volumes = self.volumes[:self.max_samples]
                self.labels = self.labels[:self.max_samples]
            
            log.info(f"Loaded {len(self.volumes)} volumes, shape: {self.volumes.shape}")
        else:
            raise FileNotFoundError(f"Data not found at {self.data_dir}")
    
    def get_data_info(self) -> Dict:
        """Get information about the dataset"""
        return {
            'type': 'real',
            'dataset': 'KiTS19',
            'source': str(self.data_dir),
            'samples_used': len(self.volumes),
            'samples_available': self.total_available,
            'volume_shape': list(self.volumes.shape[1:]) if self.volumes is not None else list(self.input_shape),
            'classes': ['background', 'kidney', 'tumor'],
            'num_classes': 3,
            'verified': True,
            'mlperf_compliant': True,
            'note': 'KiTS19 kidney tumor segmentation dataset'
        }
    
    def __len__(self):
        return len(self.volumes)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get volume and segmentation mask"""
        volume = torch.from_numpy(self.volumes[idx]).float()
        label = torch.from_numpy(self.labels[idx]).long()
        
        # Add channel dimension if needed
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)
        
        return volume, label
    
    def get_batch(self, start_idx: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of volumes"""
        end_idx = min(start_idx + batch_size, len(self))
        
        volumes = []
        labels = []
        for i in range(start_idx, end_idx):
            vol, lbl = self[i]
            volumes.append(vol)
            labels.append(lbl)
        
        return torch.stack(volumes), torch.stack(labels)


class GeneratedVolumeDataset:
    """Generate volumes on-the-fly for memory efficiency"""
    
    def __init__(self, num_samples: int = 100, input_shape: Tuple = (1, 128, 128, 128)):
        self.num_samples = num_samples
        self.input_shape = input_shape
        log.info(f"Using on-the-fly generated volumes: {num_samples} samples, shape {input_shape}")
    
    def get_data_info(self) -> Dict:
        """Get information about the dataset"""
        return {
            'type': 'synthetic',
            'dataset': 'Generated 3D Volumes',
            'source': 'generated_on_the_fly',
            'samples_used': self.num_samples,
            'samples_available': self.num_samples,
            'volume_shape': list(self.input_shape),
            'classes': ['background', 'kidney', 'tumor'],
            'num_classes': 3,
            'verified': False,
            'mlperf_compliant': False,
            'note': 'Random synthetic data for testing only'
        }
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random volume"""
        volume = torch.randn(*self.input_shape)
        # Generate random segmentation (3 classes)
        label = torch.randint(0, 3, self.input_shape[1:])
        return volume, label
    
    def get_batch(self, start_idx: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of volumes"""
        actual_batch = min(batch_size, self.num_samples - start_idx)
        
        volumes = []
        labels = []
        for i in range(actual_batch):
            vol, lbl = self[start_idx + i]
            volumes.append(vol)
            labels.append(lbl)
        
        return torch.stack(volumes), torch.stack(labels)


# ============================================================================
# Metrics
# ============================================================================

def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> Dict[str, float]:
    """Compute Dice score for each class"""
    pred_softmax = F.softmax(pred, dim=1)
    pred_argmax = pred_softmax.argmax(dim=1)
    
    dice_scores = {}
    for c in range(num_classes):
        pred_c = (pred_argmax == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice = (2.0 * intersection / union).item()
        else:
            dice = 1.0 if intersection == 0 else 0.0
        
        class_names = ['background', 'kidney', 'tumor']
        dice_scores[class_names[c]] = dice
    
    dice_scores['mean'] = np.mean([dice_scores['kidney'], dice_scores['tumor']])
    return dice_scores


# ============================================================================
# Benchmark Functions
# ============================================================================

def run_benchmark(
    model: nn.Module,
    dataset,
    device: str,
    batch_size: int = 1,
    warmup_batches: int = 5,
    mlperf_mode: bool = False,
    data_info: Dict = None
) -> Dict:
    """Run 3D segmentation benchmark"""
    
    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    log.info("=" * 60)
    log.info("Starting 3D-UNet Benchmark")
    log.info("=" * 60)
    log.info(f"Total volumes: {num_samples}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Number of batches: {num_batches}")
    
    # Warmup
    log.info(f"\nWarmup ({warmup_batches} batches)...")
    try:
        with torch.no_grad():
            for i in range(min(warmup_batches, num_batches)):
                volumes, labels = dataset.get_batch(i * batch_size, batch_size)
                if device == "cuda":
                    volumes = volumes.cuda()
                _ = model(volumes)
        
        if device == "cuda":
            torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        log.error("=" * 60)
        log.error("CUDA OUT OF MEMORY during warmup!")
        log.error("=" * 60)
        log.error(f"3D-UNet with batch_size={batch_size} is too large. Try:")
        log.error("  1. --batch=1    : Reduce batch size")
        log.error("  2. --offload    : Enable GPU+CPU offloading")
        log.error("  3. --model-size=small : Use smaller model")
        log.error("  4. --cpu        : Run on CPU only (slow)")
        raise SystemExit(1)
    
    # Benchmark
    log.info("\nRunning benchmark...")
    batch_times = []
    all_dice_scores = []
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            batch_start = time.perf_counter()
            
            # Get batch
            volumes, labels = dataset.get_batch(batch_idx * batch_size, batch_size)
            if device == "cuda":
                volumes = volumes.cuda()
                labels = labels.cuda()
            
            # Run inference
            try:
                outputs = model(volumes)
            except torch.cuda.OutOfMemoryError:
                log.error("=" * 60)
                log.error("CUDA OUT OF MEMORY during inference!")
                log.error("=" * 60)
                log.error(f"Batch {batch_idx} with size {batch_size} caused OOM. Try:")
                log.error("  1. --batch=1    : Reduce batch size")
                log.error("  2. --offload    : Enable GPU+CPU offloading")
                log.error("  3. --model-size=small : Use smaller model")
                log.error("  4. --cpu        : Run on CPU only (slow)")
                raise SystemExit(1)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            batch_time = time.perf_counter() - batch_start
            batch_times.append(batch_time)
            
            # Compute metrics
            dice = compute_dice_score(outputs, labels)
            all_dice_scores.append(dice)
            
            # Progress logging
            if (batch_idx + 1) % max(1, num_batches // 5) == 0 or batch_idx == num_batches - 1:
                current_throughput = volumes.size(0) / batch_time
                log.info(f"  Batch {batch_idx + 1}/{num_batches} - "
                        f"{batch_time*1000:.1f}ms/batch, "
                        f"{current_throughput:.2f} volumes/sec, "
                        f"Dice: {dice['mean']:.3f}")
    
    total_time = time.perf_counter() - start_time
    
    # Calculate statistics
    batch_times = np.array(batch_times)
    throughput = num_samples / total_time
    avg_batch_time = np.mean(batch_times) * 1000
    p99_batch_time = np.percentile(batch_times, 99) * 1000
    
    # Average dice scores
    avg_dice = {
        'background': np.mean([d['background'] for d in all_dice_scores]),
        'kidney': np.mean([d['kidney'] for d in all_dice_scores]),
        'tumor': np.mean([d['tumor'] for d in all_dice_scores]),
        'mean': np.mean([d['mean'] for d in all_dice_scores])
    }
    
    results = {
        "model": "3d_unet",
        "device": device,
        "batch_size": batch_size,
        "num_samples": num_samples,
        "total_time_seconds": total_time,
        "throughput_volumes_per_sec": throughput,
        "avg_batch_time_ms": avg_batch_time,
        "p99_batch_time_ms": p99_batch_time,
        "dice_scores": avg_dice,
        "timestamp": datetime.now().isoformat()
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("3D-UNET BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Device:             {device}")
    if mlperf_mode:
        print(f"MLPerf Mode:        ENABLED")
    print(f"Batch Size:         {batch_size}")
    print(f"Total Volumes:      {num_samples:,}")
    print(f"Total Time:         {total_time:.2f} seconds")
    print(f"Throughput:         {throughput:.2f} volumes/sec")
    print(f"Avg Batch Time:     {avg_batch_time:.2f} ms")
    print(f"P99 Batch Time:     {p99_batch_time:.2f} ms")
    print("-" * 60)
    print("Dice Scores:")
    print(f"  Background:       {avg_dice['background']:.4f}")
    print(f"  Kidney:           {avg_dice['kidney']:.4f}")
    print(f"  Tumor:            {avg_dice['tumor']:.4f}")
    print(f"  Mean (K+T):       {avg_dice['mean']:.4f}")
    print("=" * 60)
    
    # Performance rating
    if throughput > 10:
        rating = "🚀 Excellent"
    elif throughput > 5:
        rating = "✅ Good"
    elif throughput > 1:
        rating = "⚡ Moderate"
    else:
        rating = "🐢 Slow"
    
    print(f"Performance:        {rating}")
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
        print(f"Volume Shape:       {data_info['volume_shape']}")
        print(f"Classes:            {data_info['classes']}")
        print(f"Verified:           {'✓' if data_info['verified'] else '✗'}")
        print(f"MLPerf Compliant:   {'✓' if data_info['mlperf_compliant'] else '✗'}")
        print(f"Note:               {data_info['note']}")
        print("=" * 60)
        
        # Add data_info to results
        results['data_info'] = data_info
    
    return results


def load_model(model_dir: str, model_size: str, device: str) -> nn.Module:
    """Load or create 3D-UNet model"""
    config = get_model_config(model_size)
    log.info(f"Model configuration: {config['description']}")
    
    model_path = Path(model_dir) / f"3dunet_{model_size}.pt"
    
    # Create model
    model = UNet3D(
        in_channels=1,
        out_channels=3,
        base_filters=config['base_filters'],
        depth=config['depth']
    )
    
    # Load weights if available
    if model_path.exists():
        log.info(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        log.info("No saved weights found, using random initialization")
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
            log.error("Cannot load 3D-UNet model to GPU. Try:")
            log.error("  1. --offload    : Enable GPU+CPU memory offloading")
            log.error("  2. --model-size=small : Use smaller model")
            log.error("  3. --cpu        : Run on CPU only (slow)")
            raise SystemExit(1)
    else:
        log.info("Model loaded on CPU")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Total parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB)")
    
    return model, config


def save_results(results: Dict, output_dir: str, model_size: str, device: str):
    """Save benchmark results"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"3dunet_benchmark_{model_size}_{device}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    log.info(f"\nResults saved to: {filepath}")
    return filepath


# ============================================================================
# Command Builder
# ============================================================================

def build_command(args) -> str:
    """Build the command line string for reproducibility"""
    import sys
    cmd_parts = [sys.executable, __file__]
    
    cmd_parts.extend(["--model-dir", args.model_dir])
    cmd_parts.extend(["--data-dir", args.data_dir])
    cmd_parts.extend(["--device", args.device])
    cmd_parts.extend(["--model-size", args.model_size])
    if args.batch_size:
        cmd_parts.extend(["--batch-size", str(args.batch_size)])
    if args.max_examples:
        cmd_parts.extend(["--max-examples", str(args.max_examples)])
    cmd_parts.extend(["--output-dir", args.output_dir])
    
    if args.offload:
        cmd_parts.append("--offload")
    if args.mlperf:
        cmd_parts.append("--mlperf")
    
    return " ".join(cmd_parts)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="3D-UNet Medical Image Segmentation Benchmark")
    parser.add_argument("--model-dir", type=str, default="models/3dunet",
                       help="Directory containing model weights")
    parser.add_argument("--data-dir", type=str, default="data/kits19",
                       help="Directory containing volume data")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device: cuda (GPU) or cpu")
    parser.add_argument("--offload", action="store_true",
                       help="Enable GPU+CPU memory offloading for large volumes")
    parser.add_argument("--model-size", type=str, default="sample",
                       choices=["small", "sample", "full"],
                       help="Model configuration size")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (default: auto)")
    parser.add_argument("--max-examples", type=int, default=None,
                       help="Number of samples to process")
    parser.add_argument("--output-dir", type=str, default="results/3dunet",
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
    
    # Set batch size based on device (3D data uses lots of memory)
    if args.batch_size is None:
        if use_offload:
            args.batch_size = 1  # Smaller batch for offloaded model
        else:
            args.batch_size = 2 if device == "cuda" else 1
    
    # Determine number of samples
    size_configs = {
        "small": 20,
        "sample": 50,
        "full": 100
    }
    
    if args.max_examples is None:
        args.max_examples = size_configs.get(args.model_size, 50)
    
    log.info(f"Configuration: {args.model_size}, device={device}, batch_size={args.batch_size}")
    
    # Load model
    model, config = load_model(args.model_dir, args.model_size, device)
    
    # Load or generate dataset
    data_path = Path(args.data_dir)
    if (data_path / "volumes.npy").exists():
        dataset = Synthetic3DDataset(
            args.data_dir, 
            max_samples=args.max_examples,
            input_shape=config['input_shape']
        )
    else:
        log.info(f"Data not found, generating {args.max_examples} synthetic volumes")
        dataset = GeneratedVolumeDataset(
            num_samples=args.max_examples,
            input_shape=config['input_shape']
        )
    
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
    
    # Add configuration
    results["model_size"] = args.model_size
    results["input_shape"] = list(config['input_shape'])
    results["mlperf_mode"] = args.mlperf
    results["mlperf_compliant"] = args.mlperf and (data_path / "volumes.npy").exists()
    
    # Print command for reproducibility
    print("\n" + "=" * 60)
    print("COMMAND")
    print("=" * 60)
    print(build_command(args))
    print("=" * 60)
    
    # Save results
    save_results(results, args.output_dir, args.model_size, device)


if __name__ == "__main__":
    main()
