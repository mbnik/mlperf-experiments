#!/usr/bin/env python3
"""
MLPerf Benchmark Setup and Runner - DLRM Recommendation

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
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("DLRM-Benchmark")


# ============================================================================
# GPU Cleanup Utilities
# ============================================================================

_model_ref = None


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


# ============================================================================
# DLRM Model Definition
# ============================================================================

class DLRM(nn.Module):
    """
    Deep Learning Recommendation Model (DLRM)
    Supports both small debug and full configurations
    """
    def __init__(
        self,
        embedding_dim: int = 128,
        num_dense_features: int = 13,
        embedding_sizes: List[int] = None,
        bottom_mlp_dims: List[int] = None,
        top_mlp_dims: List[int] = None,
        use_cpu_embedding: bool = False,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_dense_features = num_dense_features
        self.use_cpu_embedding = use_cpu_embedding
        
        # Default configurations
        if embedding_sizes is None:
            # Small debug configuration
            embedding_sizes = [1000] * 26
        self.embedding_sizes = embedding_sizes
        self.num_sparse_features = len(embedding_sizes)
        
        if bottom_mlp_dims is None:
            bottom_mlp_dims = [num_dense_features, 512, 256, embedding_dim]
        if top_mlp_dims is None:
            top_mlp_dims = [embedding_dim * (self.num_sparse_features + 1), 512, 256, 1]
        
        # Bottom MLP (processes dense features)
        self.bottom_mlp = self._build_mlp(bottom_mlp_dims)
        
        # Embedding tables for sparse features
        # Use Embedding instead of EmbeddingBag for simpler single-index lookup
        self.embedding_tables = nn.ModuleList([
            nn.Embedding(size, embedding_dim)
            for size in embedding_sizes
        ])
        
        # Top MLP (processes interaction features)
        interaction_dim = embedding_dim + self.num_sparse_features * embedding_dim
        top_mlp_dims[0] = interaction_dim
        self.top_mlp = self._build_mlp(top_mlp_dims)
        
        self.sigmoid = nn.Sigmoid()
    
    def _build_mlp(self, dims: List[int]) -> nn.Sequential:
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def forward(self, dense_features: torch.Tensor, sparse_features: torch.Tensor) -> torch.Tensor:
        # Process dense features through bottom MLP
        dense_out = self.bottom_mlp(dense_features)  # [batch, embedding_dim]
        
        # Process sparse features through embeddings
        sparse_outs = []
        
        for i, emb_table in enumerate(self.embedding_tables):
            # Get indices for this sparse feature and clamp to valid range
            indices = sparse_features[:, i].long()
            indices = torch.clamp(indices, 0, self.embedding_sizes[i] - 1)
            
            # Handle CPU embeddings if offloading
            if self.use_cpu_embedding and emb_table.weight.device.type == 'cpu':
                indices = indices.cpu()
                sparse_out = emb_table(indices).to(dense_features.device)
            else:
                sparse_out = emb_table(indices)
            
            sparse_outs.append(sparse_out)
        
        sparse_out = torch.cat(sparse_outs, dim=1)  # [batch, num_sparse * embedding_dim]
        
        # Concatenate dense and sparse
        interaction = torch.cat([dense_out, sparse_out], dim=1)
        
        # Top MLP
        out = self.top_mlp(interaction)
        return self.sigmoid(out).squeeze(-1)


# ============================================================================
# Data Loading
# ============================================================================

class DLRMDataset:
    """Dataset loader supporting multiple formats"""
    
    def __init__(self, data_dir: str, max_samples: int = None):
        self.data_dir = data_dir
        self.max_samples = max_samples
        self.labels = None
        self.dense_features = None
        self.sparse_features = None
        self.total_available = 0
        self.data_format = 'unknown'
        self._load_data()
    
    def _load_data(self):
        log.info(f"Loading data from: {self.data_dir}")
        
        # Try different formats
        if os.path.exists(os.path.join(self.data_dir, "labels.npy")):
            # Our synthetic/sample format
            self.data_format = 'synthetic'
            self.labels = np.load(os.path.join(self.data_dir, "labels.npy"))
            self.dense_features = np.load(os.path.join(self.data_dir, "dense_features.npy"))
            self.sparse_features = np.load(os.path.join(self.data_dir, "sparse_features.npy"))
            self.total_available = len(self.labels)
        elif os.path.exists(os.path.join(self.data_dir, "day_23_labels.npy")):
            # Real MLPerf Criteo format - separate files
            self.data_format = 'criteo_real'
            log.info("Loading real Criteo dataset (separate files)...")
            self.labels = np.load(os.path.join(self.data_dir, "day_23_labels.npy"))
            self.dense_features = np.load(os.path.join(self.data_dir, "day_23_dense.npy"))
            self.total_available = len(self.labels)
            # For sparse features, load from the npz archive
            sparse_file = os.path.join(self.data_dir, "day_23_sparse_multi_hot.npz")
            if os.path.exists(sparse_file):
                sparse_data = np.load(sparse_file, allow_pickle=True)
                # The sparse file contains sparse feature arrays per embedding table
                # For our benchmark, we'll convert to a simple format
                sparse_keys = list(sparse_data.keys())
                log.info(f"Sparse file contains {len(sparse_keys)} tables")
                # Stack sparse features - take first few tables for sample run
                if sparse_keys:
                    # Create a combined sparse feature array
                    num_samples = len(self.labels)
                    num_tables = min(len(sparse_keys), 26)  # DLRM has 26 embedding tables
                    self.sparse_features = np.zeros((num_samples, num_tables), dtype=np.int64)
                    for i, key in enumerate(sparse_keys[:num_tables]):
                        arr = sparse_data[key]
                        if len(arr) >= num_samples:
                            # Handle multi-hot encoding (take first element)
                            if arr.ndim > 1:
                                self.sparse_features[:, i] = arr[:num_samples, 0]
                            else:
                                self.sparse_features[:, i] = arr[:num_samples]
                else:
                    raise ValueError("Sparse file is empty")
            else:
                raise FileNotFoundError(f"Sparse file not found: {sparse_file}")
        elif os.path.exists(os.path.join(self.data_dir, "day_23_sparse_multi_hot.npz")):
            # Compressed format (all in one npz with labels/dense/sparse keys)
            self.data_format = 'criteo_compressed'
            data = np.load(os.path.join(self.data_dir, "day_23_sparse_multi_hot.npz"))
            self.labels = data['labels']
            self.dense_features = data['dense']
            self.sparse_features = data['sparse']
            self.total_available = len(self.labels)
        else:
            raise FileNotFoundError(f"No valid data format found in {self.data_dir}")
        
        # Store total before limiting
        self.total_available = len(self.labels)
        
        # Limit samples if requested
        if self.max_samples and self.max_samples < len(self.labels):
            self.labels = self.labels[:self.max_samples]
            self.dense_features = self.dense_features[:self.max_samples]
            self.sparse_features = self.sparse_features[:self.max_samples]
        
        log.info(f"Loaded {len(self.labels)} samples")
    
    def __len__(self):
        return len(self.labels)
    
    def get_batch(self, start: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        end = min(start + batch_size, len(self.labels))
        
        labels = torch.from_numpy(self.labels[start:end]).float()
        dense = torch.from_numpy(self.dense_features[start:end]).float()
        sparse = torch.from_numpy(self.sparse_features[start:end]).long()
        
        return labels, dense, sparse
    
    def get_data_info(self) -> Dict:
        """Get information about the dataset"""
        is_real = self.data_format in ['criteo_real', 'criteo_compressed']
        
        dataset_name = 'Criteo Terabyte' if is_real else 'Synthetic Criteo-like'
        note = 'Real Criteo click-through rate prediction data' if is_real else 'Random synthetic data for testing only'
        
        return {
            'type': 'real' if is_real else 'synthetic',
            'dataset': dataset_name,
            'source': self.data_dir,
            'format': self.data_format,
            'samples_used': len(self.labels),
            'samples_available': self.total_available,
            'dense_features': self.dense_features.shape[1] if self.dense_features is not None else 13,
            'sparse_features': self.sparse_features.shape[1] if self.sparse_features is not None else 26,
            'embedding_tables': 26,
            'verified': is_real,
            'mlperf_compliant': is_real,
            'note': note
        }


# ============================================================================
# Command Builder
# ============================================================================

def build_command(args) -> str:
    """Build the command line string for reproducibility"""
    import sys
    cmd_parts = [sys.executable, __file__]
    
    cmd_parts.extend(["--model-size", args.model_size])
    if args.model_path:
        cmd_parts.extend(["--model-path", args.model_path])
    cmd_parts.extend(["--device", args.device])
    cmd_parts.extend(["--data-dir", args.data_dir])
    cmd_parts.extend(["--data-type", args.data_type])
    if args.max_examples:
        cmd_parts.extend(["--max-examples", str(args.max_examples)])
    cmd_parts.extend(["--batch-size", str(args.batch_size)])
    cmd_parts.extend(["--output-dir", args.output_dir])
    
    if args.offload:
        cmd_parts.append("--offload")
    if args.mlperf:
        cmd_parts.append("--mlperf")
    
    return " ".join(cmd_parts)


# ============================================================================
# Benchmark Runner
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser(description="DLRM-v2 Benchmark")
    
    # Model options
    parser.add_argument("--model-size", type=str, default="small",
                       choices=["small", "sample", "full"],
                       help="Model size: small (debug), sample, full")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to model weights (optional)")
    
    # Device options
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device: cuda (GPU) or cpu")
    parser.add_argument("--offload", action="store_true",
                       help="Enable embedding offloading to CPU (for large models)")
    parser.add_argument("--max-gpu-memory", type=str, default="10GB",
                       help="Max GPU memory for offloading")
    
    # Data options
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--data-type", type=str, default="synthetic",
                       choices=["synthetic", "real"],
                       help="Data type: synthetic or real Criteo")
    parser.add_argument("--max-examples", type=int, default=None,
                       help="Maximum samples to process")
    
    # Benchmark options
    parser.add_argument("--batch-size", type=int, default=2048,
                       help="Batch size for inference")
    parser.add_argument("--warmup-batches", type=int, default=10,
                       help="Number of warmup batches")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--mlperf", action="store_true",
                       help="Use MLPerf official settings")
    
    return parser.parse_args()


def get_model_config(model_size: str) -> Dict:
    """Get model configuration based on size"""
    
    configs = {
        "small": {
            "embedding_dim": 64,
            "num_dense_features": 13,
            "embedding_sizes": [1000] * 26,
            "bottom_mlp_dims": [13, 512, 256, 64],
            "top_mlp_dims": [64 * 27, 512, 256, 1],
            "description": "Small debug model (~10MB)"
        },
        "sample": {
            "embedding_dim": 128,
            "num_dense_features": 13,
            "embedding_sizes": [10000] * 26,  # Reduced embedding sizes
            "bottom_mlp_dims": [13, 512, 256, 128],
            "top_mlp_dims": [128 * 27, 1024, 512, 256, 1],
            "description": "Sample model (~100MB)"
        },
        "full": {
            "embedding_dim": 128,
            "num_dense_features": 13,
            # Full Criteo embedding sizes (26 categorical features)
            "embedding_sizes": [
                40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63,
                40000000, 3067956, 405282, 10, 2209, 11938, 155, 4,
                976, 14, 40000000, 40000000, 40000000, 590152, 12973, 108, 36
            ],
            "bottom_mlp_dims": [13, 512, 256, 128],
            "top_mlp_dims": [128 * 27, 1024, 1024, 512, 256, 1],
            "description": "Full MLPerf model (~97GB)"
        }
    }
    
    return configs[model_size]


def load_model(args) -> DLRM:
    """Load or create DLRM model"""
    config = get_model_config(args.model_size)
    log.info(f"Model configuration: {config['description']}")
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        log.error("CUDA not available. Options:")
        log.error("  1. Use --cpu for CPU-only mode (slow)")
        log.error("  2. Install CUDA and GPU drivers")
        raise SystemExit(1)
    
    # For large models, check if we have enough memory
    if args.model_size in ["sample", "full"] and args.device == "cuda" and not args.offload:
        # Estimate model size
        emb_params = sum(config["embedding_sizes"]) * config["embedding_dim"]
        emb_size_gb = emb_params * 4 / 1e9  # float32
        
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if emb_size_gb > gpu_mem_gb * 0.8:  # Leave 20% margin
                log.error("=" * 60)
                log.error("MODEL TOO LARGE FOR GPU!")
                log.error("=" * 60)
                log.error(f"Model embeddings need ~{emb_size_gb:.1f} GB")
                log.error(f"GPU has {gpu_mem_gb:.1f} GB total")
                log.error("")
                log.error("Use --offload to keep embeddings on CPU:")
                log.error(f"  ./scripts/run_benchmark.sh dlrm --mlperf --offload")
                raise SystemExit(1)
    
    # Determine if we need CPU offloading for embeddings
    use_cpu_embedding = args.offload
    
    model = DLRM(
        embedding_dim=config["embedding_dim"],
        num_dense_features=config["num_dense_features"],
        embedding_sizes=config["embedding_sizes"],
        bottom_mlp_dims=config["bottom_mlp_dims"],
        top_mlp_dims=config["top_mlp_dims"],
        use_cpu_embedding=use_cpu_embedding,
    )
    
    # Load weights if provided
    if args.model_path and os.path.exists(args.model_path):
        log.info(f"Loading weights from: {args.model_path}")
        state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    
    # Move model to device
    try:
        if args.device == "cpu":
            log.info("Running on CPU")
            model = model.to("cpu")
        elif args.offload:
            log.info("Running with GPU+CPU offloading")
            log.info("  - MLP layers on GPU")
            log.info("  - Embedding tables on CPU (large memory)")
            
            # Clear any existing CUDA cache first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # IMPORTANT: Ensure embeddings stay on CPU FIRST
            # (they should already be on CPU from creation)
            for i, emb in enumerate(model.embedding_tables):
                if emb.weight.device.type != 'cpu':
                    model.embedding_tables[i] = emb.cpu()
            
            # Now move only MLPs to GPU
            model.bottom_mlp = model.bottom_mlp.to("cuda")
            model.top_mlp = model.top_mlp.to("cuda")
            model.sigmoid = model.sigmoid.to("cuda")
            
            # Verify embeddings are still on CPU
            for i, emb in enumerate(model.embedding_tables):
                if emb.weight.device.type != 'cpu':
                    log.warning(f"Embedding {i} ended up on {emb.weight.device}, forcing to CPU")
                    model.embedding_tables[i] = emb.cpu()
            
            # Calculate memory distribution and store for summary
            mlp_params = sum(p.numel() for p in model.bottom_mlp.parameters()) + \
                        sum(p.numel() for p in model.top_mlp.parameters())
            emb_params = sum(p.numel() for p in model.embedding_tables.parameters())
            
            # Store offload info as model attributes for later reporting
            model._offload_info = {
                'enabled': True,
                'mlp_params': mlp_params,
                'mlp_size_mb': mlp_params * 4 / 1e6,
                'emb_params': emb_params,
                'emb_size_gb': emb_params * 4 / 1e9,
                'gpu_components': ['bottom_mlp', 'top_mlp', 'sigmoid'],
                'cpu_components': [f'embedding_table_{i} ({size:,} rows)' 
                                   for i, size in enumerate(config['embedding_sizes'])],
            }
            
            log.info(f"  - MLP params: {mlp_params:,} ({mlp_params * 4 / 1e6:.1f} MB)")
            log.info(f"  - Embedding params: {emb_params:,} ({emb_params * 4 / 1e9:.1f} GB on CPU)")
            
            # Verify GPU memory usage
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1e9
                model._offload_info['gpu_memory_gb'] = gpu_mem
                log.info(f"  - GPU memory used: {gpu_mem:.2f} GB")
        else:
            log.info("Running on GPU")
            model = model.to("cuda")
            model._offload_info = {'enabled': False}
    except torch.cuda.OutOfMemoryError:
        log.error("=" * 60)
        log.error("CUDA OUT OF MEMORY!")
        log.error("=" * 60)
        log.error("DLRM embeddings are too large for GPU. Try:")
        log.error("  1. --offload : Move embeddings to CPU")
        log.error("  2. --cpu     : Run entirely on CPU (slow)")
        log.error("  3. Use --model-size small or sample")
        raise SystemExit(1)
    
    model.eval()
    return model


def run_benchmark(model: DLRM, dataset: DLRMDataset, args, data_info: Dict = None) -> Dict:
    """Run the benchmark"""
    log.info("=" * 60)
    log.info("Starting DLRM Benchmark")
    log.info("=" * 60)
    
    # Get offload info if available
    offload_info = getattr(model, '_offload_info', {'enabled': False})
    
    device = args.device
    
    total_samples = len(dataset)
    num_batches = (total_samples + args.batch_size - 1) // args.batch_size
    
    log.info(f"Total samples: {total_samples}")
    log.info(f"Batch size: {args.batch_size}")
    log.info(f"Number of batches: {num_batches}")
    
    # Safety check: ensure embeddings are on correct device before starting
    if args.offload:
        for i, emb in enumerate(model.embedding_tables):
            if emb.weight.device.type != 'cpu':
                log.error(f"ERROR: Embedding table {i} is on {emb.weight.device}, expected CPU!")
                log.error("This could cause GPU memory issues. Aborting.")
                raise SystemExit(1)
    
    # Warmup with error handling
    log.info(f"\nWarmup ({args.warmup_batches} batches)...")
    try:
        with torch.no_grad():
            for i in range(min(args.warmup_batches, num_batches)):
                labels, dense, sparse = dataset.get_batch(i * args.batch_size, args.batch_size)
                dense = dense.to(device)
                sparse = sparse.to(device) if not args.offload else sparse
                _ = model(dense, sparse)
                
                # Clear GPU cache periodically during warmup
                if device == "cuda" and i % 5 == 0:
                    torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            log.error("=" * 60)
            log.error("GPU OUT OF MEMORY DURING WARMUP!")
            log.error("=" * 60)
            log.error("Try: --offload or --batch-size 512")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise SystemExit(1)
        raise
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    log.info("\nRunning benchmark...")
    all_predictions = []
    all_labels = []
    batch_times = []
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            batch_start = time.perf_counter()
            
            labels, dense, sparse = dataset.get_batch(batch_idx * args.batch_size, args.batch_size)
            dense = dense.to(device)
            sparse = sparse.to(device) if not args.offload else sparse
            
            predictions = model(dense, sparse)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            batch_time = time.perf_counter() - batch_start
            batch_times.append(batch_time)
            
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            
            if (batch_idx + 1) % 100 == 0 or batch_idx == num_batches - 1:
                log.info(f"  Batch {batch_idx + 1}/{num_batches} - "
                        f"{batch_time * 1000:.1f}ms/batch, "
                        f"{args.batch_size / batch_time:.0f} samples/sec")
    
    total_time = time.perf_counter() - start_time
    
    # Calculate metrics - ensure arrays are 1D
    all_predictions = np.array(all_predictions).ravel()
    all_labels = np.array(all_labels).ravel()
    
    # Accuracy (threshold at 0.5)
    binary_preds = (all_predictions > 0.5).astype(int)
    accuracy = (binary_preds == all_labels).mean()
    
    # AUC-ROC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_predictions)
    except:
        auc = None
    
    # Throughput
    throughput = total_samples / total_time
    avg_batch_time = np.mean(batch_times)
    p99_batch_time = np.percentile(batch_times, 99)
    
    results = {
        "model_size": args.model_size,
        "device": args.device,
        "offload_enabled": offload_info.get('enabled', False),
        "data_type": args.data_type,
        "batch_size": args.batch_size,
        "total_samples": total_samples,
        "total_time_sec": total_time,
        "throughput_samples_per_sec": throughput,
        "avg_batch_time_ms": avg_batch_time * 1000,
        "p99_batch_time_ms": p99_batch_time * 1000,
        "accuracy": accuracy,
        "auc_roc": auc,
        "mlperf_mode": args.mlperf,
        "mlperf_compliant": args.mlperf and args.data_type == "real",
    }
    
    # Add offload details if enabled
    if offload_info.get('enabled'):
        results["offload_details"] = {
            "gpu_memory_gb": offload_info.get('gpu_memory_gb', 0),
            "mlp_size_mb": offload_info.get('mlp_size_mb', 0),
            "embedding_size_gb": offload_info.get('emb_size_gb', 0),
        }
    
    # Print summary
    print("\n" + "=" * 60)
    print("DLRM BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Model Size:         {args.model_size}")
    print(f"Device:             {args.device}")
    
    # Show memory distribution when offloading is enabled
    if offload_info.get('enabled'):
        print(f"Offload Mode:       ENABLED")
        print("-" * 40)
        print("Memory Distribution:")
        print(f"  🖥️  GPU:")
        print(f"      - Bottom MLP (dense features)")
        print(f"      - Top MLP (interaction + output)")
        print(f"      - Memory: {offload_info.get('gpu_memory_gb', 0):.2f} GB")
        print(f"  💾 CPU:")
        print(f"      - 26 Embedding tables ({offload_info.get('emb_params', 0):,} params)")
        print(f"      - Memory: {offload_info.get('emb_size_gb', 0):.1f} GB")
        print("-" * 40)
    
    print(f"Data Type:          {args.data_type}")
    if args.mlperf:
        print(f"MLPerf Mode:        ENABLED")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Total Samples:      {total_samples:,}")
    print(f"Total Time:         {total_time:.2f} seconds")
    print(f"Throughput:         {throughput:,.0f} samples/sec")
    print(f"Avg Batch Time:     {avg_batch_time * 1000:.2f} ms")
    print(f"P99 Batch Time:     {p99_batch_time * 1000:.2f} ms")
    print(f"Accuracy:           {accuracy * 100:.2f}%")
    if auc:
        print(f"AUC-ROC:            {auc:.4f}")
    print("=" * 60)
    
    # Performance rating
    if throughput >= 100000:
        print("Performance:        🚀 Excellent")
    elif throughput >= 10000:
        print("Performance:        ✅ Good")
    elif throughput >= 1000:
        print("Performance:        ⚠️ Moderate")
    else:
        print("Performance:        🐢 Slow")
    print("=" * 60)
    
    # Print data information
    if data_info:
        print("\n" + "=" * 60)
        print("DATA INFORMATION")
        print("=" * 60)
        print(f"Type:               {data_info['type']}")
        print(f"Dataset:            {data_info['dataset']}")
        print(f"Source:             {data_info['source']}")
        print(f"Format:             {data_info['format']}")
        print(f"Samples Used:       {data_info['samples_used']:,}")
        print(f"Samples Available:  {data_info['samples_available']:,}")
        print(f"Dense Features:     {data_info['dense_features']}")
        print(f"Sparse Features:    {data_info['sparse_features']}")
        print(f"Embedding Tables:   {data_info['embedding_tables']}")
        print(f"Verified:           {'✓' if data_info['verified'] else '✗'}")
        print(f"MLPerf Compliant:   {'✓' if data_info['mlperf_compliant'] else '✗'}")
        print(f"Note:               {data_info['note']}")
        print("=" * 60)
        
        # Add data_info to results
        results['data_info'] = data_info
    
    return results


def main():
    global _model_ref
    
    args = get_args()
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, "dlrm")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check data directory
    if not os.path.exists(args.data_dir):
        log.error(f"Data directory not found: {args.data_dir}")
        log.error("Run: ./scripts/download_dlrm.sh --all --size small")
        sys.exit(1)
    
    try:
        # Load dataset
        dataset = DLRMDataset(args.data_dir, args.max_examples)
        
        # Get data info
        data_info = dataset.get_data_info()
        
        # Load model
        model = load_model(args)
        _model_ref = model
        
        # Run benchmark
        results = run_benchmark(model, dataset, args, data_info=data_info)
        
        # Print command for reproducibility
        print("\n" + "=" * 60)
        print("COMMAND")
        print("=" * 60)
        print(build_command(args))
        print("=" * 60)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_dir, f"dlrm_benchmark_{args.model_size}_{args.device}_{timestamp}.json")
        
        with open(result_file, "w") as f:
            json.dump({
                "summary": results,
                "timestamp": datetime.now().isoformat(),
                "args": vars(args),
            }, f, indent=2)
        
        log.info(f"\nResults saved to: {result_file}")
    
    finally:
        cleanup_gpu()


if __name__ == "__main__":
    main()
