#!/bin/bash
# ============================================================================
# MLPerf Benchmark Setup and Runner - DLRM Data Download Script
#
# Author: Mehdi Nik
# Created: Jan 2026
#
# DISCLAIMER:
# This repository is NOT an official MLPerf implementation.
# It contains personal tooling and scripts to run MLPerf workloads.
#
# This software is provided "as is" without warranty of any kind, express or
# implied. The author assumes no responsibility for errors, omissions, or
# damages arising from its use.
#
# All rights reserved. If you use or reference this work, please provide
# attribution to the original author.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default paths
MODEL_DIR="${PROJECT_DIR}/models/dlrm"
DATA_DIR="${PROJECT_DIR}/data/criteo"

# Parse arguments
DOWNLOAD_MODEL=false
DOWNLOAD_DATASET=false
SIZE="sample"  # small, sample, full

print_usage() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║         DLRM-v2 Model & Dataset Downloader                 ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}Usage:${NC} $0 [options]"
    echo ""
    echo -e "${YELLOW}Download Options:${NC}"
    echo "  --model       Download model weights only"
    echo "  --dataset     Download dataset only"
    echo "  --all         Download both model and dataset"
    echo ""
    echo -e "${YELLOW}Size Options:${NC}"
    echo "  --size small  Debug size (~1GB model, synthetic data)"
    echo "  --size sample Sample dataset (~10GB) with full model"
    echo "  --size full   Full official MLPerf (~200GB total)"
    echo ""
    echo -e "${YELLOW}Other Options:${NC}"
    echo "  --model-dir   Custom model directory"
    echo "  --data-dir    Custom data directory"
    echo "  --help, -h    Show this help"
    echo ""
    echo -e "${YELLOW}Size Comparison:${NC}"
    echo "  ┌──────────┬────────────┬─────────────┬──────────────────────┐"
    echo "  │ Size     │ Model      │ Dataset     │ Use Case             │"
    echo "  ├──────────┼────────────┼─────────────┼──────────────────────┤"
    echo "  │ small    │ ~1GB       │ Synthetic   │ Quick testing        │"
    echo "  │ sample   │ ~97GB      │ ~10GB       │ Development/debug    │"
    echo "  │ full     │ ~97GB      │ ~100GB      │ Official MLPerf      │"
    echo "  └──────────┴────────────┴─────────────┴──────────────────────┘"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 --all --size small      # Quick test setup (~1GB)"
    echo "  $0 --all --size sample     # Development setup (~107GB)"
    echo "  $0 --model --size full     # Download full model only (~97GB)"
    echo "  $0 --all --size full       # Full MLPerf setup (~200GB)"
    echo ""
}

check_disk_space() {
    local required_gb=$1
    local target_dir=$2
    
    # Get available space in GB
    local available_gb=$(df -BG "$target_dir" 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G')
    
    if [[ -z "$available_gb" ]]; then
        echo -e "${YELLOW}Warning: Could not check disk space${NC}"
        return 0
    fi
    
    if [[ "$available_gb" -lt "$required_gb" ]]; then
        echo -e "${RED}Error: Not enough disk space${NC}"
        echo -e "Required: ${required_gb}GB, Available: ${available_gb}GB"
        echo -e "Target directory: $target_dir"
        return 1
    fi
    
    echo -e "${GREEN}Disk space OK:${NC} ${available_gb}GB available, ${required_gb}GB required"
    return 0
}

download_with_progress() {
    local url=$1
    local output_dir=$2
    local description=$3
    
    echo -e "${CYAN}Downloading: ${description}${NC}"
    echo -e "Target: ${output_dir}"
    
    mkdir -p "$output_dir"
    cd "$output_dir"
    
    # Use the MLCommons R2 downloader
    bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) "$url"
    
    echo -e "${GREEN}✓ Download complete: ${description}${NC}"
}

generate_synthetic_data() {
    local output_dir=$1
    local num_samples=${2:-1000000}  # Default 1M samples
    echo -e "${CYAN}Generating synthetic Criteo data for testing...${NC}"
    
    mkdir -p "$output_dir"
    
    python3 << PYTHON_EOF
import numpy as np
import os

output_dir = "$output_dir"
num_samples = $num_samples

print(f"Generating synthetic Criteo-like data ({num_samples} samples)...")

# DLRM-v2 configuration
num_dense_features = 13
num_sparse_features = 26

# Embedding table sizes for sample config (10K per table)
# This works with all model sizes (small: 1K, sample: 10K, full: >10K)
embedding_sizes = [10000] * 26

# Generate labels (click/no-click)
labels = np.random.randint(0, 2, size=num_samples).astype(np.int32)

# Generate dense features (normalized)
dense_features = np.random.randn(num_samples, num_dense_features).astype(np.float32)

# Generate sparse features (indices into embedding tables)
sparse_features = []
for i, size in enumerate(embedding_sizes):
    indices = np.random.randint(0, size, size=num_samples).astype(np.int32)
    sparse_features.append(indices)

sparse_features = np.stack(sparse_features, axis=1)

print(f"Data shapes: dense={dense_features.shape}, sparse={sparse_features.shape}, labels={labels.shape}")

# Save as binary files (similar to Criteo format)
np.save(os.path.join(output_dir, "labels.npy"), labels)
np.save(os.path.join(output_dir, "dense_features.npy"), dense_features)
np.save(os.path.join(output_dir, "sparse_features.npy"), sparse_features)

# Also save as day_23 format for compatibility
with open(os.path.join(output_dir, "day_23_sparse_multi_hot.npz"), 'wb') as f:
    np.savez_compressed(f, 
                       labels=labels,
                       dense=dense_features,
                       sparse=sparse_features)

print(f"Generated {num_samples} samples")
print(f"  Labels shape: {labels.shape}")
print(f"  Dense features shape: {dense_features.shape}")
print(f"  Sparse features shape: {sparse_features.shape}")
print(f"Saved to: {output_dir}")

# Create a metadata file
with open(os.path.join(output_dir, "metadata.txt"), 'w') as f:
    f.write(f"num_samples={num_samples}\\n")
    f.write(f"num_dense_features={num_dense_features}\\n")
    f.write(f"num_sparse_features={num_sparse_features}\\n")
    f.write("type=synthetic\\n")

PYTHON_EOF
    
    echo -e "${GREEN}✓ Synthetic data generated${NC}"
}

generate_small_model() {
    local output_dir=$1
    echo -e "${CYAN}Generating small debug model...${NC}"
    
    mkdir -p "$output_dir"
    
    python3 << PYTHON_EOF
import torch
import os

output_dir = "${output_dir}"

print("Generating small DLRM debug model...")

# Small embedding dimensions for testing
embedding_dim = 64
num_sparse_features = 26
bottom_mlp_dims = [13, 512, 256, embedding_dim]
top_mlp_dims = [512, 256, 1]

# Small embedding table sizes
embedding_sizes = [1000] * num_sparse_features

# Create a simple model state dict
state_dict = {}

# Bottom MLP weights
in_dim = 13
for i, out_dim in enumerate(bottom_mlp_dims[1:]):
    state_dict[f'bottom_mlp.{i*2}.weight'] = torch.randn(out_dim, in_dim)
    state_dict[f'bottom_mlp.{i*2}.bias'] = torch.zeros(out_dim)
    in_dim = out_dim

# Embedding tables (small)
for i, size in enumerate(embedding_sizes):
    state_dict[f'embedding_tables.{i}.weight'] = torch.randn(size, embedding_dim)

# Top MLP weights
in_dim = embedding_dim * (num_sparse_features + 1)  # sparse + dense interaction
for i, out_dim in enumerate(top_mlp_dims):
    if i == 0:
        in_dim = 512  # Simplified
    state_dict[f'top_mlp.{i*2}.weight'] = torch.randn(out_dim, in_dim)
    state_dict[f'top_mlp.{i*2}.bias'] = torch.zeros(out_dim)
    in_dim = out_dim

# Save model
model_path = os.path.join(output_dir, "dlrm_small.pt")
torch.save(state_dict, model_path)

# Save config
config = {
    'embedding_dim': embedding_dim,
    'num_sparse_features': num_sparse_features,
    'embedding_sizes': embedding_sizes,
    'bottom_mlp_dims': bottom_mlp_dims,
    'top_mlp_dims': top_mlp_dims,
    'model_type': 'small_debug'
}

import json
with open(os.path.join(output_dir, "config.json"), 'w') as f:
    json.dump(config, f, indent=2)

size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"Model saved to: {model_path}")
print(f"Model size: {size_mb:.1f} MB")

PYTHON_EOF
    
    echo -e "${GREEN}✓ Small debug model generated${NC}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            print_usage
            exit 0
            ;;
        --model)
            DOWNLOAD_MODEL=true
            shift
            ;;
        --dataset)
            DOWNLOAD_DATASET=true
            shift
            ;;
        --all)
            DOWNLOAD_MODEL=true
            DOWNLOAD_DATASET=true
            shift
            ;;
        --size)
            SIZE="$2"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Validate size option
if [[ "$SIZE" != "small" && "$SIZE" != "sample" && "$SIZE" != "full" ]]; then
    echo -e "${RED}Invalid size option: $SIZE${NC}"
    echo "Valid options: small, sample, full"
    exit 1
fi

# Check if anything to download
if [[ "$DOWNLOAD_MODEL" == false && "$DOWNLOAD_DATASET" == false ]]; then
    echo -e "${YELLOW}No download option specified. Use --model, --dataset, or --all${NC}"
    print_usage
    exit 1
fi

# Print summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         DLRM-v2 Download Configuration                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Size:${NC}        $SIZE"
echo -e "${YELLOW}Model:${NC}       $DOWNLOAD_MODEL -> $MODEL_DIR"
echo -e "${YELLOW}Dataset:${NC}     $DOWNLOAD_DATASET -> $DATA_DIR"
echo ""

# Estimate sizes and check disk space
case $SIZE in
    small)
        MODEL_SIZE_GB=1
        DATA_SIZE_GB=1
        ;;
    sample)
        MODEL_SIZE_GB=100
        DATA_SIZE_GB=15
        ;;
    full)
        MODEL_SIZE_GB=100
        DATA_SIZE_GB=110
        ;;
esac

TOTAL_NEEDED=0
[[ "$DOWNLOAD_MODEL" == true ]] && TOTAL_NEEDED=$((TOTAL_NEEDED + MODEL_SIZE_GB))
[[ "$DOWNLOAD_DATASET" == true ]] && TOTAL_NEEDED=$((TOTAL_NEEDED + DATA_SIZE_GB))

echo -e "${YELLOW}Estimated download size:${NC} ~${TOTAL_NEEDED}GB"
echo ""

# Check disk space
mkdir -p "$MODEL_DIR" "$DATA_DIR"
if ! check_disk_space "$TOTAL_NEEDED" "$PROJECT_DIR"; then
    echo -e "${RED}Aborting due to insufficient disk space${NC}"
    exit 1
fi

echo ""
read -p "Continue with download? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Download based on size
case $SIZE in
    small)
        echo -e "\n${CYAN}=== Setting up SMALL debug configuration ===${NC}\n"
        
        if [[ "$DOWNLOAD_MODEL" == true ]]; then
            generate_small_model "$MODEL_DIR"
        fi
        
        if [[ "$DOWNLOAD_DATASET" == true ]]; then
            generate_synthetic_data "$DATA_DIR"
        fi
        ;;
        
    sample)
        echo -e "\n${CYAN}=== Setting up SAMPLE configuration ===${NC}\n"
        
        if [[ "$DOWNLOAD_MODEL" == true ]]; then
            echo -e "${CYAN}Downloading full model weights (~97GB)...${NC}"
            echo -e "${YELLOW}This will take a while...${NC}"
            download_with_progress \
                "https://inference.mlcommons-storage.org/metadata/dlrm-v2-model-weights.uri" \
                "$MODEL_DIR" \
                "DLRM-v2 Model Weights"
        fi
        
        if [[ "$DOWNLOAD_DATASET" == true ]]; then
            echo -e "${CYAN}Downloading sample dataset...${NC}"
            # For sample, we generate synthetic but larger
            python3 << PYTHON_EOF
import numpy as np
import os

output_dir = "${DATA_DIR}"
os.makedirs(output_dir, exist_ok=True)

print("Generating sample Criteo-like data (1M samples)...")

num_samples = 1000000  # 1M samples
num_dense_features = 13
num_sparse_features = 26

labels = np.random.randint(0, 2, size=num_samples).astype(np.int32)
dense_features = np.random.randn(num_samples, num_dense_features).astype(np.float32)

embedding_sizes = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63,
    40000000, 3067956, 405282, 10, 2209, 11938, 155, 4,
    976, 14, 40000000, 40000000, 40000000, 590152, 12973, 108
]

sparse_features = []
for i, size in enumerate(embedding_sizes):
    indices = np.random.randint(0, min(size, 1000000), size=num_samples).astype(np.int32)
    sparse_features.append(indices)
sparse_features = np.stack(sparse_features, axis=1)

np.save(os.path.join(output_dir, "labels.npy"), labels)
np.save(os.path.join(output_dir, "dense_features.npy"), dense_features)
np.save(os.path.join(output_dir, "sparse_features.npy"), sparse_features)

with open(os.path.join(output_dir, "day_23_sparse_multi_hot.npz"), 'wb') as f:
    np.savez_compressed(f, labels=labels, dense=dense_features, sparse=sparse_features)

print(f"Generated {num_samples} samples")

with open(os.path.join(output_dir, "metadata.txt"), 'w') as f:
    f.write(f"num_samples={num_samples}\n")
    f.write("type=sample\n")

PYTHON_EOF
        fi
        ;;
        
    full)
        echo -e "\n${CYAN}=== Setting up FULL MLPerf configuration ===${NC}\n"
        
        if [[ "$DOWNLOAD_MODEL" == true ]]; then
            echo -e "${CYAN}Downloading full model weights (~97GB)...${NC}"
            echo -e "${YELLOW}This will take a while...${NC}"
            download_with_progress \
                "https://inference.mlcommons-storage.org/metadata/dlrm-v2-model-weights.uri" \
                "$MODEL_DIR" \
                "DLRM-v2 Model Weights"
        fi
        
        if [[ "$DOWNLOAD_DATASET" == true ]]; then
            echo -e "${CYAN}Downloading full preprocessed dataset (~100GB)...${NC}"
            echo -e "${YELLOW}This will take a while...${NC}"
            download_with_progress \
                "https://inference.mlcommons-storage.org/metadata/dlrm-v2-preprocessed-dataset.uri" \
                "$DATA_DIR" \
                "DLRM-v2 Preprocessed Dataset"
        fi
        ;;
esac

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Download Complete!                            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Model directory:   ${MODEL_DIR}"
echo -e "Dataset directory: ${DATA_DIR}"
echo ""
echo -e "To run DLRM benchmark:"
echo -e "  ${YELLOW}./run_mlperf.sh dlrm --gpu${NC}"
echo -e "  ${YELLOW}./run_mlperf.sh dlrm --offload${NC}  (for large model)"
echo ""

# Save configuration
cat > "${PROJECT_DIR}/models/dlrm/dlrm_config.sh" << EOF
# DLRM Configuration
export DLRM_SIZE="${SIZE}"
export DLRM_MODEL_DIR="${MODEL_DIR}"
export DLRM_DATA_DIR="${DATA_DIR}"
export DLRM_DOWNLOADED="$(date -Iseconds)"
EOF

echo -e "Configuration saved to: ${PROJECT_DIR}/models/dlrm/dlrm_config.sh"
