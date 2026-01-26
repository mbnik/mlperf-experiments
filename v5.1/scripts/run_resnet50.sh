#!/bin/bash
# ============================================================================
# MLPerf Benchmark Setup and Runner - ResNet50 Image Classification
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
MLPERF_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Defaults
DEVICE="cuda"
USE_OFFLOAD=false
MAX_EXAMPLES=1000
DATA_TYPE="synthetic"
BATCH_SIZE=32
MLPERF_MODE=false
EXTRA_ARGS=()

# Help message
show_help() {
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║           ResNet50 Image Classification Benchmark          ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --gpu              Run on GPU with CUDA (default)"
    echo "  --cpu              Run on CPU only"
    echo "  --offload          Enable CPU offloading"
    echo "  --samples=N        Number of images (default: 1000)"
    echo "  --data=TYPE        Data type: synthetic, real (default: synthetic)"
    echo "  --batch=N          Batch size (default: 32)"
    echo "  --mlperf           Use official MLPerf settings (auto-downloads ImageNet subset)"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --gpu                         # Run on GPU with synthetic data"
    echo "  $0 --gpu --samples=5000          # Process 5000 images"
    echo "  $0 --gpu --data=real             # Use real ImageNet data"
    echo "  $0 --gpu --mlperf                # Official MLPerf settings"
    echo "  $0 --cpu --batch=8               # Run on CPU with smaller batch"
    echo ""
    exit 0
}

# Parse arguments
# First check for explicit --data=synthetic before parsing
EXPLICIT_SYNTHETIC=false
for arg in "$@"; do
    if [[ "$arg" == "--data=synthetic" ]]; then
        EXPLICIT_SYNTHETIC=true
        break
    fi
done

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            DEVICE="cuda"
            shift
            ;;
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        --offload)
            USE_OFFLOAD=true
            shift
            ;;
        --samples=*)
            MAX_EXAMPLES="${1#*=}"
            shift
            ;;
        --data=*)
            DATA_TYPE="${1#*=}"
            shift
            ;;
        --batch=*)
            BATCH_SIZE="${1#*=}"
            shift
            ;;
        --mlperf)
            MLPERF_MODE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Apply MLPerf settings
if [[ "$MLPERF_MODE" == "true" ]]; then
    if [[ "$EXPLICIT_SYNTHETIC" != "true" ]]; then
        DATA_TYPE="real"
    fi
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           ResNet50 Image Classification Benchmark          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo "  Device:   ${DEVICE}"
echo "  Offload:  ${USE_OFFLOAD}"
echo "  Data:     ${DATA_TYPE}"
echo "  Samples:  ${MAX_EXAMPLES}"
echo "  Batch:    ${BATCH_SIZE}"

# MLPerf status
if [[ "$MLPERF_MODE" == "true" ]]; then
    echo -e "  MLPerf:   ${GREEN}ENABLED${NC}"
    
    if [[ "$DATA_TYPE" == "synthetic" ]]; then
        echo ""
        echo -e "${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${YELLOW}║  ⚠️  SYNTHETIC DATA WITH MLPerf MODE                        ║${NC}"
        echo -e "${YELLOW}╠════════════════════════════════════════════════════════════╣${NC}"
        echo -e "${YELLOW}║  Results are NOT comparable to official MLPerf benchmarks  ║${NC}"
        echo -e "${YELLOW}║  For official comparison, use: --mlperf --data=real        ║${NC}"
        echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"
    fi
else
    echo -e "  MLPerf:   ${CYAN}disabled${NC} (use --mlperf for official settings)"
fi
echo ""

# Auto-download data if using real data
if [[ "$DATA_TYPE" == "real" ]]; then
    if [ ! -d "${MLPERF_ROOT}/data/imagenet/val" ] || [ -z "$(ls -A ${MLPERF_ROOT}/data/imagenet/val 2>/dev/null)" ]; then
        echo -e "${YELLOW}► ImageNet data not found, downloading via HuggingFace...${NC}"
        echo "  (Requires HuggingFace login with approved access to ILSVRC/imagenet-1k)"
        echo ""
        mkdir -p "${MLPERF_ROOT}/data/imagenet/val"
        
        export MLPERF_ROOT="${MLPERF_ROOT}"
        python3 << 'PYEOF'
from datasets import load_dataset
from PIL import Image
import os

val_dir = os.environ.get('MLPERF_ROOT', '.') + '/data/imagenet/val'
os.makedirs(val_dir, exist_ok=True)

print("  Loading ImageNet-1K validation subset (1000 images)...")
ds = load_dataset('ILSVRC/imagenet-1k', split='validation', streaming=True)

count = 0
for item in ds:
    if count >= 1000:
        break
    label_dir = os.path.join(val_dir, str(item['label']))
    os.makedirs(label_dir, exist_ok=True)
    img_path = os.path.join(label_dir, f'img_{count}.JPEG')
    item['image'].save(img_path)
    count += 1
    if count % 200 == 0:
        print(f"  Downloaded {count}/1000 images...")

print(f"  ✓ Saved {count} images to {val_dir}")
PYEOF
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Downloaded ImageNet subset${NC}"
        else
            echo -e "${RED}✗ Download failed. Make sure you're logged in: huggingface-cli login${NC}"
            echo -e "${YELLOW}Falling back to synthetic data...${NC}"
            DATA_TYPE="synthetic"
        fi
        echo ""
    fi
fi

# Build command
CMD="python ${SCRIPT_DIR}/run_resnet50_benchmark.py"
CMD="$CMD --device ${DEVICE}"
CMD="$CMD --max-examples ${MAX_EXAMPLES}"
CMD="$CMD --data-type ${DATA_TYPE}"
CMD="$CMD --batch-size ${BATCH_SIZE}"
CMD="$CMD --data-dir ${MLPERF_ROOT}/data/imagenet"
CMD="$CMD --output-dir ${MLPERF_ROOT}/results/resnet50"

[ "$USE_OFFLOAD" = true ] && CMD="$CMD --offload"
[ "$MLPERF_MODE" = true ] && CMD="$CMD --mlperf"

# Add extra arguments
CMD="$CMD ${EXTRA_ARGS[*]}"

echo "Running: $CMD"
echo ""

eval $CMD
