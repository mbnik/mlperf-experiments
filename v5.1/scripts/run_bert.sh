#!/bin/bash
# ============================================================================
# MLPerf Benchmark Setup and Runner - BERT Question Answering
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
MAX_EXAMPLES=100
DATA_TYPE="synthetic"
BATCH_SIZE=8
MLPERF_MODE=false
DOWNLOAD_METHOD="curl"
EXTRA_ARGS=()

# Help message
show_help() {
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║             BERT Question Answering Benchmark              ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --gpu              Run on GPU with CUDA (default)"
    echo "  --cpu              Run on CPU only"
    echo "  --offload          Enable CPU offloading"
    echo "  --samples=N        Number of samples (default: 100)"
    echo "  --data=TYPE        Data type: synthetic, real (default: synthetic)"
    echo "  --batch=N          Batch size (default: 8)"
    echo "  --download=M       Download method: curl (default), wget"
    echo "  --mlperf           Use official MLPerf settings (auto-downloads real data)"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --gpu                         # Run on GPU with synthetic data"
    echo "  $0 --gpu --samples=500           # Process 500 samples"
    echo "  $0 --gpu --data=real             # Use real SQuAD data"
    echo "  $0 --cpu --batch=4               # Run on CPU with smaller batch"
    echo "  $0 --gpu --mlperf                # Official MLPerf settings"
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
        --download=*)
            DOWNLOAD_METHOD="${1#*=}"
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
echo "║             BERT Question Answering Benchmark              ║"
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
    if [ ! -f "${MLPERF_ROOT}/data/squad/dev-v1.1.json" ]; then
        echo -e "${YELLOW}► SQuAD data not found, downloading...${NC}"
        mkdir -p "${MLPERF_ROOT}/data/squad"
        SQUAD_URL="https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
        
        if [[ "$DOWNLOAD_METHOD" == "wget" ]]; then
            wget -q --show-progress -O "${MLPERF_ROOT}/data/squad/dev-v1.1.json" "$SQUAD_URL"
        else
            curl -L -o "${MLPERF_ROOT}/data/squad/dev-v1.1.json" "$SQUAD_URL"
        fi
        echo -e "${GREEN}✓ Downloaded SQuAD v1.1${NC}"
        echo ""
    fi
fi

# Build command
CMD="python ${SCRIPT_DIR}/run_bert_benchmark.py"
CMD="$CMD --device ${DEVICE}"
CMD="$CMD --max-examples ${MAX_EXAMPLES}"
CMD="$CMD --data-type ${DATA_TYPE}"
CMD="$CMD --batch-size ${BATCH_SIZE}"
CMD="$CMD --data-dir ${MLPERF_ROOT}/data/squad"
CMD="$CMD --output-dir ${MLPERF_ROOT}/results/bert"

[ "$USE_OFFLOAD" = true ] && CMD="$CMD --offload"
[ "$MLPERF_MODE" = true ] && CMD="$CMD --mlperf"

# Add extra arguments
CMD="$CMD ${EXTRA_ARGS[*]}"

echo "Running: $CMD"
echo ""

eval $CMD
