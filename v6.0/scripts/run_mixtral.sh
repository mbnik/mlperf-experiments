#!/bin/bash
# ============================================================================
# MLPerf Benchmark Setup and Runner - Mixtral-8x7B Text Generation
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
DATA_DIR="$PROJECT_DIR/data/openorca"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default settings
DEVICE="cuda"
USE_OFFLOAD=false
QUANTIZATION="none"
MAX_EXAMPLES=10
DATA_TYPE="synthetic"
SKIP_DOWNLOAD=false
MLPERF_MODE=false
MAX_NEW_TOKENS=128

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Mixtral-8x7B is a large Mixture-of-Experts model (~93GB parameters)"
    echo ""
    echo "Options:"
    echo "  --gpu           Run on GPU (requires ~90GB VRAM full, ~48GB 8-bit, ~24GB 4-bit)"
    echo "  --cpu           Run on CPU only (very slow)"
    echo "  --offload       Use GPU+RAM offloading for limited VRAM (FP16 only)"
    echo "  --4bit          Use 4-bit quantization (~24GB VRAM)"
    echo "  --8bit          Use 8-bit quantization (~48GB VRAM)"
    echo "  --data=TYPE     Data type: synthetic, real (default: synthetic)"
    echo "  --samples=N     Number of samples to process (default: 10)"
    echo "  --mlperf        Use official MLPerf settings (auto-downloads real data)"
    echo "  -h, --help      Show this help"
    echo ""
    echo "IMPORTANT: --4bit/--8bit and --offload cannot be used together!"
    echo "  bitsandbytes quantization does not support CPU offloading."
    echo "  Choose either quantization OR offloading, not both."
    echo ""
    echo "MLPerf Compliance:"
    echo "  --mlperf        Uses max_new_tokens=1024, real OpenOrca data"
    echo "                  Results are comparable to official MLPerf benchmarks"
    echo ""
    echo "Data Options:"
    echo "  synthetic  - Use predefined prompts (fast, no download)"
    echo "  real       - Use OpenOrca dataset (MLPerf official)"
    echo ""
    echo "Examples:"
    echo "  $0 --4bit --mlperf                # 4-bit quantization (~24GB VRAM)"
    echo "  $0 --offload --mlperf             # FP16 with CPU offload (slower)"
    echo "  $0 --gpu --data=synthetic         # Synthetic prompts (large VRAM)"
}

# Parse arguments
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
        --4bit)
            QUANTIZATION="4bit"
            shift
            ;;
        --8bit)
            QUANTIZATION="8bit"
            shift
            ;;
        --data=*)
            DATA_TYPE="${1#*=}"
            shift
            ;;
        --samples=*)
            MAX_EXAMPLES="${1#*=}"
            shift
            ;;
        --mlperf)
            MLPERF_MODE=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

echo -e "${CYAN}Running Mixtral-8x7B Text Generation Benchmark${NC}"
echo -e "  Mode: ${BLUE}$DEVICE${NC}"
echo -e "  Data: ${CYAN}$DATA_TYPE${NC}"

# Apply MLPerf settings if --mlperf flag is set
if [[ "$MLPERF_MODE" == "true" ]]; then
    MAX_NEW_TOKENS=1024
    
    if [[ "$EXPLICIT_SYNTHETIC" != "true" ]]; then
        DATA_TYPE="real"
    fi
fi

# ============================================================================
# Check for existing data
# ============================================================================
check_existing_data() {
    local data_exists=false
    local data_size_str=""
    
    if [ "$DATA_TYPE" = "real" ] && [ -f "$DATA_DIR/test.json" ]; then
        local data_size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
        data_exists=true
        data_size_str="$data_size"
    fi
    
    if $data_exists; then
        echo ""
        echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${CYAN}║             Existing OpenOrca Data Detected                ║${NC}"
        echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "  ${GREEN}✓${NC} Dataset found: ${data_size_str} at $DATA_DIR"
        echo ""
        SKIP_DOWNLOAD=true
    fi
}

# ============================================================================
# Download OpenOrca Dataset
# ============================================================================
download_openorca() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║             Downloading OpenOrca Dataset                   ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}OpenOrca (MLPerf official dataset for Mixtral)${NC}"
    echo "  - Instruction-following dataset"
    echo "  - Download via HuggingFace datasets"
    echo ""
    
    mkdir -p "$DATA_DIR"
    
    python3 << PYTHON_EOF
import os
import json
from datasets import load_dataset

data_dir = "$DATA_DIR"

print("Downloading OpenOrca from HuggingFace...")
# Using a smaller subset for testing
dataset = load_dataset("Open-Orca/OpenOrca", split="train[:5000]", trust_remote_code=True)

# Save as test set
test_data = []
for item in dataset:
    test_data.append({
        "question": item.get("question", ""),
        "response": item.get("response", ""),
        "system_prompt": item.get("system_prompt", ""),
    })

test_path = os.path.join(data_dir, "test.json")
with open(test_path, 'w') as f:
    json.dump(test_data, f)

print(f"✓ Saved {len(test_data)} examples to {test_path}")
PYTHON_EOF

    echo -e "${GREEN}✓ OpenOrca ready!${NC}"
}

# ============================================================================
# Main execution
# ============================================================================

if [ "$DATA_TYPE" = "real" ]; then
    check_existing_data
    if [ "$SKIP_DOWNLOAD" = false ]; then
        download_openorca
    fi
fi

mkdir -p "${PROJECT_DIR}/results/mixtral-8x7b"

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           Mixtral-8x7B Benchmark Configuration             ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo "  Model:    mistralai/Mixtral-8x7B-Instruct-v0.1"
echo "  Device:   $DEVICE"
echo "  Offload:  $USE_OFFLOAD"
echo "  Quant:    $QUANTIZATION"
echo "  Data:     $DATA_TYPE"
echo "  Samples:  $MAX_EXAMPLES"

# MLPerf status
if [[ "$MLPERF_MODE" == "true" ]]; then
    echo -e "  MLPerf:   ${GREEN}ENABLED${NC} (max_new_tokens=${MAX_NEW_TOKENS})"
    
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

# Recommend offload for Mixtral
if [[ "$USE_OFFLOAD" == "false" && "$QUANTIZATION" == "none" && "$DEVICE" == "cuda" ]]; then
    echo -e "${YELLOW}⚠️  Warning: Mixtral-8x7B requires ~90GB VRAM without quantization${NC}"
    echo -e "${YELLOW}   Consider using: --4bit or --offload (but not both together)${NC}"
    echo ""
fi

# Check for incompatible combination: quantization + offload
if [[ "$USE_OFFLOAD" == "true" && "$QUANTIZATION" != "none" ]]; then
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ⚠️  INCOMPATIBLE OPTIONS: --${QUANTIZATION} + --offload   ║${NC}"
    echo -e "${RED}╠════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${RED}║  bitsandbytes quantization does NOT support CPU offloading ║${NC}"
    echo -e "${RED}║                                                            ║${NC}"
    echo -e "${RED}║  Choose ONE of these options:                              ║${NC}"
    echo -e "${RED}║    --4bit     : 4-bit quantization (~24GB VRAM)            ║${NC}"
    echo -e "${RED}║    --offload  : FP16 with CPU offloading (slower)          ║${NC}"
    echo -e "${RED}║                                                            ║${NC}"
    echo -e "${RED}║  Examples:                                                 ║${NC}"
    echo -e "${RED}║    $0 --4bit --mlperf                                      ║${NC}"
    echo -e "${RED}║    $0 --offload --mlperf                                   ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    exit 1
fi

# Build command
CMD="python ${SCRIPT_DIR}/run_mixtral_benchmark.py"
CMD="$CMD --device $DEVICE"
CMD="$CMD --max-examples $MAX_EXAMPLES"
CMD="$CMD --data-type $DATA_TYPE"
CMD="$CMD --data-dir $DATA_DIR"
CMD="$CMD --max-new-tokens $MAX_NEW_TOKENS"
CMD="$CMD --output-dir ${PROJECT_DIR}/results/mixtral-8x7b"

[ "$USE_OFFLOAD" = true ] && CMD="$CMD --offload"
[ "$QUANTIZATION" = "4bit" ] && CMD="$CMD --4bit"
[ "$QUANTIZATION" = "8bit" ] && CMD="$CMD --8bit"
[ "$MLPERF_MODE" = true ] && CMD="$CMD --mlperf"

# Run with OOM error handling
set +e
$CMD
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -ne 0 ]; then
    # Check if it was an OOM error
    if [ $EXIT_CODE -eq 1 ] && [ "$USE_OFFLOAD" = false ] && [ "$DEVICE" = "cuda" ]; then
        echo ""
        echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║  ⚠️  GPU OUT OF MEMORY ERROR                                ║${NC}"
        echo -e "${RED}╠════════════════════════════════════════════════════════════╣${NC}"
        echo -e "${RED}║  Mixtral-8x7B requires ~90GB VRAM without quantization.    ║${NC}"
        echo -e "${RED}║                                                            ║${NC}"
        echo -e "${RED}║  Solutions:                                                ║${NC}"
        echo -e "${RED}║  1. Use --4bit for 4-bit quantization (~26GB VRAM)         ║${NC}"
        echo -e "${RED}║  2. Use --offload to enable CPU offloading                 ║${NC}"
        echo -e "${RED}║  3. Combine both: --4bit --offload (lowest VRAM)           ║${NC}"
        echo -e "${RED}║                                                            ║${NC}"
        echo -e "${RED}║  Example: $0 --4bit --offload --mlperf         ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
    fi
    exit $EXIT_CODE
fi
