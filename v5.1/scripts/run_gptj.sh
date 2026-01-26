#!/bin/bash
# ============================================================================
# MLPerf Benchmark Setup and Runner - GPT-J Text Generation
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
MODEL_DIR="$PROJECT_DIR/models/gptj"
DATA_DIR="$PROJECT_DIR/data/cnn-dailymail"

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
    echo "Options:"
    echo "  --gpu           Run on GPU only (requires ~24GB VRAM full, ~12GB 8-bit, ~6GB 4-bit)"
    echo "  --cpu           Run on CPU only"
    echo "  --offload       Use GPU+RAM offloading for limited VRAM (FP16 only)"
    echo "  --4bit          Use 4-bit quantization (~6GB VRAM)"
    echo "  --8bit          Use 8-bit quantization (~12GB VRAM)"
    echo "  --data=TYPE     Data type: synthetic, real (default: synthetic)"
    echo "  --samples=N     Number of samples to process (default: 10)"
    echo "  --mlperf        Use official MLPerf settings (auto-downloads real data)"
    echo "  -h, --help      Show this help"
    echo ""
    echo "IMPORTANT: --4bit/--8bit and --offload cannot be used together!"
    echo "  bitsandbytes quantization does not support CPU offloading."
    echo ""
    echo "MLPerf Compliance:"
    echo "  --mlperf        Uses max_new_tokens=128, real CNN-DailyMail data"
    echo "                  Results are comparable to official MLPerf benchmarks"
    echo ""
    echo "Data Options:"
    echo "  synthetic  - Use predefined prompts (fast, no download)"
    echo "  real       - Download CNN-DailyMail dataset (~300MB)"
    echo ""
    echo "Examples:"
    echo "  $0 --offload --data=real          # FP16 with offloading"
    echo "  $0 --4bit --mlperf                # 4-bit quantization"
}

# Parse arguments
# First check for explicit --data=synthetic before parsing (to preserve the info)
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

echo -e "${CYAN}Running GPT-J Text Generation Benchmark${NC}"
echo -e "  Mode: ${BLUE}$DEVICE${NC}"
echo -e "  Data: ${CYAN}$DATA_TYPE${NC}"

# Apply MLPerf settings if --mlperf flag is set
if [[ "$MLPERF_MODE" == "true" ]]; then
    MAX_NEW_TOKENS=128  # GPT-J MLPerf uses 128 tokens
    
    # If user did NOT explicitly set --data=synthetic, default to real data
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
        echo -e "${CYAN}║           Existing CNN-DailyMail Data Detected             ║${NC}"
        echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "  ${GREEN}✓${NC} Dataset found: ${data_size_str} at $DATA_DIR"
        echo ""
        echo "Options:"
        echo "  [S] Skip download - Use existing data (default)"
        echo "  [R] Re-download data"
        echo "  [Q] Quit"
        echo ""
        
        read -t 30 -n 1 -p "" choice || choice="s"
        echo ""
        
        case ${choice,,} in
            r)
                echo -e "${YELLOW}Re-downloading...${NC}"
                SKIP_DOWNLOAD=false
                ;;
            q)
                echo -e "${RED}Exiting.${NC}"
                exit 0
                ;;
            *)
                echo -e "${GREEN}Using existing data...${NC}"
                SKIP_DOWNLOAD=true
                ;;
        esac
    fi
}

# ============================================================================
# Download CNN-DailyMail
# ============================================================================
download_cnn_dailymail() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║           Downloading CNN-DailyMail Dataset                ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}CNN-DailyMail (Official MLPerf dataset for text summarization)${NC}"
    echo "  - News articles with summaries"
    echo "  - Test set: ~11,490 articles"
    echo "  - Download via HuggingFace datasets"
    echo ""
    
    mkdir -p "$DATA_DIR"
    
    python3 << PYTHON_EOF
import os
import json
from datasets import load_dataset

data_dir = "$DATA_DIR"

print("Downloading CNN-DailyMail from HuggingFace...")
dataset = load_dataset("cnn_dailymail", "3.0.0", trust_remote_code=True)

# Save test set
test_data = []
for item in dataset["test"]:
    test_data.append({
        "article": item["article"],
        "highlights": item["highlights"],
        "id": item["id"]
    })

test_path = os.path.join(data_dir, "test.json")
with open(test_path, 'w') as f:
    json.dump(test_data, f)

print(f"✓ Saved {len(test_data)} test articles to {test_path}")

# Also save a small validation subset
val_data = []
for item in list(dataset["validation"])[:1000]:
    val_data.append({
        "article": item["article"],
        "highlights": item["highlights"],
        "id": item["id"]
    })

val_path = os.path.join(data_dir, "validation.json")
with open(val_path, 'w') as f:
    json.dump(val_data, f)

print(f"✓ Saved {len(val_data)} validation articles to {val_path}")
PYTHON_EOF

    echo -e "${GREEN}✓ CNN-DailyMail ready!${NC}"
}

# ============================================================================
# Main execution
# ============================================================================

if [ "$DATA_TYPE" = "real" ]; then
    check_existing_data
    if [ "$SKIP_DOWNLOAD" = false ]; then
        download_cnn_dailymail
    fi
fi

mkdir -p "${PROJECT_DIR}/results/gptj"

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              GPT-J Benchmark Configuration                 ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo "  Model:    EleutherAI/gpt-j-6b"
echo "  Device:   $DEVICE"
echo "  Offload:  $USE_OFFLOAD"
echo "  Quant:    $QUANTIZATION"
echo "  Data:     $DATA_TYPE"
echo "  Samples:  $MAX_EXAMPLES"

# MLPerf status
if [[ "$MLPERF_MODE" == "true" ]]; then
    echo -e "  MLPerf:   ${GREEN}ENABLED${NC} (max_new_tokens=${MAX_NEW_TOKENS})"
    
    # Show warning if using synthetic data
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

# Check for incompatible combination: quantization + offload
if [[ "$USE_OFFLOAD" == "true" && "$QUANTIZATION" != "none" ]]; then
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ⚠️  INCOMPATIBLE OPTIONS: --${QUANTIZATION} + --offload   ║${NC}"
    echo -e "${RED}╠════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${RED}║  bitsandbytes quantization does NOT support CPU offloading ║${NC}"
    echo -e "${RED}║                                                            ║${NC}"
    echo -e "${RED}║  Choose ONE of these options:                              ║${NC}"
    echo -e "${RED}║    --4bit     : 4-bit quantization (~6GB VRAM)             ║${NC}"
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
CMD="python ${SCRIPT_DIR}/run_gptj_benchmark.py"
CMD="$CMD --device $DEVICE"
CMD="$CMD --max-examples $MAX_EXAMPLES"
CMD="$CMD --data-type $DATA_TYPE"
CMD="$CMD --data-dir $DATA_DIR"
CMD="$CMD --max-new-tokens $MAX_NEW_TOKENS"
CMD="$CMD --output-dir ${PROJECT_DIR}/results/gptj"

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
        echo -e "${RED}║  GPT-J 6B requires ~12GB VRAM in FP16 mode.                ║${NC}"
        echo -e "${RED}║                                                            ║${NC}"
        echo -e "${RED}║  Solutions:                                                ║${NC}"
        echo -e "${RED}║  1. Use --offload to enable CPU offloading                 ║${NC}"
        echo -e "${RED}║  2. Use --4bit for 4-bit quantization (~6GB VRAM)          ║${NC}"
        echo -e "${RED}║                                                            ║${NC}"
        echo -e "${RED}║  Example: $0 --offload --mlperf                ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
    fi
    exit $EXIT_CODE
fi
