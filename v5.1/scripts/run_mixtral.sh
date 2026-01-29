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
DATA_DIR="$PROJECT_DIR/data/mixtral"

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
    echo "  --mlperf        Uses max_new_tokens=1024, official 15K combined dataset"
    echo "                  (OpenOrca + GSM8k + MBXP - 5K samples each)"
    echo "                  Results are comparable to official MLPerf benchmarks"
    echo ""
    echo "Data Options:"
    echo "  synthetic  - Use predefined prompts (fast, no download)"
    echo "  real       - Use MLPerf 15K combined dataset (OpenOrca+GSM8k+MBXP)"
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
        echo -e "${CYAN}║         Existing MLPerf Mixtral Data Detected              ║${NC}"
        echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "  ${GREEN}✓${NC} Dataset found: ${data_size_str} at $DATA_DIR"
        echo ""
        SKIP_DOWNLOAD=true
    fi
}

# ============================================================================
# Download MLPerf Combined Dataset (OpenOrca + GSM8k + MBXP)
# ============================================================================
download_mixtral_dataset() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║         Downloading MLPerf Mixtral Dataset                 ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}MLPerf Mixtral Combined Dataset (Official)${NC}"
    echo "  - 15,000 samples total:"
    echo "    • 5,000 from OpenOrca (instruction following)"
    echo "    • 5,000 from GSM8k (math problems)"
    echo "    • 5,000 from MBXP (code generation)"
    echo "  - Preprocessed .pkl format"
    echo ""
    
    mkdir -p "$DATA_DIR"
    
    local pkl_file="${DATA_DIR}/09292024_mixtral_15k_mintoken2_v1.pkl"
    local dataset_url="https://inference.mlcommons-storage.org/mixtral_8x7b/09292024_mixtral_15k_mintoken2_v1.pkl"
    
    if [ ! -f "$pkl_file" ]; then
        echo -e "${CYAN}Downloading MLPerf combined dataset (~100MB)...${NC}"
        wget -q --show-progress -O "$pkl_file" "$dataset_url"
    else
        echo -e "${GREEN}✓ Dataset already downloaded${NC}"
    fi
    
    # Convert pkl to JSON for our benchmark runner
    echo -e "${CYAN}Processing dataset...${NC}"
    python3 << PYTHON_EOF
import os
import json
import pickle

data_dir = "$DATA_DIR"
pkl_path = os.path.join(data_dir, "09292024_mixtral_15k_mintoken2_v1.pkl")

print(f"Loading MLPerf dataset from {pkl_path}...")
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print(f"Dataset type: {type(data)}")
if isinstance(data, dict):
    print(f"Keys: {list(data.keys())}")

# Convert to our format
test_data = []
if isinstance(data, list):
    for item in data:
        if isinstance(item, dict):
            test_data.append({
                "question": item.get("input", item.get("question", item.get("prompt", ""))),
                "response": item.get("output", item.get("response", item.get("target", ""))),
                "source": item.get("source", "unknown"),
            })
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            test_data.append({
                "question": str(item[0]),
                "response": str(item[1]),
                "source": "combined",
            })
elif isinstance(data, dict):
    # Handle dict format
    if 'input_ids' in data or 'prompts' in data:
        prompts = data.get('prompts', data.get('input', []))
        responses = data.get('responses', data.get('output', []))
        sources = data.get('sources', ['unknown'] * len(prompts))
        for i, (p, r) in enumerate(zip(prompts, responses)):
            test_data.append({
                "question": str(p),
                "response": str(r),
                "source": sources[i] if i < len(sources) else "unknown",
            })
    else:
        # Try to iterate items
        for k, v in data.items():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        test_data.append({
                            "question": item.get("input", item.get("question", "")),
                            "response": item.get("output", item.get("response", "")),
                            "source": k,
                        })

if not test_data:
    print("Warning: Could not parse dataset, trying pandas...")
    try:
        import pandas as pd
        df = pd.read_pickle(pkl_path)
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        for idx, row in df.iterrows():
            test_data.append({
                "question": str(row.get('input', row.get('prompt', row.iloc[0] if len(row) > 0 else ''))),
                "response": str(row.get('output', row.get('target', row.iloc[1] if len(row) > 1 else ''))),
                "source": str(row.get('source', 'combined')),
            })
    except Exception as e:
        print(f"Pandas fallback failed: {e}")

test_path = os.path.join(data_dir, "test.json")
with open(test_path, 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"✓ Processed {len(test_data)} samples")
print(f"  Saved to: {test_path}")

# Show source distribution
sources = {}
for item in test_data:
    src = item.get('source', 'unknown')
    sources[src] = sources.get(src, 0) + 1
print(f"  Source distribution: {sources}")
PYTHON_EOF

    echo -e "${GREEN}✓ MLPerf Mixtral dataset ready!${NC}"
}

# ============================================================================
# Main execution
# ============================================================================

if [ "$DATA_TYPE" = "real" ]; then
    check_existing_data
    if [ "$SKIP_DOWNLOAD" = false ]; then
        download_mixtral_dataset
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
        echo -e "${RED}║  ⚠️  GPU OUT OF MEMORY ERROR                               ║${NC}"
        echo -e "${RED}╠════════════════════════════════════════════════════════════╣${NC}"
        echo -e "${RED}║  Mixtral-8x7B requires ~90GB VRAM without quantization.    ║${NC}"
        echo -e "${RED}║                                                            ║${NC}"
        echo -e "${RED}║  Solutions:                                                ║${NC}"
        echo -e "${RED}║  1. Use --4bit for 4-bit quantization (~26GB VRAM)         ║${NC}"
        echo -e "${RED}║  2. Use --offload to enable CPU offloading                 ║${NC}"
        echo -e "${RED}║  3. Combine both: --4bit --offload (lowest VRAM)           ║${NC}"
        echo -e "${RED}║                                                            ║${NC}"
        echo -e "${RED}║  Example: $0 --4bit --offload --mlperf                     ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
    fi
    exit $EXIT_CODE
fi
