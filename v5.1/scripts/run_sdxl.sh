#!/bin/bash
# ============================================================================
# MLPerf Benchmark Setup and Runner - Stable Diffusion XL Image Generation
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
MODEL_DIR="$PROJECT_DIR/models/sdxl"
DATA_DIR="$PROJECT_DIR/data/coco-2014"

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
MAX_EXAMPLES=5
DATA_TYPE="synthetic"
SKIP_DOWNLOAD=false
NUM_STEPS=20
MLPERF_MODE=false

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --gpu           Run on GPU only"
    echo "  --cpu           Run on CPU only (very slow)"
    echo "  --offload       Use GPU+RAM offloading (default)"
    echo "  --data=TYPE     Data type: synthetic, real (default: synthetic)"
    echo "  --samples=N     Number of images to generate (default: 5)"
    echo "  --steps=N       Number of diffusion steps (default: 20)"
    echo "  -h, --help      Show this help"
    echo ""
    echo "Data Options:"
    echo "  synthetic  - Use predefined prompts (fast, no download)"
    echo "  real       - Download COCO 2014 captions (~1MB)"
    echo ""
    echo "Examples:"
    echo "  $0 --offload --data=real          # Real COCO captions"
    echo "  $0 --offload --data=synthetic     # Synthetic prompts (fast)"
    echo "  $0 --offload --mlperf             # Official MLPerf settings"
}

# Parse arguments
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
        --data=*)
            DATA_TYPE="${1#*=}"
            shift
            ;;
        --samples=*)
            MAX_EXAMPLES="${1#*=}"
            shift
            ;;
        --steps=*)
            NUM_STEPS="${1#*=}"
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

# Check for explicit synthetic
EXPLICIT_SYNTHETIC=false
for arg in "$@"; do
    if [[ "$arg" == "--data=synthetic" ]]; then
        EXPLICIT_SYNTHETIC=true
        break
    fi
done

# Apply MLPerf settings
if [[ "$MLPERF_MODE" == "true" ]]; then
    NUM_STEPS=20  # Official MLPerf uses 20 steps
    if [[ "$EXPLICIT_SYNTHETIC" != "true" ]]; then
        DATA_TYPE="real"
    fi
fi

echo -e "${CYAN}Running Stable Diffusion XL Image Generation Benchmark${NC}"
echo -e "  Mode: ${BLUE}$DEVICE${NC}"
echo -e "  Data: ${CYAN}$DATA_TYPE${NC}"

# MLPerf status
if [[ "$MLPERF_MODE" == "true" ]]; then
    echo -e "  MLPerf: ${GREEN}ENABLED${NC} (steps=${NUM_STEPS})"
    
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
    echo -e "  MLPerf: ${CYAN}disabled${NC} (use --mlperf for official settings)"
fi

# ============================================================================
# Check for existing data
# ============================================================================
check_existing_data() {
    local data_exists=false
    local data_size_str=""
    
    if [ "$DATA_TYPE" = "real" ] && [ -f "$DATA_DIR/captions.json" ]; then
        local data_size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
        data_exists=true
        data_size_str="$data_size"
    fi
    
    if $data_exists; then
        echo ""
        echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${CYAN}║           Existing COCO Captions Data Detected             ║${NC}"
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
# Download COCO Captions
# ============================================================================
download_coco_captions() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║           Downloading COCO 2014 Captions                   ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}COCO 2014 Captions (Official MLPerf prompts for image generation)${NC}"
    echo "  - Real image descriptions from COCO dataset"
    echo "  - 5,000 validation captions"
    echo "  - Download size: ~1MB (annotations only)"
    echo ""
    
    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"
    
    # Download COCO 2014 validation annotations
    COCO_URL="http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    
    if [ ! -f "annotations_trainval2014.zip" ] && [ ! -f "captions.json" ]; then
        echo -e "${CYAN}Downloading COCO 2014 annotations...${NC}"
        wget -q --show-progress "$COCO_URL" -O annotations_trainval2014.zip
        unzip -q annotations_trainval2014.zip
    fi
    
    # Extract captions
    echo -e "${CYAN}Extracting captions...${NC}"
    python3 << 'PYTHON_EOF'
import os
import json
from pathlib import Path

data_dir = os.environ.get('DATA_DIR', 'data/coco-2014')

# Load COCO annotations
ann_file = Path(data_dir) / "annotations" / "captions_val2014.json"

if ann_file.exists():
    with open(ann_file) as f:
        coco = json.load(f)
    
    # Extract unique captions (one per image)
    image_captions = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_captions:
            image_captions[img_id] = ann["caption"]
    
    captions = list(image_captions.values())
    
    # Save captions
    captions_file = Path(data_dir) / "captions.json"
    with open(captions_file, 'w') as f:
        json.dump(captions, f, indent=2)
    
    print(f"✓ Extracted {len(captions)} captions")
    print(f"  Location: {captions_file}")
else:
    print(f"✗ Annotations not found at {ann_file}")
PYTHON_EOF

    cd "$PROJECT_DIR"
    echo -e "${GREEN}✓ COCO captions ready!${NC}"
}

# ============================================================================
# Main execution
# ============================================================================

if [ "$DATA_TYPE" = "real" ]; then
    check_existing_data
    if [ "$SKIP_DOWNLOAD" = false ]; then
        export DATA_DIR
        download_coco_captions
    fi
fi

mkdir -p "${PROJECT_DIR}/results/sdxl"

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              SDXL Benchmark Configuration                  ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo "  Model:    stabilityai/stable-diffusion-xl-base-1.0"
echo "  Device:   $DEVICE"
echo "  Offload:  $USE_OFFLOAD"
echo "  Data:     $DATA_TYPE"
echo "  Samples:  $MAX_EXAMPLES"
echo "  Steps:    $NUM_STEPS"
echo ""

# Build command
CMD="python ${SCRIPT_DIR}/run_sdxl_benchmark.py"
CMD="$CMD --device $DEVICE"
CMD="$CMD --max-examples $MAX_EXAMPLES"
CMD="$CMD --num-steps $NUM_STEPS"
CMD="$CMD --data-type $DATA_TYPE"
CMD="$CMD --data-dir $DATA_DIR"
CMD="$CMD --output-dir ${PROJECT_DIR}/results/sdxl"

[ "$USE_OFFLOAD" = true ] && CMD="$CMD --offload"
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
        echo -e "${RED}║  SDXL requires ~6.5GB VRAM for full GPU mode.              ║${NC}"
        echo -e "${RED}║                                                            ║${NC}"
        echo -e "${RED}║  Solutions:                                                ║${NC}"
        echo -e "${RED}║  1. Use --offload to enable CPU offloading (~3GB VRAM)     ║${NC}"
        echo -e "${RED}║     Example: $0 --offload --mlperf             ║${NC}"
        echo -e "${RED}║                                                            ║${NC}"
        echo -e "${RED}║  2. Use --cpu to run on CPU only (very slow)               ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
    fi
    exit $EXIT_CODE
fi
