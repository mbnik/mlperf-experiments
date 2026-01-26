#!/bin/bash
# ============================================================================
# MLPerf Benchmark Setup and Runner - Whisper Speech Recognition
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
MODEL_DIR="$PROJECT_DIR/models/whisper"
DATA_DIR="$PROJECT_DIR/data/librispeech"

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
MAX_EXAMPLES=10
DATA_TYPE="synthetic"
SKIP_DOWNLOAD=false
MLPERF_MODE=false

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --gpu           Run on GPU only"
    echo "  --cpu           Run on CPU only"
    echo "  --offload       Use GPU+RAM offloading"
    echo "  --data=TYPE     Data type: synthetic, real (default: synthetic)"
    echo "  --samples=N     Number of samples to process (default: 10)"
    echo "  -h, --help      Show this help"
    echo ""
    echo "Data Options:"
    echo "  synthetic  - Generate synthetic audio (fast, no download)"
    echo "  real       - Download LibriSpeech test-clean (~350MB)"
    echo ""
    echo "Examples:"
    echo "  $0 --gpu --data=real              # Real LibriSpeech data"
    echo "  $0 --gpu --data=synthetic         # Synthetic audio (fast)"
    echo "  $0 --gpu --mlperf                 # Official MLPerf settings"
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
    if [[ "$EXPLICIT_SYNTHETIC" != "true" ]]; then
        DATA_TYPE="real"
    fi
fi

echo -e "${CYAN}Running Whisper Speech Recognition Benchmark${NC}"
echo -e "  Mode: ${BLUE}$DEVICE${NC}"
echo -e "  Data: ${CYAN}$DATA_TYPE${NC}"

# MLPerf status
if [[ "$MLPERF_MODE" == "true" ]]; then
    echo -e "  MLPerf: ${GREEN}ENABLED${NC}"
    
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
    
    if [ "$DATA_TYPE" = "real" ] && [ -d "$DATA_DIR/LibriSpeech" ]; then
        local data_size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
        data_exists=true
        data_size_str="$data_size"
    fi
    
    if $data_exists; then
        echo ""
        echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${CYAN}║              Existing LibriSpeech Data Detected            ║${NC}"
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
# Download LibriSpeech
# ============================================================================
download_librispeech() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║              Downloading LibriSpeech Dataset               ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}LibriSpeech test-clean (Official MLPerf dataset)${NC}"
    echo "  - 2,620 audio samples from read audiobooks"
    echo "  - ~5.4 hours of speech"
    echo "  - Download size: ~350MB"
    echo ""
    
    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"
    
    # Download test-clean subset
    LIBRI_URL="https://www.openslr.org/resources/12/test-clean.tar.gz"
    LIBRI_FILE="test-clean.tar.gz"
    
    if [ ! -f "$LIBRI_FILE" ]; then
        echo -e "${CYAN}Downloading LibriSpeech test-clean...${NC}"
        wget -q --show-progress "$LIBRI_URL" -O "$LIBRI_FILE"
    else
        echo -e "${GREEN}✓ Archive already downloaded${NC}"
    fi
    
    # Extract
    if [ ! -d "LibriSpeech/test-clean" ]; then
        echo -e "${CYAN}Extracting...${NC}"
        tar -xzf "$LIBRI_FILE"
    else
        echo -e "${GREEN}✓ Already extracted${NC}"
    fi
    
    # Create manifest file for easy loading
    echo -e "${CYAN}Creating audio manifest...${NC}"
    python3 << 'PYTHON_EOF'
import os
import json
from pathlib import Path

data_dir = os.environ.get('DATA_DIR', 'data/librispeech')
libri_dir = Path(data_dir) / "LibriSpeech" / "test-clean"

manifest = []
for speaker_dir in sorted(libri_dir.iterdir()):
    if not speaker_dir.is_dir():
        continue
    for chapter_dir in sorted(speaker_dir.iterdir()):
        if not chapter_dir.is_dir():
            continue
        # Read transcript
        trans_file = list(chapter_dir.glob("*.trans.txt"))[0]
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    audio_id, text = parts
                    audio_path = chapter_dir / f"{audio_id}.flac"
                    if audio_path.exists():
                        manifest.append({
                            "audio_path": str(audio_path),
                            "text": text,
                            "speaker_id": speaker_dir.name,
                            "chapter_id": chapter_dir.name,
                        })

# Save manifest
manifest_path = Path(data_dir) / "manifest.json"
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"✓ Created manifest with {len(manifest)} samples")
print(f"  Location: {manifest_path}")
PYTHON_EOF

    cd "$PROJECT_DIR"
    echo -e "${GREEN}✓ LibriSpeech ready!${NC}"
}

# ============================================================================
# Main execution
# ============================================================================

if [ "$DATA_TYPE" = "real" ]; then
    check_existing_data
    if [ "$SKIP_DOWNLOAD" = false ]; then
        export DATA_DIR
        download_librispeech
    fi
fi

mkdir -p "${PROJECT_DIR}/results/whisper"

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              Whisper Benchmark Configuration               ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo "  Model:    openai/whisper-large-v3"
echo "  Device:   $DEVICE"
echo "  Offload:  $USE_OFFLOAD"
echo "  Data:     $DATA_TYPE"
echo "  Samples:  $MAX_EXAMPLES"
echo ""

# Build command
CMD="python ${SCRIPT_DIR}/run_whisper_benchmark.py"
CMD="$CMD --model-name openai/whisper-large-v3"
CMD="$CMD --device $DEVICE"
CMD="$CMD --max-examples $MAX_EXAMPLES"
CMD="$CMD --data-type $DATA_TYPE"
CMD="$CMD --data-dir $DATA_DIR"
CMD="$CMD --output-dir ${PROJECT_DIR}/results/whisper"

[ "$USE_OFFLOAD" = true ] && CMD="$CMD --offload"
[ "$MLPERF_MODE" = true ] && CMD="$CMD --mlperf"

exec $CMD
