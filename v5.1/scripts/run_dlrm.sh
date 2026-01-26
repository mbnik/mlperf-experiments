#!/bin/bash
# ============================================================================
# MLPerf Benchmark Setup and Runner - DLRM Recommendation
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
CYAN='\033[0;36m'
NC='\033[0m'

# Defaults
DEVICE="cuda"
USE_OFFLOAD=false
MODEL_SIZE="small"
DATA_TYPE="synthetic"
MAX_EXAMPLES=""
BATCH_SIZE=""
FORCE_DOWNLOAD=false
SKIP_MODEL_DOWNLOAD=false
MLPERF_MODE=false
EXTRA_ARGS=()

print_usage() {
    echo -e "${CYAN}DLRM-v2 Benchmark Runner${NC}"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Device Options:"
    echo "  --gpu         Run on GPU (default)"
    echo "  --cpu         Run on CPU"
    echo "  --offload     GPU+CPU offloading (recommended for large models)"
    echo ""
    echo "Model Size Options:"
    echo "  --size=small  Debug model (~10MB) - quick testing"
    echo "  --size=sample Full model (~97GB) with sample synthetic data"
    echo "  --size=full   Full model (~97GB) with larger synthetic data"
    echo ""
    echo "Data Options:"
    echo "  --data=synthetic  Use synthetic Criteo-like data (default)"
    echo "  --data=real       Download Criteo Terabyte dataset (~1TB)"
    echo "                    Note: Real Criteo data requires significant storage"
    echo ""
    echo "Other Options:"
    echo "  --force-download  Re-download even if data exists"
    echo "  --max-samples=N   Limit number of samples"
    echo "  --batch-size=N    Set batch size"
    echo "  --mlperf          Use official MLPerf settings"
    echo ""
    echo "Real Data Note:"
    echo "  The Criteo Terabyte dataset is very large (~1TB)."
    echo "  For MLPerf official submission, Day 23 is used for validation."
    echo "  Download from: https://ailab.criteo.com/ressources/"
    echo ""
}

# Parse arguments
for arg in "$@"; do
    case $arg in
        --help|-h)
            print_usage
            exit 0
            ;;
        --gpu)
            DEVICE="cuda"
            echo "Running DLRM on GPU"
            ;;
        --cpu)
            DEVICE="cpu"
            echo "Running DLRM on CPU"
            ;;
        --offload)
            USE_OFFLOAD=true
            echo "Running DLRM with GPU+CPU offloading (embeddings on CPU)"
            ;;
        --size=*)
            MODEL_SIZE="${arg#*=}"
            ;;
        --max_examples=*|--max-examples=*|--max-samples=*|--samples=*)
            MAX_EXAMPLES="${arg#*=}"
            ;;
        --batch-size=*|--batch_size=*)
            BATCH_SIZE="${arg#*=}"
            ;;
        --data=*)
            DATA_TYPE="${arg#*=}"
            ;;
        --force-download)
            FORCE_DOWNLOAD=true
            ;;
        --mlperf)
            MLPERF_MODE=true
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

# Apply MLPerf settings
if [[ "$MLPERF_MODE" == "true" ]]; then
    # Check for explicit synthetic
    EXPLICIT_SYNTHETIC=false
    for arg in "$@"; do
        if [[ "$arg" == "--data=synthetic" ]]; then
            EXPLICIT_SYNTHETIC=true
            break
        fi
    done
    
    if [[ "$EXPLICIT_SYNTHETIC" != "true" ]]; then
        DATA_TYPE="real"
    fi
fi

# Validate model size
if [[ "$MODEL_SIZE" != "small" && "$MODEL_SIZE" != "sample" && "$MODEL_SIZE" != "full" ]]; then
    echo -e "${RED}Invalid model size: $MODEL_SIZE${NC}"
    echo "Valid options: small, sample, full"
    exit 1
fi

# Validate data type
if [[ "$DATA_TYPE" != "synthetic" && "$DATA_TYPE" != "real" ]]; then
    echo -e "${RED}Invalid data type: $DATA_TYPE${NC}"
    echo "Valid options: synthetic, real"
    exit 1
fi

# If --data=real is requested, force full model size
if [[ "$DATA_TYPE" == "real" && "$MODEL_SIZE" == "small" ]]; then
    echo -e "${YELLOW}Note: --data=real requires full model. Setting --size=sample${NC}"
    MODEL_SIZE="sample"
fi

# Set directories
DATA_DIR="${PROJECT_DIR}/data/criteo"
MODEL_DIR="${PROJECT_DIR}/models/dlrm"

# ============================================================================
# Auto-download logic based on model size
# ============================================================================

check_existing_data() {
    local has_model=false
    local has_data=false
    local model_size_gb=0
    local data_size_mb=0
    
    # Check for model
    if [[ -d "${MODEL_DIR}/model_weights" ]]; then
        has_model=true
        model_size_gb=$(du -sg "${MODEL_DIR}/model_weights" 2>/dev/null | cut -f1 || echo "0")
    fi
    
    # Check for data
    if [[ -f "${DATA_DIR}/labels.npy" ]]; then
        has_data=true
        data_size_mb=$(du -sm "${DATA_DIR}" 2>/dev/null | cut -f1 || echo "0")
    fi
    
    echo "$has_model $has_data $model_size_gb $data_size_mb"
}

prompt_existing_data() {
    local has_model=$1
    local has_data=$2
    local model_size_gb=$3
    local data_size_mb=$4
    
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║              Existing DLRM Data Detected                   ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    if [[ "$has_model" == "true" ]]; then
        echo -e "  ${GREEN}✓${NC} Model weights found: ${model_size_gb}GB at ${MODEL_DIR}/model_weights"
    else
        echo -e "  ${RED}✗${NC} Model weights: Not found"
    fi
    
    if [[ "$has_data" == "true" ]]; then
        echo -e "  ${GREEN}✓${NC} Dataset found: ${data_size_mb}MB at ${DATA_DIR}"
    else
        echo -e "  ${RED}✗${NC} Dataset: Not found"
    fi
    
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  [S] Skip download - Use existing data (default)"
    echo "  [R] Re-download - Fresh download (overwrites existing)"
    echo "  [Q] Quit"
    echo ""
    
    read -p "Choose [S/r/q]: " -n 1 -r choice
    echo ""
    
    case "${choice,,}" in
        r)
            echo -e "${YELLOW}Will re-download...${NC}"
            return 1  # needs download
            ;;
        q)
            echo "Aborted."
            exit 0
            ;;
        *)
            echo -e "${GREEN}Using existing data...${NC}"
            return 0  # skip download
            ;;
    esac
}

needs_download() {
    # Check what exists
    read has_model has_data model_size_gb data_size_mb <<< $(check_existing_data)
    
    case $MODEL_SIZE in
        small)
            # For small, we just need synthetic data
            if [[ "$has_data" == "true" ]]; then
                if [[ "$FORCE_DOWNLOAD" == "true" ]]; then
                    return 0  # force download
                fi
                # Ask user what to do
                if prompt_existing_data "$has_model" "$has_data" "$model_size_gb" "$data_size_mb"; then
                    return 1  # skip download
                else
                    return 0  # re-download
                fi
            fi
            return 0  # needs download (no data)
            ;;
        sample|full)
            # For sample/full, need model weights
            if [[ "$has_model" == "true" && "$has_data" == "true" ]]; then
                if [[ "$FORCE_DOWNLOAD" == "true" ]]; then
                    return 0  # force download
                fi
                # Both exist, ask user
                if prompt_existing_data "$has_model" "$has_data" "$model_size_gb" "$data_size_mb"; then
                    return 1  # skip download
                else
                    return 0  # re-download
                fi
            elif [[ "$has_model" == "true" && "$has_data" == "false" ]]; then
                # Only model exists, just need data
                echo -e "${YELLOW}Model exists (${model_size_gb}GB), only generating dataset...${NC}"
                SKIP_MODEL_DOWNLOAD=true
                return 0
            fi
            return 0  # needs download
            ;;
    esac
    return 0  # default: needs download
}

download_small() {
    echo -e "${CYAN}Setting up DLRM small (debug) configuration...${NC}"
    
    mkdir -p "${DATA_DIR}" "${MODEL_DIR}"
    
    # Generate synthetic data
    echo -e "${YELLOW}Generating synthetic Criteo data...${NC}"
    python3 << 'PYTHON_EOF'
import numpy as np
import os

output_dir = os.environ.get('DATA_DIR', 'data/criteo')
os.makedirs(output_dir, exist_ok=True)

num_samples = 100000  # 100K samples
num_dense_features = 13
num_sparse_features = 26

print(f"Generating {num_samples:,} synthetic samples...")

labels = np.random.randint(0, 2, size=num_samples).astype(np.int32)
dense_features = np.random.randn(num_samples, num_dense_features).astype(np.float32)
sparse_features = np.random.randint(0, 1000, size=(num_samples, num_sparse_features)).astype(np.int32)

np.save(os.path.join(output_dir, "labels.npy"), labels)
np.save(os.path.join(output_dir, "dense_features.npy"), dense_features)
np.save(os.path.join(output_dir, "sparse_features.npy"), sparse_features)

# Save metadata
with open(os.path.join(output_dir, "metadata.txt"), 'w') as f:
    f.write(f"num_samples={num_samples}\n")
    f.write("type=synthetic_small\n")

print(f"✓ Data saved to {output_dir}")
PYTHON_EOF
    
    echo -e "${GREEN}✓ Small configuration ready${NC}"
}

download_sample() {
    echo -e "${CYAN}Setting up DLRM sample configuration...${NC}"
    
    mkdir -p "${DATA_DIR}" "${MODEL_DIR}"
    
    # Download model weights only if not skipped
    if [[ "$SKIP_MODEL_DOWNLOAD" != "true" ]]; then
        echo -e "${YELLOW}This includes the full 97GB model with synthetic data${NC}"
        echo ""
        
        # Check disk space (~100GB needed)
        AVAIL_GB=$(df -BG "${PROJECT_DIR}" | awk 'NR==2 {print $4}' | tr -d 'G')
        if [[ "$AVAIL_GB" -lt 110 ]]; then
            echo -e "${RED}Warning: Only ${AVAIL_GB}GB available, need ~110GB${NC}"
            read -p "Continue anyway? [y/N] " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Aborted. Use --size=small for quick testing."
                exit 1
            fi
        fi
        
        # Download model weights
        echo -e "${CYAN}Downloading model weights (~97GB)...${NC}"
        echo -e "${YELLOW}This will take a while depending on your connection...${NC}"
        
        cd "${MODEL_DIR}"
        bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
            https://inference.mlcommons-storage.org/metadata/dlrm-v2-model-weights.uri
    else
        echo -e "${GREEN}Using existing model weights...${NC}"
    fi
    
    # Generate synthetic data (larger than small)
    echo -e "${CYAN}Generating sample dataset (1M samples)...${NC}"
    export DATA_DIR="${DATA_DIR}"
    python3 << 'PYTHON_EOF'
import numpy as np
import os

output_dir = os.environ.get('DATA_DIR', 'data/criteo')
os.makedirs(output_dir, exist_ok=True)

num_samples = 1000000  # 1M samples
num_dense_features = 13
num_sparse_features = 26

print(f"Generating {num_samples:,} synthetic samples...")

# Use proper embedding sizes for compatibility
embedding_sizes = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63,
    40000000, 3067956, 405282, 10, 2209, 11938, 155, 4,
    976, 14, 40000000, 40000000, 40000000, 590152, 12973, 108
]

labels = np.random.randint(0, 2, size=num_samples).astype(np.int32)
dense_features = np.random.randn(num_samples, num_dense_features).astype(np.float32)

sparse_features = []
for size in embedding_sizes:
    indices = np.random.randint(0, min(size, 1000000), size=num_samples).astype(np.int32)
    sparse_features.append(indices)
sparse_features = np.stack(sparse_features, axis=1)

np.save(os.path.join(output_dir, "labels.npy"), labels)
np.save(os.path.join(output_dir, "dense_features.npy"), dense_features)
np.save(os.path.join(output_dir, "sparse_features.npy"), sparse_features)

with open(os.path.join(output_dir, "metadata.txt"), 'w') as f:
    f.write(f"num_samples={num_samples}\n")
    f.write("type=synthetic_sample\n")

print(f"✓ Data saved to {output_dir}")
PYTHON_EOF
    
    echo -e "${GREEN}✓ Sample configuration ready${NC}"
}

download_full() {
    echo -e "${CYAN}Setting up DLRM full MLPerf configuration...${NC}"
    echo -e "${YELLOW}This includes the full 97GB model AND 100GB Criteo dataset${NC}"
    echo ""
    
    mkdir -p "${DATA_DIR}" "${MODEL_DIR}"
    
    # Check disk space (~210GB needed)
    AVAIL_GB=$(df -BG "${PROJECT_DIR}" | awk 'NR==2 {print $4}' | tr -d 'G')
    if [[ "$AVAIL_GB" -lt 220 ]]; then
        echo -e "${RED}Warning: Only ${AVAIL_GB}GB available, need ~220GB${NC}"
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted. Use --size=sample for synthetic data."
            exit 1
        fi
    fi
    
    # Download model weights
    echo -e "${CYAN}Downloading model weights (~97GB)...${NC}"
    cd "${MODEL_DIR}"
    bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
        https://inference.mlcommons-storage.org/metadata/dlrm-v2-model-weights.uri
    
    # Download preprocessed dataset
    echo -e "${CYAN}Downloading preprocessed Criteo dataset (~100GB)...${NC}"
    cd "${DATA_DIR}"
    bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
        -d . https://inference.mlcommons-storage.org/metadata/dlrm-v2-preprocessed-dataset.uri
    
    echo -e "${GREEN}✓ Full MLPerf configuration ready${NC}"
}

download_real_criteo() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║        Downloading Real Criteo Terabyte Dataset            ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Criteo Terabyte Dataset (Official MLPerf dataset)${NC}"
    echo "  - Used for DLRM recommendation benchmark"
    echo "  - Day 23 is used for validation"
    echo "  - Preprocessed version available from MLCommons"
    echo ""
    
    mkdir -p "${DATA_DIR}" "${MODEL_DIR}"
    
    # Check disk space (~210GB needed)
    AVAIL_GB=$(df -BG "${PROJECT_DIR}" | awk 'NR==2 {print $4}' | tr -d 'G')
    if [[ "$AVAIL_GB" -lt 220 ]]; then
        echo -e "${RED}Warning: Only ${AVAIL_GB}GB available, need ~220GB${NC}"
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted. Use --data=synthetic instead."
            exit 1
        fi
    fi
    
    # Download model weights if not present
    if [[ ! -d "${MODEL_DIR}/model_weights" ]] || [[ "$SKIP_MODEL_DOWNLOAD" != "true" ]]; then
        echo -e "${CYAN}Downloading DLRM model weights (~97GB)...${NC}"
        cd "${MODEL_DIR}"
        bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
            https://inference.mlcommons-storage.org/metadata/dlrm-v2-model-weights.uri
    else
        echo -e "${GREEN}✓ Model weights already present${NC}"
    fi
    
    # Download preprocessed Criteo dataset
    echo -e "${CYAN}Downloading preprocessed Criteo Terabyte dataset (~100GB)...${NC}"
    echo "This is the official MLPerf preprocessed dataset."
    cd "${DATA_DIR}"
    bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
        -d . https://inference.mlcommons-storage.org/metadata/dlrm-v2-preprocessed-dataset.uri
    
    # Mark as real data
    echo "type=real_criteo" > "${DATA_DIR}/metadata.txt"
    echo "source=mlcommons_preprocessed" >> "${DATA_DIR}/metadata.txt"
    
    echo -e "${GREEN}✓ Real Criteo dataset ready${NC}"
}

# Check if download needed
if [[ "$FORCE_DOWNLOAD" == true ]] || needs_download; then
    echo -e "${YELLOW}DLRM data/model not found or incomplete${NC}"
    echo -e "Model size selected: ${GREEN}${MODEL_SIZE}${NC}"
    echo -e "Data type selected: ${GREEN}${DATA_TYPE}${NC}"
    echo ""
    
    if [[ "$DATA_TYPE" == "real" ]]; then
        echo "This will download:"
        echo "  - Model weights (~97GB)"
        echo "  - Preprocessed Criteo Terabyte dataset (~100GB)"
        echo -e "${YELLOW}Total: ~200GB, Estimated time: ~2 hours at 30MB/s${NC}"
    else
        case $MODEL_SIZE in
            small)
                echo "This will generate ~100K synthetic samples (~50MB)"
                ;;
            sample)
                echo "This will download ~97GB model + generate 1M synthetic samples"
                echo -e "${YELLOW}Estimated time: ~1 hour at 30MB/s${NC}"
                ;;
            full)
                echo "This will download ~97GB model + ~100GB Criteo dataset"
                echo -e "${YELLOW}Estimated time: ~2 hours at 30MB/s${NC}"
                ;;
        esac
    fi
    
    echo ""
    read -p "Proceed with download/setup? [Y/n] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    
    export DATA_DIR="${DATA_DIR}"
    
    # Handle data download based on --data option
    if [[ "$DATA_TYPE" == "real" ]]; then
        download_real_criteo
    else
        case $MODEL_SIZE in
            small)
                download_small
                ;;
            sample)
                download_sample
                ;;
            full)
                download_full
                ;;
        esac
    fi
    
    echo ""
fi

# Build command
CMD="python ${SCRIPT_DIR}/run_dlrm_benchmark.py"
CMD="$CMD --device $DEVICE"
CMD="$CMD --model-size $MODEL_SIZE"
CMD="$CMD --data-dir ${DATA_DIR}"
CMD="$CMD --data-type ${DATA_TYPE}"
CMD="$CMD --output-dir ${PROJECT_DIR}/results"

if [[ -n "$MAX_EXAMPLES" ]]; then
    CMD="$CMD --max-examples $MAX_EXAMPLES"
fi

if [[ -n "$BATCH_SIZE" ]]; then
    CMD="$CMD --batch-size $BATCH_SIZE"
fi

# Check for model weights (for sample/full)
if [[ -d "${MODEL_DIR}/model" ]]; then
    MODEL_PATH="${MODEL_DIR}/model"
    CMD="$CMD --model-path $MODEL_PATH"
fi

# Print configuration
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              DLRM Benchmark Configuration                  ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo -e "  Model Size: ${MODEL_SIZE}"
echo -e "  Device:     ${DEVICE}"
echo -e "  Offload:    ${USE_OFFLOAD}"
echo -e "  Data Type:  ${DATA_TYPE}"
echo -e "  Data Dir:   ${DATA_DIR}"
if [[ "$MLPERF_MODE" == "true" ]]; then
    echo -e "  MLPerf:     ${GREEN}ENABLED${NC}"
    
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
    echo -e "  MLPerf:     ${CYAN}disabled${NC} (use --mlperf for official settings)"
fi
if [[ "$MODEL_SIZE" != "small" ]]; then
    echo -e "  Model Dir:  ${MODEL_DIR}"
fi
echo ""

# Add offload flag if enabled
[ "$USE_OFFLOAD" = true ] && CMD="$CMD --offload"
[ "$MLPERF_MODE" = true ] && CMD="$CMD --mlperf"

# Run benchmark
cd "$PROJECT_DIR"
$CMD "${EXTRA_ARGS[@]}"
