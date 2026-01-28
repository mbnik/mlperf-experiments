#!/bin/bash
# =============================================================================
# MLPerf Benchmark Setup and Runner - Llama Model Family
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
MODEL="llama3-8b"
DEVICE="cuda"
USE_OFFLOAD=false
MAX_EXAMPLES=10
DATA_TYPE="synthetic"
QUANTIZATION="none"
MLPERF_MODE=false
MAX_NEW_TOKENS=128
DOWNLOAD_METHOD="hf"
EXTRA_ARGS=()

print_usage() {
    echo -e "${CYAN}Llama Model Family Benchmark Runner${NC}"
    echo ""
    echo "Usage: $0 [model] [options]"
    echo ""
    echo "Models:"
    echo "  llama3-8b       Llama 3.1 8B Instruct (default) - ~16GB GPU or offload"
    echo "  llama2-7b       Llama 2 7B Chat - ~14GB GPU or offload"
    echo "  llama2-13b      Llama 2 13B Chat - ~26GB GPU or offload"
    echo "  llama2-70b      Llama 2 70B Chat - ~140GB, requires offload or multi-GPU"
    echo "  llama3-70b      Llama 3.1 70B Instruct - ~140GB, requires offload or multi-GPU"
    echo ""
    echo "Device Options:"
    echo "  --gpu           Run on GPU (requires sufficient VRAM)"
    echo "  --cpu           Run on CPU only (slow)"
    echo "  --offload       GPU + CPU RAM offloading (for large models)"
    echo ""
    echo "Quantization Options:"
    echo "  --4bit          Use 4-bit quantization (reduces memory ~4x)"
    echo "  --8bit          Use 8-bit quantization (reduces memory ~2x)"
    echo ""
    echo "Data Options:"
    echo "  --data=TYPE     Data type: synthetic, real (default: synthetic)"
    echo "  --samples=N     Number of samples (default: 10)"
    echo "  --download=M    Download method: hf (default), wget"
    echo ""
    echo "MLPerf Compliance:"
    echo "  --mlperf        Use official MLPerf settings (max_new_tokens=1024, real data)"
    echo "                  Automatically downloads CNN-DailyMail dataset"
    echo ""
    echo "Memory Requirements (FP16):"
    echo "  llama2-7b:   ~14GB VRAM  |  4-bit: ~4GB"
    echo "  llama3-8b:   ~16GB VRAM  |  4-bit: ~5GB"
    echo "  llama2-13b:  ~26GB VRAM  |  4-bit: ~8GB"
    echo "  llama2-70b:  ~140GB VRAM |  4-bit: ~35GB"
    echo "  llama3-70b:  ~140GB VRAM |  4-bit: ~35GB"
    echo ""
    echo "Examples:"
    echo "  $0 llama3-8b --gpu                    # 8B on GPU (needs 16GB+)"
    echo "  $0 llama3-8b --gpu --4bit             # 8B quantized (needs 5GB+)"
    echo "  $0 llama3-8b --offload                # 8B with CPU offload (FP16)"
    echo "  $0 llama2-70b --4bit                  # 70B quantized (needs ~35GB)"
    echo "  $0 llama3-8b --gpu --data=real        # With CNN-DailyMail data"
    echo "  $0 llama3-8b --gpu --4bit --mlperf    # Official MLPerf settings"
    echo ""
    echo "NOTE: --4bit/--8bit and --offload cannot be combined!"
    echo "      bitsandbytes quantization does not support CPU offloading."
    echo ""
}

detect_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1)
        if [[ -n "$GPU_MEM" ]]; then
            echo -e "${GREEN}Detected GPU: ${GPU_NAME} (${GPU_MEM}MB VRAM)${NC}"
            if [[ "$GPU_COUNT" -gt 1 ]]; then
                echo -e "${GREEN}Multi-GPU: ${GPU_COUNT} GPUs available${NC}"
            fi
            return 0
        fi
    fi
    echo -e "${YELLOW}No NVIDIA GPU detected${NC}"
    return 1
}

get_model_info() {
    local model=$1
    case $model in
        llama2-7b)
            HF_MODEL="meta-llama/Llama-2-7b-chat-hf"
            MODEL_SIZE_GB=14
            ;;
        llama3-8b)
            HF_MODEL="meta-llama/Llama-3.1-8B-Instruct"
            MODEL_SIZE_GB=16
            ;;
        llama2-13b)
            HF_MODEL="meta-llama/Llama-2-13b-chat-hf"
            MODEL_SIZE_GB=26
            ;;
        llama2-70b)
            HF_MODEL="meta-llama/Llama-2-70b-chat-hf"
            MODEL_SIZE_GB=140
            ;;
        llama3-70b)
            HF_MODEL="meta-llama/Llama-3.1-70B-Instruct"
            MODEL_SIZE_GB=140
            ;;
        *)
            echo -e "${RED}Unknown model: $model${NC}"
            echo "Valid models: llama2-7b, llama3-8b, llama2-13b, llama2-70b, llama3-70b"
            exit 1
            ;;
    esac
}

check_hf_token() {
    # Check if already logged in using Python (more reliable than CLI)
    local login_status=$(python3 -c "
try:
    from huggingface_hub import whoami
    user = whoami()
    print(f'LOGGED_IN:{user[\"name\"]}')
except:
    print('NOT_LOGGED_IN')
" 2>/dev/null)
    
    if [[ "$login_status" == LOGGED_IN:* ]]; then
        local username="${login_status#LOGGED_IN:}"
        echo -e "${GREEN}✓ Logged in to HuggingFace as: $username${NC}"
        return 0
    fi
    
    # Check for token in environment
    if [[ -n "$HF_TOKEN" ]] || [[ -n "$HUGGING_FACE_HUB_TOKEN" ]]; then
        echo -e "${GREEN}✓ HuggingFace token found in environment${NC}"
        return 0
    fi
    
    # No token found - prompt user for login
    echo -e "${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  HuggingFace Authentication Required                       ║${NC}"
    echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Llama models require a HuggingFace account with access granted."
    echo ""
    echo -e "${CYAN}Before you begin, make sure you have:${NC}"
    echo "  1. A HuggingFace account (https://huggingface.co/join)"
    echo "  2. An access token (https://huggingface.co/settings/tokens)"
    echo "  3. Requested access to Llama (https://huggingface.co/meta-llama)"
    echo ""
    
    read -p "Would you like to login now? [Y/n] " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo ""
        echo -e "${CYAN}Enter your HuggingFace access token:${NC}"
        echo "(You can find/create one at: https://huggingface.co/settings/tokens)"
        echo ""
        read -r -s -p "Token (hidden): " HF_TOKEN_INPUT
        echo ""
        
        if [[ -n "$HF_TOKEN_INPUT" ]]; then
            # Try to login using the token
            echo ""
            echo "Logging in to HuggingFace..."
            
            # Use Python to login (more reliable than CLI)
            if python3 << PYTHON_EOF
import sys
try:
    from huggingface_hub import login, whoami
    token = "$HF_TOKEN_INPUT"
    login(token=token, add_to_git_credential=False)
    user = whoami()
    print(f"✓ Successfully logged in as: {user['name']}")
    sys.exit(0)
except Exception as e:
    print(f"✗ Login failed: {e}")
    sys.exit(1)
PYTHON_EOF
            then
                export HF_TOKEN="$HF_TOKEN_INPUT"
                export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_INPUT"
                echo -e "${GREEN}✓ Login successful!${NC}"
                echo ""
                
                # Offer to save token permanently
                read -p "Save token to ~/.bashrc for future sessions? [y/N] " -n 1 -r
                echo ""
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    echo "" >> ~/.bashrc
                    echo "# HuggingFace token (added by MLPerf setup)" >> ~/.bashrc
                    echo "export HF_TOKEN=\"$HF_TOKEN_INPUT\"" >> ~/.bashrc
                    echo -e "${GREEN}✓ Token saved to ~/.bashrc${NC}"
                fi
            else
                echo -e "${RED}✗ Login failed. Please check your token.${NC}"
                read -p "Continue anyway? [y/N] " -n 1 -r
                echo ""
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    exit 1
                fi
            fi
        else
            echo -e "${YELLOW}No token entered.${NC}"
            read -p "Continue anyway? [y/N] " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    else
        echo -e "${YELLOW}Skipping login. Download may fail without authentication.${NC}"
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

check_model_access() {
    local model_id="$1"
    
    echo -e "${CYAN}Checking model access for: $model_id${NC}"
    
    # Try to actually download a small file to verify access
    python3 << PYTHON_EOF
import sys
import os

try:
    from huggingface_hub import hf_hub_download, whoami
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    
    model_id = "$model_id"
    
    # First check if logged in
    try:
        user = whoami()
        print(f"Logged in as: {user['name']}")
    except Exception:
        print("⚠ Not logged in to HuggingFace")
        print("  The model download may fail without authentication.")
        sys.exit(1)
    
    # Try to download config.json (small file) to verify actual access
    try:
        path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            local_dir_use_symlinks=False,
        )
        print(f"✓ Access confirmed: {model_id}")
        sys.exit(0)
    except GatedRepoError as e:
        print(f"")
        print(f"╔{'═'*60}╗")
        print(f"║  ACCESS DENIED - Model Access Required{' '*20}║")
        print(f"╚{'═'*60}╝")
        print(f"")
        print(f"You need to request access to: {model_id}")
        print(f"")
        print(f"  ► Click here: https://huggingface.co/{model_id}")
        print(f"")
        print(f"  Steps:")
        print(f"  1. Click 'Expand to review and access'")
        print(f"  2. Accept the license agreement")
        print(f"  3. Wait for approval (usually instant)")
        print(f"  4. Re-run this script")
        print(f"")
        sys.exit(1)
    except RepositoryNotFoundError:
        print(f"✗ Model not found: {model_id}")
        sys.exit(1)
    except Exception as e:
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            print(f"")
            print(f"╔{'═'*60}╗")
            print(f"║  ACCESS DENIED{' '*45}║")
            print(f"╚{'═'*60}╝")
            print(f"")
            print(f"  ► Request access: https://huggingface.co/{model_id}")
            print(f"")
            sys.exit(1)
        else:
            print(f"⚠ Could not verify access: {e}")
            sys.exit(0)

except ImportError as e:
    print(f"⚠ Missing library: {e}")
    print("  Skipping access check")
    sys.exit(0)
PYTHON_EOF
    
    local result=$?
    if [[ $result -ne 0 ]]; then
        echo ""
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    echo ""
}

download_cnn_dailymail_wget() {
    local data_dir="${PROJECT_DIR}/data/cnn-dailymail"
    local parquet_dir="${data_dir}/parquet"
    
    echo -e "${CYAN}► Downloading CNN-DailyMail via wget...${NC}"
    mkdir -p "$parquet_dir"
    
    # CNN-DailyMail is a public dataset (no auth needed)
    local base_url="https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/main/3.0.0"
    
    # Download test parquet
    echo "  Downloading test-00000-of-00001.parquet..."
    wget -q --show-progress -O "${parquet_dir}/test-00000-of-00001.parquet" \
        "${base_url}/test-00000-of-00001.parquet"
    
    # Download validation parquet
    echo "  Downloading validation-00000-of-00001.parquet..."
    wget -q --show-progress -O "${parquet_dir}/validation-00000-of-00001.parquet" \
        "${base_url}/validation-00000-of-00001.parquet"
    
    # Extract to JSON using pyarrow
    echo "  Extracting data from parquet files..."
    python3 << PYTHON_EOF
import pyarrow.parquet as pq
import json
import os

data_dir = "$data_dir"
parquet_dir = "$parquet_dir"

# Process test set
test_table = pq.read_table(os.path.join(parquet_dir, "test-00000-of-00001.parquet"))
test_data = [{"article": row["article"], "highlights": row["highlights"]} 
             for row in test_table.to_pylist()]
with open(os.path.join(data_dir, "test.json"), "w") as f:
    json.dump(test_data, f)
print(f"  ✓ Saved {len(test_data)} test articles")

# Process validation set
val_table = pq.read_table(os.path.join(parquet_dir, "validation-00000-of-00001.parquet"))
val_data = [{"article": row["article"], "highlights": row["highlights"]} 
            for row in val_table.to_pylist()]
with open(os.path.join(data_dir, "validation.json"), "w") as f:
    json.dump(val_data, f)
print(f"  ✓ Saved {len(val_data)} validation articles")
PYTHON_EOF
    
    echo -e "${GREEN}✓ CNN-DailyMail downloaded via wget${NC}"
}

download_cnn_dailymail_hf() {
    local data_dir="${PROJECT_DIR}/data/cnn-dailymail"
    
    echo -e "${CYAN}Downloading CNN-DailyMail via HuggingFace...${NC}"
    mkdir -p "$data_dir"
    
    python3 << PYTHON_EOF
from datasets import load_dataset
import json
import os

data_dir = "$data_dir"
print("Loading CNN-DailyMail from HuggingFace...")
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Save test set
test_data = [{"article": item["article"], "highlights": item["highlights"]} 
             for item in dataset["test"]]
with open(os.path.join(data_dir, "test.json"), "w") as f:
    json.dump(test_data, f)
print(f"✓ Saved {len(test_data)} test articles")

# Save validation set
val_data = [{"article": item["article"], "highlights": item["highlights"]} 
            for item in dataset["validation"]]
with open(os.path.join(data_dir, "validation.json"), "w") as f:
    json.dump(val_data, f)
print(f"✓ Saved {len(val_data)} validation articles")
PYTHON_EOF
    
    echo -e "${GREEN}✓ CNN-DailyMail ready${NC}"
}

download_cnn_dailymail() {
    local data_dir="${PROJECT_DIR}/data/cnn-dailymail"
    
    if [[ -f "$data_dir/test.json" ]]; then
        echo -e "${GREEN}✓ CNN-DailyMail data found${NC}"
        return 0
    fi
    
    if [[ "$DOWNLOAD_METHOD" == "wget" ]]; then
        download_cnn_dailymail_wget
    else
        download_cnn_dailymail_hf
    fi
}

download_model_wget() {
    local model_id="$1"
    local model_dir="${PROJECT_DIR}/models/llama/${MODEL}"
    
    # Check if model already downloaded
    if [[ -f "$model_dir/config.json" ]]; then
        echo -e "${GREEN}✓ Model files found at $model_dir${NC}"
        MODEL_PATH="$model_dir"
        return 0
    fi
    
    echo -e "${CYAN}► Downloading model via wget...${NC}"
    echo "  Model: $model_id"
    echo "  Target: $model_dir"
    echo ""
    
    # Prompt for HuggingFace token
    local url="https://huggingface.co/${model_id}"
    local url_len=${#url}
    local padding=$((58 - url_len))
    local spaces=$(printf '%*s' "$padding" '')
    
    echo -e "${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  HuggingFace Token Required for Model Download             ║${NC}"
    echo -e "${YELLOW}╠════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${YELLOW}║  Llama models require authentication. Get your token at:   ║${NC}"
    echo -e "${YELLOW}║  https://huggingface.co/settings/tokens                    ║${NC}"
    echo -e "${YELLOW}║                                                            ║${NC}"
    echo -e "${YELLOW}║  You must also accept the license agreement at:            ║${NC}"
    echo -e "${YELLOW}║  ${url}${spaces}║${NC}"
    echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -n "Enter your HuggingFace token (hf_...): "
    read -r -s HF_TOKEN
    echo ""
    
    if [[ -z "$HF_TOKEN" ]]; then
        echo -e "${RED}Error: No token provided${NC}"
        exit 1
    fi
    
    mkdir -p "$model_dir"
    
    local base_url="https://huggingface.co/${model_id}/resolve/main"
    
    # Get the list of files from the model repo
    echo "  Fetching model file list..."
    
    # Download model index to determine shard count
    wget -q --header="Authorization: Bearer $HF_TOKEN" \
        -O "$model_dir/model.safetensors.index.json" \
        "${base_url}/model.safetensors.index.json" 2>/dev/null || true
    
    # Parse shard count from index file
    local shard_count=0
    if [[ -f "$model_dir/model.safetensors.index.json" ]]; then
        shard_count=$(python3 -c "
import json
with open('$model_dir/model.safetensors.index.json') as f:
    data = json.load(f)
    files = set(data.get('weight_map', {}).values())
    print(len(files))
" 2>/dev/null || echo "0")
    fi
    
    if [[ "$shard_count" -eq 0 ]]; then
        echo -e "${RED}Error: Could not determine model file count. Check your token and model access.${NC}"
        rm -rf "$model_dir"
        exit 1
    fi
    
    echo "  Model has $shard_count weight shards"
    echo ""
    
    # Download config files
    local config_files=(
        "config.json"
        "generation_config.json"
        "tokenizer.json"
        "tokenizer_config.json"
        "special_tokens_map.json"
    )
    
    echo "  Downloading configuration files..."
    for file in "${config_files[@]}"; do
        echo -n "    $file... "
        if wget -q --header="Authorization: Bearer $HF_TOKEN" \
            -O "$model_dir/$file" \
            "${base_url}/$file" 2>/dev/null; then
            echo "✓"
        else
            echo "skipped"
        fi
    done
    
    # Download model weight shards
    echo ""
    echo "  Downloading model weights (~${MODEL_SIZE_GB}GB total)..."
    local padded_total=$(printf "%05d" $shard_count)
    
    for ((i=1; i<=shard_count; i++)); do
        local padded_i=$(printf "%05d" $i)
        local shard_file="model-${padded_i}-of-${padded_total}.safetensors"
        echo "    Downloading shard $i/$shard_count: $shard_file"
        
        if ! wget -q --show-progress --header="Authorization: Bearer $HF_TOKEN" \
            -O "$model_dir/$shard_file" \
            "${base_url}/$shard_file"; then
            echo -e "${RED}Error: Failed to download $shard_file${NC}"
            echo "  Check your token and ensure you have access to $model_id"
            rm -rf "$model_dir"
            exit 1
        fi
    done
    
    echo ""
    echo -e "${GREEN}✓ Model downloaded to $model_dir${NC}"
    MODEL_PATH="$model_dir"
}

# Parse arguments - first arg might be model name
if [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; then
    MODEL="$1"
    shift
fi

# Check for explicit --data=synthetic BEFORE parsing (to preserve the info)
EXPLICIT_SYNTHETIC=false
for arg in "$@"; do
    if [[ "$arg" == "--data=synthetic" ]]; then
        EXPLICIT_SYNTHETIC=true
        break
    fi
done

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            print_usage
            exit 0
            ;;
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
        --download=*)
            DOWNLOAD_METHOD="${1#*=}"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Get model info
get_model_info "$MODEL"

# Apply MLPerf settings if --mlperf flag is set
if [[ "$MLPERF_MODE" == "true" ]]; then
    MAX_NEW_TOKENS=1024
    
    # If user did NOT explicitly set --data=synthetic, default to real data
    if [[ "$EXPLICIT_SYNTHETIC" != "true" ]]; then
        DATA_TYPE="real"
    fi
fi

# Print header
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              Llama Benchmark Runner                        ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Detect GPU
detect_gpu
GPU_AVAILABLE=$?

# Calculate memory requirements
if [[ "$QUANTIZATION" == "4bit" ]]; then
    REQUIRED_MEM=$((MODEL_SIZE_GB / 4))
    QUANT_STR="4-bit quantized"
elif [[ "$QUANTIZATION" == "8bit" ]]; then
    REQUIRED_MEM=$((MODEL_SIZE_GB / 2))
    QUANT_STR="8-bit quantized"
else
    REQUIRED_MEM=$MODEL_SIZE_GB
    QUANT_STR="FP16"
fi

echo ""
echo -e "Model:        ${GREEN}$MODEL${NC} ($HF_MODEL)"
echo -e "Precision:    ${QUANT_STR}"
echo -e "Required:     ~${REQUIRED_MEM}GB VRAM"
echo -e "Device:       ${DEVICE}"
echo -e "Data:         ${DATA_TYPE}"
echo -e "Samples:      ${MAX_EXAMPLES}"

# MLPerf status
if [[ "$MLPERF_MODE" == "true" ]]; then
    echo -e "MLPerf Mode:  ${GREEN}ENABLED${NC} (max_new_tokens=${MAX_NEW_TOKENS})"
    
    # Show warning if using synthetic data
    if [[ "$DATA_TYPE" == "synthetic" ]]; then
        echo ""
        echo -e "${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${YELLOW}║  ⚠️  SYNTHETIC DATA WITH MLPerf MODE                       ║${NC}"
        echo -e "${YELLOW}╠════════════════════════════════════════════════════════════╣${NC}"
        echo -e "${YELLOW}║  Results are NOT comparable to official MLPerf benchmarks  ║${NC}"
        echo -e "${YELLOW}║  For official comparison, use: --mlperf --data=real        ║${NC}"
        echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"
    fi
else
    echo -e "MLPerf Mode:  ${CYAN}disabled${NC} (use --mlperf for official settings)"
fi
echo ""

# Check HuggingFace token (only if using hf download method)
if [[ "$DOWNLOAD_METHOD" != "wget" ]]; then
    check_hf_token
    check_model_access "$HF_MODEL"
fi

# Download model via wget if requested
MODEL_PATH="$HF_MODEL"
if [[ "$DOWNLOAD_METHOD" == "wget" ]]; then
    download_model_wget "$HF_MODEL"
fi

# Memory warning
if [[ "$GPU_AVAILABLE" -eq 0 ]] && [[ "$DEVICE" == "cuda" ]]; then
    if [[ "$GPU_MEM" -lt $((REQUIRED_MEM * 1024)) ]]; then
        echo -e "${YELLOW}+------------------------------------------------------------+${NC}"
        echo -e "${YELLOW}|  WARNING: GPU memory may be insufficient                   |${NC}"
        echo -e "${YELLOW}+------------------------------------------------------------+${NC}"
        echo ""
        echo -e "  GPU Memory:     ${GPU_MEM}MB"
        echo -e "  Required:       ~${REQUIRED_MEM}GB (${REQUIRED_MEM}000MB)"
        echo ""
        echo "Options:"
        echo "  1. Use --offload (CPU memory offloading)"
        echo "  2. Use --4bit (4-bit quantization)"
        echo "  3. Use --offload + --4bit (both)"
        echo "  4. Continue anyway"
        echo "  5. Stop"
        echo ""
        read -p "Select option [1-5]: " -n 1 -r
        echo ""
        case $REPLY in
            1)
                echo -e "${GREEN}> Enabling --offload${NC}"
                USE_OFFLOAD=true
                ;;
            2)
                echo -e "${GREEN}> Enabling --4bit${NC}"
                QUANTIZATION="4bit"
                ;;
            3)
                echo -e "${YELLOW}> Cannot use --offload + --4bit together (bitsandbytes limitation)${NC}"
                echo "  Please choose option 1 or 2 instead."
                exit 1
                ;;
            4)
                echo -e "${YELLOW}> Continuing without changes (may fail with OOM)${NC}"
                ;;
            5|*)
                echo "Exiting."
                exit 0
                ;;
        esac
    fi
fi

# Download data if needed
if [[ "$DATA_TYPE" == "real" ]]; then
    download_cnn_dailymail
    DATA_DIR="${PROJECT_DIR}/data/cnn-dailymail"
fi

# Create results directory
RESULTS_DIR="${PROJECT_DIR}/results/llama"
mkdir -p "$RESULTS_DIR"

# Check for incompatible combination: quantization + offload
if [[ "$USE_OFFLOAD" == "true" && "$QUANTIZATION" != "none" ]]; then
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ⚠️  INCOMPATIBLE OPTIONS: --${QUANTIZATION} + --offload   ║${NC}"
    echo -e "${RED}╠════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${RED}║  bitsandbytes quantization does NOT support CPU offloading ║${NC}"
    echo -e "${RED}║                                                            ║${NC}"
    echo -e "${RED}║  Choose ONE of these options:                              ║${NC}"
    echo -e "${RED}║    --4bit     : 4-bit quantization (lowest VRAM)           ║${NC}"
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
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              Running Llama Benchmark                       ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

CMD="python3 ${SCRIPT_DIR}/run_llama_benchmark.py"
CMD="$CMD --model-name $MODEL_PATH"
CMD="$CMD --device $DEVICE"
CMD="$CMD --quantization $QUANTIZATION"
CMD="$CMD --data-type $DATA_TYPE"
CMD="$CMD --data-dir ${PROJECT_DIR}/data/cnn-dailymail"
CMD="$CMD --max-examples $MAX_EXAMPLES"
CMD="$CMD --max-new-tokens $MAX_NEW_TOKENS"
CMD="$CMD --output-dir $RESULTS_DIR"

[ "$USE_OFFLOAD" = true ] && CMD="$CMD --offload"
[ "$MLPERF_MODE" = true ] && CMD="$CMD --mlperf"

# Run with OOM error handling
set +e
$CMD "${EXTRA_ARGS[@]}"
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -ne 0 ]; then
    # Check if it was an OOM error
    if [ $EXIT_CODE -eq 1 ] && [ "$USE_OFFLOAD" = false ] && [ "$DEVICE" = "cuda" ]; then
        echo ""
        echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║  ⚠️  GPU OUT OF MEMORY ERROR                               ║${NC}"
        echo -e "${RED}╠════════════════════════════════════════════════════════════╣${NC}"
        echo -e "${RED}║  Llama models require significant VRAM:                    ║${NC}"
        echo -e "${RED}║    - Llama 3.1 8B: ~16GB    - Llama 2 7B: ~14GB            ║${NC}"
        echo -e "${RED}║    - Llama 2 13B: ~26GB    - Llama 2 70B: ~140GB           ║${NC}"
        echo -e "${RED}║                                                            ║${NC}"
        echo -e "${RED}║  Solutions:                                                ║${NC}"
        echo -e "${RED}║  1. Use --offload to enable CPU offloading                 ║${NC}"
        echo -e "${RED}║     Example: $0 --offload --mlperf                         ║${NC}"
        echo -e "${RED}║                                                            ║${NC}"
        echo -e "${RED}║  2. Use --quantization=4bit (requires less VRAM)           ║${NC}"
        echo -e "${RED}║     Example: $0 --quantization=4bit                        ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
    fi
    exit $EXIT_CODE
fi
