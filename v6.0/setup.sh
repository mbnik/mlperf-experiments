#!/bin/bash
# ============================================================================
# MLPerf Benchmark Setup and Runner - v6.0 Setup Script
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
VERSION="6.0"
GIT_TAG="v6.0.0pre"
INFERENCE_DIR="${SCRIPT_DIR}/inference"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_banner() {
    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║          MLPerf Inference v${VERSION} Setup Script              ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Setup Options:"
    echo "  --clone         Clone/update MLCommons inference repository"
    echo "  --loadgen       Install LoadGen library"
    echo "  --deps          Install Python dependencies"
    echo "  --all           Do all of the above"
    echo "  --status        Show current setup status"
    echo ""
    echo "Clean Options:"
    echo "  --clean              Clean ALL (inference + data + models + results)"
    echo "  --clean-inference    Remove only inference/ folder"
    echo "  --clean-data         Remove only data/ folder"
    echo "  --clean-models       Remove only models/ folder"
    echo "  --clean-results      Remove only results/ folder"
    echo "  --clean-cache        Remove data/ + models/ (large files)"
    echo ""
    echo "  -h, --help      Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --all                    # Full setup"
    echo "  $0 --clone                  # Clone inference repo only"
    echo "  $0 --clone --loadgen        # Clone repo and install LoadGen"
    echo "  $0 --status                 # Check current status"
    echo "  $0 --clean                  # Clean everything (with confirmation)"
    echo ""
    echo "Supported Benchmarks (v${VERSION}):"
    echo "  BERT, ResNet50, 3D-UNet, DLRM-v3,"
    echo "  Llama, Mixtral-8x7B, SDXL, Whisper"
}

check_status() {
    echo -e "${CYAN}MLPerf v${VERSION} Setup Status${NC}"
    echo "================================"
    echo ""
    
    # Check inference repo
    if [ -d "${INFERENCE_DIR}" ]; then
        cd "${INFERENCE_DIR}"
        local git_version=$(git describe --tags 2>/dev/null || echo "unknown")
        echo -e "  Inference Repo:  ${GREEN}✓ Present${NC} (${git_version})"
    else
        echo -e "  Inference Repo:  ${RED}✗ Not cloned${NC}"
    fi
    
    # Check LoadGen
    if python3 -c "import mlcommons_loadgen" 2>/dev/null; then
        echo -e "  LoadGen:         ${GREEN}✓ Installed${NC}"
    else
        echo -e "  LoadGen:         ${YELLOW}✗ Not installed${NC}"
    fi
    
    # Check directories
    echo ""
    echo "  Directories:"
    [ -d "${SCRIPT_DIR}/data" ] && echo -e "    data/:    ${GREEN}✓${NC}" || echo -e "    data/:    ${YELLOW}✗${NC}"
    [ -d "${SCRIPT_DIR}/models" ] && echo -e "    models/:  ${GREEN}✓${NC}" || echo -e "    models/:  ${YELLOW}✗${NC}"
    [ -d "${SCRIPT_DIR}/results" ] && echo -e "    results/: ${GREEN}✓${NC}" || echo -e "    results/: ${YELLOW}✗${NC}"
    [ -d "${SCRIPT_DIR}/scripts" ] && echo -e "    scripts/: ${GREEN}✓${NC}" || echo -e "    scripts/: ${YELLOW}✗${NC}"
    
    # Check scripts (v6.0 benchmarks - no RetinaNet, no GPT-J)
    echo ""
    echo "  Benchmark Scripts:"
    local scripts=(bert resnet50 3dunet dlrm llama mixtral sdxl whisper)
    for s in "${scripts[@]}"; do
        if [ -f "${SCRIPT_DIR}/scripts/run_${s}.sh" ]; then
            echo -e "    ${s}: ${GREEN}✓${NC}"
        else
            echo -e "    ${s}: ${RED}✗${NC}"
        fi
    done
    
    echo ""
}

clone_inference() {
    echo -e "${BLUE}► Cloning MLCommons Inference Repository${NC}"
    echo "  Tag: ${GIT_TAG}"
    echo ""
    
    if [ -d "${INFERENCE_DIR}" ]; then
        echo -e "${YELLOW}  Repository already exists.${NC}"
        
        cd "${INFERENCE_DIR}"
        local current_tag=$(git describe --tags 2>/dev/null || echo "unknown")
        
        if [ "$current_tag" == "$GIT_TAG" ]; then
            echo -e "${GREEN}  ✓ Already at correct version (${GIT_TAG})${NC}"
            return 0
        else
            echo -e "${YELLOW}  Current version: ${current_tag}${NC}"
            echo -e "${YELLOW}  Expected version: ${GIT_TAG}${NC}"
            echo ""
            read -p "  Update to ${GIT_TAG}? [y/N] " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "  Fetching tags..."
                git fetch --tags
                git checkout "${GIT_TAG}"
                echo -e "${GREEN}  ✓ Updated to ${GIT_TAG}${NC}"
            else
                echo -e "${YELLOW}  Skipped update${NC}"
            fi
        fi
    else
        echo "  Cloning from GitHub..."
        git clone --depth 1 --branch "${GIT_TAG}" \
            https://github.com/mlcommons/inference.git "${INFERENCE_DIR}"
        echo -e "${GREEN}  ✓ Cloned successfully${NC}"
    fi
}

install_loadgen() {
    echo -e "${BLUE}► Installing LoadGen${NC}"
    echo ""
    
    if ! [ -d "${INFERENCE_DIR}" ]; then
        echo -e "${RED}  ✗ Inference repository not found. Run --clone first.${NC}"
        return 1
    fi
    
    cd "${INFERENCE_DIR}/loadgen"
    
    echo "  Building and installing..."
    pip install . --quiet
    
    # Verify
    if python3 -c "import mlcommons_loadgen" 2>/dev/null; then
        echo -e "${GREEN}  ✓ LoadGen installed and verified${NC}"
    else
        echo -e "${RED}  ✗ LoadGen installation failed${NC}"
        return 1
    fi
}

install_deps() {
    echo -e "${BLUE}► Installing Python Dependencies${NC}"
    echo ""
    
    pip install --quiet \
        torch \
        torchvision \
        torchaudio \
        transformers \
        accelerate \
        bitsandbytes \
        datasets \
        diffusers \
        numpy \
        pillow \
        tqdm \
        requests \
        soundfile \
        librosa
    
    echo -e "${GREEN}  ✓ Dependencies installed${NC}"
}

setup_directories() {
    echo -e "${BLUE}► Setting up directories${NC}"
    
    mkdir -p "${SCRIPT_DIR}/data"
    mkdir -p "${SCRIPT_DIR}/models"
    mkdir -p "${SCRIPT_DIR}/results"
    
    echo -e "${GREEN}  ✓ Directories created${NC}"
}

# =============================================================================
# Clean Functions
# =============================================================================

get_folder_size() {
    local folder="$1"
    if [ -d "$folder" ]; then
        du -sh "$folder" 2>/dev/null | cut -f1
    else
        echo "0"
    fi
}

confirm_delete() {
    local folder="$1"
    local size="$2"
    
    echo -e "${YELLOW}  Warning: This will permanently delete:${NC}"
    echo -e "    ${folder} (${size})"
    echo ""
    read -p "  Are you sure? [y/N] " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

clean_folder() {
    local folder="$1"
    local name="$2"
    
    if [ ! -d "$folder" ]; then
        echo -e "${YELLOW}  ${name} folder does not exist, nothing to clean${NC}"
        return 0
    fi
    
    local size=$(get_folder_size "$folder")
    
    echo -e "${BLUE}► Cleaning ${name}${NC}"
    
    if confirm_delete "$folder" "$size"; then
        rm -rf "$folder"
        echo -e "${GREEN}  ✓ ${name} deleted${NC}"
    else
        echo -e "${YELLOW}  Skipped${NC}"
    fi
}

clean_inference() {
    clean_folder "${SCRIPT_DIR}/inference" "inference"
}

clean_data() {
    clean_folder "${SCRIPT_DIR}/data" "data"
}

clean_models() {
    clean_folder "${SCRIPT_DIR}/models" "models"
}

clean_results() {
    clean_folder "${SCRIPT_DIR}/results" "results"
}

clean_cache() {
    echo -e "${BLUE}► Cleaning cache (data + models)${NC}"
    
    local data_size=$(get_folder_size "${SCRIPT_DIR}/data")
    local models_size=$(get_folder_size "${SCRIPT_DIR}/models")
    
    local folders_to_delete=""
    [ -d "${SCRIPT_DIR}/data" ] && folders_to_delete="${folders_to_delete}    data/ (${data_size})\n"
    [ -d "${SCRIPT_DIR}/models" ] && folders_to_delete="${folders_to_delete}    models/ (${models_size})\n"
    
    if [ -z "$folders_to_delete" ]; then
        echo -e "${YELLOW}  No cache folders to clean${NC}"
        return 0
    fi
    
    echo -e "${YELLOW}  Warning: This will permanently delete:${NC}"
    echo -e "$folders_to_delete"
    read -p "  Are you sure? [y/N] " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        [ -d "${SCRIPT_DIR}/data" ] && rm -rf "${SCRIPT_DIR}/data"
        [ -d "${SCRIPT_DIR}/models" ] && rm -rf "${SCRIPT_DIR}/models"
        echo -e "${GREEN}  ✓ Cache cleaned${NC}"
    else
        echo -e "${YELLOW}  Skipped${NC}"
    fi
}

clean_all() {
    echo -e "${BLUE}► Cleaning ALL${NC}"
    
    local inference_size=$(get_folder_size "${SCRIPT_DIR}/inference")
    local data_size=$(get_folder_size "${SCRIPT_DIR}/data")
    local models_size=$(get_folder_size "${SCRIPT_DIR}/models")
    local results_size=$(get_folder_size "${SCRIPT_DIR}/results")
    
    local folders_to_delete=""
    [ -d "${SCRIPT_DIR}/inference" ] && folders_to_delete="${folders_to_delete}    inference/ (${inference_size})\n"
    [ -d "${SCRIPT_DIR}/data" ] && folders_to_delete="${folders_to_delete}    data/ (${data_size})\n"
    [ -d "${SCRIPT_DIR}/models" ] && folders_to_delete="${folders_to_delete}    models/ (${models_size})\n"
    [ -d "${SCRIPT_DIR}/results" ] && folders_to_delete="${folders_to_delete}    results/ (${results_size})\n"
    
    if [ -z "$folders_to_delete" ]; then
        echo -e "${YELLOW}  No folders to clean${NC}"
        return 0
    fi
    
    echo -e "${YELLOW}  Warning: This will permanently delete:${NC}"
    echo -e "$folders_to_delete"
    read -p "  Are you sure? [y/N] " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        [ -d "${SCRIPT_DIR}/inference" ] && rm -rf "${SCRIPT_DIR}/inference"
        [ -d "${SCRIPT_DIR}/data" ] && rm -rf "${SCRIPT_DIR}/data"
        [ -d "${SCRIPT_DIR}/models" ] && rm -rf "${SCRIPT_DIR}/models"
        [ -d "${SCRIPT_DIR}/results" ] && rm -rf "${SCRIPT_DIR}/results"
        echo -e "${GREEN}  ✓ All folders cleaned${NC}"
    else
        echo -e "${YELLOW}  Skipped${NC}"
    fi
}

# =============================================================================
# Main
# =============================================================================

print_banner

if [ $# -eq 0 ]; then
    print_usage
    exit 0
fi

DO_CLONE=false
DO_LOADGEN=false
DO_DEPS=false
DO_STATUS=false
DO_CLEAN=false
DO_CLEAN_INFERENCE=false
DO_CLEAN_DATA=false
DO_CLEAN_MODELS=false
DO_CLEAN_RESULTS=false
DO_CLEAN_CACHE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clone)
            DO_CLONE=true
            shift
            ;;
        --loadgen)
            DO_LOADGEN=true
            shift
            ;;
        --deps)
            DO_DEPS=true
            shift
            ;;
        --all)
            DO_CLONE=true
            DO_LOADGEN=true
            DO_DEPS=true
            shift
            ;;
        --status)
            DO_STATUS=true
            shift
            ;;
        --clean)
            DO_CLEAN=true
            shift
            ;;
        --clean-inference)
            DO_CLEAN_INFERENCE=true
            shift
            ;;
        --clean-data)
            DO_CLEAN_DATA=true
            shift
            ;;
        --clean-models)
            DO_CLEAN_MODELS=true
            shift
            ;;
        --clean-results)
            DO_CLEAN_RESULTS=true
            shift
            ;;
        --clean-cache)
            DO_CLEAN_CACHE=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo ""
            print_usage
            exit 1
            ;;
    esac
done

# Handle status
if [ "$DO_STATUS" = true ]; then
    check_status
    exit 0
fi

# Handle clean operations
if [ "$DO_CLEAN" = true ]; then
    clean_all
    exit 0
fi

if [ "$DO_CLEAN_INFERENCE" = true ]; then
    clean_inference
    exit 0
fi

if [ "$DO_CLEAN_DATA" = true ]; then
    clean_data
    exit 0
fi

if [ "$DO_CLEAN_MODELS" = true ]; then
    clean_models
    exit 0
fi

if [ "$DO_CLEAN_RESULTS" = true ]; then
    clean_results
    exit 0
fi

if [ "$DO_CLEAN_CACHE" = true ]; then
    clean_cache
    exit 0
fi

# Handle setup operations
if [ "$DO_CLONE" = false ] && [ "$DO_LOADGEN" = false ] && [ "$DO_DEPS" = false ]; then
    echo -e "${YELLOW}No setup option specified. Use --help for usage.${NC}"
    exit 0
fi

echo "Setting up MLPerf Inference v${VERSION}..."
echo ""

setup_directories

if [ "$DO_CLONE" = true ]; then
    clone_inference
    echo ""
fi

if [ "$DO_LOADGEN" = true ]; then
    install_loadgen
    echo ""
fi

if [ "$DO_DEPS" = true ]; then
    install_deps
    echo ""
fi

echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Next steps:"
echo "  1. Run a benchmark:  ./scripts/run_benchmark.sh bert --gpu"
echo "  2. Check status:     ./setup.sh --status"
echo ""
