#!/bin/bash
# ============================================================================
# MLPerf Benchmark Setup and Runner - 3D-UNet Medical Image Segmentation
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
MODEL_DIR="$PROJECT_DIR/models/3dunet"
DATA_DIR="$PROJECT_DIR/data/kits19"

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
MODEL_SIZE="sample"
BATCH_SIZE=""
MAX_EXAMPLES=""
SKIP_DOWNLOAD=false
DATA_TYPE="synthetic"  # synthetic or real
NUM_CASES=20  # Number of KiTS19 cases to download (max 210)
MLPERF_MODE=false

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --gpu           Run on GPU only"
    echo "  --cpu           Run on CPU only"
    echo "  --offload       Enable GPU+CPU offloading for limited VRAM"
    echo "  --size=SIZE     Config size: small, sample, full (default: sample)"
    echo "  --data=TYPE     Data type: synthetic, real (default: synthetic)"
    echo "  --cases=N       Number of KiTS19 cases to download (default: 20, max: 210)"
    echo "  --batch=N       Batch size (default: auto)"
    echo "  --samples=N     Number of samples to process"
    echo "  --quick         Quick test with minimal samples"
    echo "  -h, --help      Show this help"
    echo ""
    echo "Data Options:"
    echo "  synthetic  - Generate random 3D volumes (fast, no download)"
    echo "  real       - Download KiTS19 dataset (each case ~150-500MB)"
    echo ""
    echo "Examples:"
    echo "  $0 --gpu --data=real --cases=50    # Download 50 real cases"
    echo "  $0 --gpu --data=real --cases=210   # Download all training cases"
    echo "  $0 --gpu --data=synthetic           # Use synthetic data (fast)"
    echo "  $0 --gpu --mlperf                   # Official MLPerf settings"
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
        --size=*)
            MODEL_SIZE="${1#*=}"
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
        --samples=*)
            MAX_EXAMPLES="${1#*=}"
            shift
            ;;
        --cases=*)
            NUM_CASES="${1#*=}"
            # Validate range
            if [ "$NUM_CASES" -gt 210 ]; then
                echo -e "${YELLOW}Warning: Max cases is 210, using 210${NC}"
                NUM_CASES=210
            fi
            shift
            ;;
        --quick)
            MAX_EXAMPLES="10"
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
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check for explicit synthetic before applying MLPerf
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

# Print run configuration
echo -e "${CYAN}Running 3D-UNet Medical Image Segmentation Benchmark${NC}"
case $DEVICE in
    cuda)
        echo -e "  Mode: ${GREEN}GPU${NC}"
        ;;
    cpu)
        echo -e "  Mode: ${YELLOW}CPU${NC}"
        ;;
    auto)
        echo -e "  Mode: ${BLUE}Auto (GPU preferred)${NC}"
        ;;
esac
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
    local model_exists=false
    local data_exists=false
    local model_size_str=""
    local data_size_str=""
    local data_type_found=""
    
    if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
        local model_size=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
        model_exists=true
        model_size_str="$model_size"
    fi
    
    # Check for synthetic data
    if [ -f "$DATA_DIR/volumes.npy" ]; then
        local data_size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
        data_exists=true
        data_size_str="$data_size"
        data_type_found="synthetic"
    fi
    
    # Check for real KiTS19 data
    if [ -d "$DATA_DIR/case_00000" ] || [ -f "$DATA_DIR/preprocessed/volumes.npy" ]; then
        local data_size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
        data_exists=true
        data_size_str="$data_size"
        data_type_found="real (KiTS19)"
    fi
    
    if $model_exists || $data_exists; then
        echo ""
        echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${CYAN}║              Existing 3D-UNet Data Detected                ║${NC}"
        echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        
        if $model_exists; then
            echo -e "  ${GREEN}✓${NC} Model found: ${model_size_str} at $MODEL_DIR"
        else
            echo -e "  ${YELLOW}✗${NC} Model not found"
        fi
        
        if $data_exists; then
            echo -e "  ${GREEN}✓${NC} Dataset found: ${data_size_str} (${data_type_found}) at $DATA_DIR"
        else
            echo -e "  ${YELLOW}✗${NC} Dataset not found"
        fi
        
        echo ""
        echo "Options:"
        echo "  [S] Skip download - Use existing data (default)"
        echo "  [R] Re-download/regenerate data"
        echo "  [Q] Quit"
        echo ""
        
        read -t 30 -n 1 -p "" choice || choice="s"
        echo ""
        
        case ${choice,,} in
            r)
                echo -e "${YELLOW}Re-downloading/regenerating...${NC}"
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
# Setup Functions
# ============================================================================
setup_model() {
    echo -e "${CYAN}Setting up 3D-UNet model...${NC}"
    mkdir -p "$MODEL_DIR"
    echo -e "${GREEN}✓ Model directory ready${NC}"
}

generate_synthetic_data() {
    local num_volumes=$1
    local volume_size=$2
    
    echo -e "${CYAN}Generating $num_volumes synthetic 3D medical volumes (${volume_size}³)...${NC}"
    mkdir -p "$DATA_DIR"
    
    python3 << PYTHON_EOF
import numpy as np
import os
import json

data_dir = "$DATA_DIR"
num_volumes = $num_volumes
volume_size = $volume_size

print(f"Generating {num_volumes} synthetic 3D volumes ({volume_size}x{volume_size}x{volume_size})...")

# Generate volumes (CT-like data)
print("Generating volume data...")
volumes = np.random.randn(num_volumes, 1, volume_size, volume_size, volume_size).astype(np.float32)
volumes = volumes * 0.3

print(f"Volume array shape: {volumes.shape}")
print(f"Volume memory: {volumes.nbytes / 1024 / 1024:.1f} MB")

# Generate segmentation labels (3 classes)
print("Generating segmentation labels...")
labels = np.zeros((num_volumes, volume_size, volume_size, volume_size), dtype=np.int64)

for i in range(num_volumes):
    center_d = np.random.randint(volume_size//4, 3*volume_size//4)
    center_h = np.random.randint(volume_size//4, 3*volume_size//4)
    center_w = np.random.randint(volume_size//4, 3*volume_size//4)
    radius = np.random.randint(volume_size//8, volume_size//4)
    
    d, h, w = np.ogrid[:volume_size, :volume_size, :volume_size]
    dist = np.sqrt((d - center_d)**2 + (h - center_h)**2 + (w - center_w)**2)
    
    labels[i][dist < radius] = 1
    labels[i][dist < radius // 3] = 2
    
    if (i + 1) % 10 == 0:
        print(f"  Generated {i + 1}/{num_volumes} labels", end='\\r')

print(f"\\nLabel array shape: {labels.shape}")

# Save data
print("Saving volumes...")
np.save(os.path.join(data_dir, "volumes.npy"), volumes)
print("Saving labels...")
np.save(os.path.join(data_dir, "labels.npy"), labels)

metadata = {
    'num_volumes': num_volumes,
    'volume_shape': [1, volume_size, volume_size, volume_size],
    'num_classes': 3,
    'class_names': ['background', 'kidney', 'tumor'],
    'type': 'synthetic'
}
with open(os.path.join(data_dir, "metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\\nData saved to {data_dir}")
print(f"Total size: {(volumes.nbytes + labels.nbytes) / 1024 / 1024:.1f} MB")
PYTHON_EOF
    
    echo -e "${GREEN}✓ Synthetic data generated${NC}"
}

download_kits19_data() {
    local num_cases=$1
    
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║              Downloading KiTS19 Dataset                    ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}KiTS19 (Kidney Tumor Segmentation Challenge 2019)${NC}"
    echo "  - 300 CT scans with kidney/tumor annotations"
    echo "  - Each case: ~100-500MB (NIfTI format)"
    echo "  - Downloading $num_cases cases for benchmarking"
    echo ""
    
    mkdir -p "$DATA_DIR"
    
    # Check if git is available
    if ! command -v git &> /dev/null; then
        echo -e "${RED}Error: git is required to download KiTS19${NC}"
        echo "Install with: sudo apt install git"
        exit 1
    fi
    
    KITS_REPO_DIR="$DATA_DIR/kits19_repo"
    
    # Clone the KiTS19 repository if not exists
    if [ ! -d "$KITS_REPO_DIR/.git" ]; then
        echo -e "${CYAN}Cloning KiTS19 repository...${NC}"
        rm -rf "$KITS_REPO_DIR"
        git clone https://github.com/neheller/kits19.git "$KITS_REPO_DIR"
    else
        echo -e "${GREEN}✓ KiTS19 repository already cloned${NC}"
    fi
    
    # Install requirements
    pip install -q nibabel requests tqdm
    
    # Download using the official KiTS19 starter code
    echo -e "${CYAN}Downloading KiTS19 imaging data ($num_cases cases)...${NC}"
    echo -e "${YELLOW}This may take a while depending on your connection speed.${NC}"
    echo ""
    
    python3 << PYTHON_EOF
import os
import sys
import json
import numpy as np

data_dir = "$DATA_DIR"
repo_dir = "$KITS_REPO_DIR"
num_cases = $num_cases

# Add kits19 repo to path
sys.path.insert(0, repo_dir)

print(f"Downloading {num_cases} KiTS19 cases using official method...")
print()

# KiTS19 uses DigitalOcean Spaces for hosting the imaging data
# URL pattern: https://kits19.sfo2.digitaloceanspaces.com/master_XXXXX.nii.gz
# Segmentation labels are included in the git repository

import nibabel as nib
from tqdm import tqdm
import requests

# Create raw data directory
raw_dir = os.path.join(data_dir, "raw")
os.makedirs(raw_dir, exist_ok=True)

downloaded_cases = []
failed_cases = []

# Direct download from DigitalOcean Spaces (official KiTS19 hosting)
DO_SPACES_BASE = "https://kits19.sfo2.digitaloceanspaces.com"

for case_id in tqdm(range(num_cases), desc="Downloading cases"):
    case_name = f"case_{case_id:05d}"
    case_dir = os.path.join(raw_dir, case_name)
    os.makedirs(case_dir, exist_ok=True)
    
    imaging_path = os.path.join(case_dir, "imaging.nii.gz")
    seg_path = os.path.join(case_dir, "segmentation.nii.gz")
    
    if os.path.exists(imaging_path) and os.path.exists(seg_path):
        downloaded_cases.append(case_id)
        continue
    
    try:
        # Download imaging from DigitalOcean Spaces
        # Format: https://kits19.sfo2.digitaloceanspaces.com/master_XXXXX.nii.gz
        img_url = f"{DO_SPACES_BASE}/master_{case_id:05d}.nii.gz"
        
        # Segmentation is in the git repo, copy from there
        repo_seg_path = os.path.join(repo_dir, "data", case_name, "segmentation.nii.gz")
        
        # Download imaging with progress
        response = requests.get(img_url, stream=True, timeout=120)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            with open(imaging_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
            
            # Copy segmentation from repo
            if os.path.exists(repo_seg_path):
                import shutil
                shutil.copy2(repo_seg_path, seg_path)
                downloaded_cases.append(case_id)
            else:
                raise Exception("Segmentation not in repo")
        else:
            raise Exception(f"HTTP {response.status_code}")
    except Exception as e:
        print(f"\n  Case {case_id}: {str(e)[:40]}")
        failed_cases.append(case_id)

print(f"\n\nDownload summary: {len(downloaded_cases)}/{num_cases} cases downloaded")

if failed_cases:
    print(f"Failed cases: {len(failed_cases)}")
    print("\nNote: Some cases may require manual download from:")
    print("  https://github.com/neheller/kits19")
    print("  Run: python -m starter_code.get_imaging")

# Preprocess downloaded data
print("\n" + "="*60)
print("Preprocessing data for benchmarking...")
print("="*60)

volumes_list = []
labels_list = []
target_size = 128

for case_id in tqdm(range(num_cases), desc="Processing"):
    case_name = f"case_{case_id:05d}"
    case_dir = os.path.join(raw_dir, case_name)
    
    imaging_path = os.path.join(case_dir, "imaging.nii.gz")
    seg_path = os.path.join(case_dir, "segmentation.nii.gz")
    
    if os.path.exists(imaging_path) and os.path.exists(seg_path):
        # Load real data
        try:
            img = nib.load(imaging_path)
            seg = nib.load(seg_path)
            
            img_data = img.get_fdata().astype(np.float32)
            seg_data = seg.get_fdata().astype(np.int64)
            
            # Normalize CT values (window: -200 to 300 HU)
            img_data = np.clip(img_data, -200, 300)
            img_data = (img_data - 50) / 250  # Center around 0
            
            # Center crop to target size
            d, h, w = img_data.shape
            start_d = max(0, (d - target_size) // 2)
            start_h = max(0, (h - target_size) // 2)
            start_w = max(0, (w - target_size) // 2)
            
            img_crop = img_data[start_d:start_d+target_size,
                               start_h:start_h+target_size,
                               start_w:start_w+target_size]
            seg_crop = seg_data[start_d:start_d+target_size,
                               start_h:start_h+target_size,
                               start_w:start_w+target_size]
            
            # Pad if smaller
            final_img = np.zeros((target_size, target_size, target_size), dtype=np.float32)
            final_seg = np.zeros((target_size, target_size, target_size), dtype=np.int64)
            
            pd, ph, pw = min(target_size, img_crop.shape[0]), min(target_size, img_crop.shape[1]), min(target_size, img_crop.shape[2])
            final_img[:pd, :ph, :pw] = img_crop[:pd, :ph, :pw]
            final_seg[:pd, :ph, :pw] = seg_crop[:pd, :ph, :pw]
            
            volumes_list.append(final_img[np.newaxis, ...])
            labels_list.append(final_seg)
            continue
        except Exception as e:
            print(f"\n  Error processing {case_name}: {e}")
    
    # Generate synthetic for missing cases
    volume = np.random.randn(1, target_size, target_size, target_size).astype(np.float32) * 0.5
    label = np.zeros((target_size, target_size, target_size), dtype=np.int64)
    
    # Add kidney structures
    d, h, w = np.ogrid[:target_size, :target_size, :target_size]
    kd, kh, kw = target_size//2, target_size//2 - 20, target_size//2
    
    dist_l = np.sqrt((d - kd)**2 + (h - kh)**2 + (w - kw)**2)
    dist_r = np.sqrt((d - kd)**2 + (h - (kh + 40))**2 + (w - kw)**2)
    
    label[dist_l < 25] = 1
    label[dist_r < 25] = 1
    volume[0][dist_l < 25] += 0.5
    volume[0][dist_r < 25] += 0.5
    
    if np.random.random() > 0.3:
        td, th, tw = kd + np.random.randint(-10, 10), kh + np.random.randint(-10, 10), kw + np.random.randint(-5, 5)
        dist_t = np.sqrt((d - td)**2 + (h - th)**2 + (w - tw)**2)
        label[dist_t < 10] = 2
    
    volumes_list.append(volume)
    labels_list.append(label)

# Save
volumes = np.stack(volumes_list, axis=0)
labels = np.stack(labels_list, axis=0)

print(f"\nSaving preprocessed data...")
print(f"  Volumes: {volumes.shape}")
print(f"  Labels: {labels.shape}")

np.save(os.path.join(data_dir, "volumes.npy"), volumes)
np.save(os.path.join(data_dir, "labels.npy"), labels)

# Metadata
real_count = len(downloaded_cases)
metadata = {
    'num_volumes': num_cases,
    'real_cases': real_count,
    'synthetic_cases': num_cases - real_count,
    'volume_shape': [1, target_size, target_size, target_size],
    'num_classes': 3,
    'class_names': ['background', 'kidney', 'tumor'],
    'type': 'kits19_real' if real_count > num_cases // 2 else 'kits19_mixed',
    'downloaded_case_ids': downloaded_cases
}
with open(os.path.join(data_dir, "metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'='*60}")
print(f"✓ KiTS19 data ready!")
print(f"  Real cases: {real_count}")
print(f"  Synthetic cases: {num_cases - real_count}")
print(f"  Total size: {(volumes.nbytes + labels.nbytes) / 1024 / 1024:.1f} MB")
print(f"  Location: {data_dir}")
PYTHON_EOF
    
    cd "$PROJECT_DIR"
    echo -e "${GREEN}✓ KiTS19 data prepared${NC}"
}

# ============================================================================
# Main Execution
# ============================================================================

export MODEL_DIR DATA_DIR

check_existing_data

if [ "$SKIP_DOWNLOAD" = false ]; then
    setup_model
    
    case $MODEL_SIZE in
        small)
            VOL_SIZE=64
            DEFAULT_VOLUMES=20
            ;;
        sample)
            VOL_SIZE=128
            DEFAULT_VOLUMES=50
            ;;
        full)
            VOL_SIZE=128
            DEFAULT_VOLUMES=100
            ;;
    esac
    
    # Use NUM_CASES if specified, otherwise use default based on model size
    if [ "$DATA_TYPE" = "real" ]; then
        NUM_VOLUMES=${NUM_CASES:-$DEFAULT_VOLUMES}
        echo -e "${CYAN}Downloading $NUM_VOLUMES KiTS19 cases...${NC}"
    else
        NUM_VOLUMES=$DEFAULT_VOLUMES
    fi
    
    case $DATA_TYPE in
        real)
            download_kits19_data $NUM_VOLUMES
            ;;
        synthetic|*)
            generate_synthetic_data $NUM_VOLUMES $VOL_SIZE
            ;;
    esac
fi

# Auto-detect actual data type for display
ACTUAL_DATA_TYPE="$DATA_TYPE"
if [ -d "$DATA_DIR/case_00000" ] || [ -f "$DATA_DIR/preprocessed/volumes.npy" ]; then
    ACTUAL_DATA_TYPE="real"
elif [ -f "$DATA_DIR/volumes.npy" ]; then
    # Check metadata to distinguish real preprocessed from synthetic
    if [ -f "$DATA_DIR/metadata.json" ]; then
        TYPE_CHECK=$(python3 -c "import json; d=json.load(open('$DATA_DIR/metadata.json')); print(d.get('type', 'synthetic'))" 2>/dev/null || echo "synthetic")
        if [[ "$TYPE_CHECK" == *"kits19"* ]]; then
            ACTUAL_DATA_TYPE="real"
        else
            ACTUAL_DATA_TYPE="synthetic"
        fi
    else
        ACTUAL_DATA_TYPE="synthetic"
    fi
fi

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              3D-UNet Benchmark Configuration               ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo "  Model Size: $MODEL_SIZE"
echo "  Data Type:  $ACTUAL_DATA_TYPE"
echo "  Device:     $DEVICE"
echo "  Offload:    $USE_OFFLOAD"
echo "  Data Dir:   $DATA_DIR"
echo ""

CMD="python3 $SCRIPT_DIR/run_3dunet_benchmark.py"
CMD="$CMD --model-dir $MODEL_DIR"
CMD="$CMD --data-dir $DATA_DIR"
CMD="$CMD --device $DEVICE"
CMD="$CMD --model-size $MODEL_SIZE"

[ -n "$BATCH_SIZE" ] && CMD="$CMD --batch-size $BATCH_SIZE"
[ -n "$MAX_EXAMPLES" ] && CMD="$CMD --max-examples $MAX_EXAMPLES"
[ "$USE_OFFLOAD" = true ] && CMD="$CMD --offload"
[ "$MLPERF_MODE" = true ] && CMD="$CMD --mlperf"

exec $CMD