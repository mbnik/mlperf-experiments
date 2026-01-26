#!/bin/bash
# ============================================================================
# MLPerf Benchmark Setup and Runner - RetinaNet Object Detection
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
MODEL_DIR="$PROJECT_DIR/models/retinanet"
DATA_DIR="$PROJECT_DIR/data/openimages"

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
DATA_TYPE="synthetic"
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
    echo "  --batch=N       Batch size (default: auto)"
    echo "  --samples=N     Number of samples to process"
    echo "  --quick         Quick test with minimal samples"
    echo "  -h, --help      Show this help"
    echo ""
    echo "Data Options:"
    echo "  synthetic  - Generate random images (fast, no download)"
    echo "  real       - Download OpenImages validation subset (~2GB)"
    echo ""
    echo "Examples:"
    echo "  $0 --gpu --data=real              # Real OpenImages data"
    echo "  $0 --gpu --data=synthetic         # Synthetic images (fast)"
    echo "  $0 --gpu --mlperf                 # Official MLPerf settings"
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
        --quick)
            MAX_EXAMPLES="50"
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

# Apply MLPerf settings
if [[ "$MLPERF_MODE" == "true" ]]; then
    if [[ "$EXPLICIT_SYNTHETIC" != "true" ]]; then
        DATA_TYPE="real"
    fi
fi

# Print run configuration
echo -e "${CYAN}Running RetinaNet Object Detection Benchmark${NC}"
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
    
    if [ -f "$MODEL_DIR/retinanet_resnet50_fpn.pt" ]; then
        local model_size=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
        model_exists=true
        model_size_str="$model_size"
    fi
    
    if [ -f "$DATA_DIR/images.npy" ]; then
        local data_size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
        data_exists=true
        data_size_str="$data_size"
    fi
    
    if $model_exists || $data_exists; then
        echo ""
        echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${CYAN}║              Existing RetinaNet Data Detected              ║${NC}"
        echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        
        if $model_exists; then
            echo -e "  ${GREEN}✓${NC} Model found: ${model_size_str} at $MODEL_DIR"
        else
            echo -e "  ${YELLOW}✗${NC} Model not found"
        fi
        
        if $data_exists; then
            echo -e "  ${GREEN}✓${NC} Dataset found: ${data_size_str} at $DATA_DIR"
        else
            echo -e "  ${YELLOW}✗${NC} Dataset not found"
        fi
        
        echo ""
        echo "Options:"
        echo "  [S] Skip download - Use existing data (default)"
        echo "  [R] Re-download - Fresh download"
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
# Download/Setup Functions
# ============================================================================
download_model() {
    echo -e "${CYAN}Downloading RetinaNet model...${NC}"
    mkdir -p "$MODEL_DIR"
    
    python3 << 'PYTHON_EOF'
import torch
import torchvision
import os

model_dir = os.environ.get('MODEL_DIR', 'models/retinanet')
os.makedirs(model_dir, exist_ok=True)

print("Downloading RetinaNet ResNet50 FPN model...")
model = torchvision.models.detection.retinanet_resnet50_fpn(
    weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.COCO_V1
)
model.eval()

model_path = os.path.join(model_dir, "retinanet_resnet50_fpn.pt")
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB)")
PYTHON_EOF
    
    echo -e "${GREEN}✓ Model downloaded${NC}"
}

generate_data() {
    local num_images=$1
    echo -e "${CYAN}Generating $num_images synthetic images...${NC}"
    mkdir -p "$DATA_DIR"
    
    python3 << PYTHON_EOF
import numpy as np
import os
import json

data_dir = "$DATA_DIR"
num_images = $num_images

print(f"Generating {num_images} synthetic images (640x480)...")

image_height, image_width = 480, 640
batch_size = 100
all_images = []

for i in range(0, num_images, batch_size):
    batch_count = min(batch_size, num_images - i)
    batch = np.random.randint(0, 256, (batch_count, image_height, image_width, 3), dtype=np.uint8)
    all_images.append(batch)
    print(f"  Generated {i + batch_count}/{num_images}", end='\r')

images = np.concatenate(all_images, axis=0)
print(f"\nSaving {images.shape}...")
np.save(os.path.join(data_dir, "images.npy"), images)

# Generate annotations
annotations = []
for i in range(num_images):
    num_objects = np.random.randint(1, 11)
    ann = {'image_id': i, 'boxes': [], 'labels': []}
    for _ in range(num_objects):
        x1, y1 = np.random.randint(0, image_width-50), np.random.randint(0, image_height-50)
        x2, y2 = x1 + np.random.randint(20, 100), y1 + np.random.randint(20, 100)
        ann['boxes'].append([int(x1), int(y1), int(min(x2, image_width)), int(min(y2, image_height))])
        ann['labels'].append(int(np.random.randint(1, 91)))
    annotations.append(ann)

with open(os.path.join(data_dir, "annotations.json"), 'w') as f:
    json.dump(annotations, f)

print(f"Data saved to {data_dir}")
PYTHON_EOF
    
    echo -e "${GREEN}✓ Data generated${NC}"
}

download_openimages() {
    local num_images=$1
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║           Downloading OpenImages Dataset                   ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}OpenImages V6 (Official MLPerf dataset for object detection)${NC}"
    echo "  - Downloading $num_images validation images"
    echo "  - Images with COCO-style annotations"
    echo ""
    
    mkdir -p "$DATA_DIR"
    
    python3 << PYTHON_EOF
import os
import json
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import csv

data_dir = "$DATA_DIR"
num_images = $num_images

# OpenImages validation set annotations (COCO-style classes)
# We'll use the fiftyone library to download or fallback to direct URLs

print("Setting up OpenImages data...")

try:
    import fiftyone as fo
    import fiftyone.zoo as foz
    
    print("Downloading OpenImages validation subset via FiftyOne...")
    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        split="validation", 
        max_samples=num_images,
        label_types=["detections"],
    )
    
    # Convert to our format
    images_list = []
    annotations = []
    
    for i, sample in enumerate(tqdm(dataset, desc="Processing")):
        # Load image
        img = Image.open(sample.filepath)
        img = img.convert("RGB")
        img = img.resize((640, 480))
        images_list.append(np.array(img))
        
        # Convert annotations
        ann = {'image_id': i, 'boxes': [], 'labels': []}
        if sample.ground_truth and sample.ground_truth.detections:
            for det in sample.ground_truth.detections:
                x1 = int(det.bounding_box[0] * 640)
                y1 = int(det.bounding_box[1] * 480)
                w = int(det.bounding_box[2] * 640)
                h = int(det.bounding_box[3] * 480)
                ann['boxes'].append([x1, y1, x1+w, y1+h])
                ann['labels'].append(1)  # Generic object class
        annotations.append(ann)
    
    images = np.stack(images_list)
    
except ImportError:
    print("FiftyOne not available, using COCO validation set instead...")
    
    # Download COCO 2017 validation images (more accessible)
    COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
    COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    import zipfile
    import urllib.request
    
    val_zip = os.path.join(data_dir, "val2017.zip")
    ann_zip = os.path.join(data_dir, "annotations_trainval2017.zip")
    
    if not os.path.exists(os.path.join(data_dir, "val2017")):
        print("Downloading COCO 2017 validation images (~1GB)...")
        print("This may take a while...")
        urllib.request.urlretrieve(COCO_VAL_URL, val_zip)
        print("Extracting...")
        with zipfile.ZipFile(val_zip, 'r') as z:
            z.extractall(data_dir)
    
    if not os.path.exists(os.path.join(data_dir, "annotations")):
        print("Downloading COCO 2017 annotations...")
        urllib.request.urlretrieve(COCO_ANN_URL, ann_zip)
        print("Extracting...")
        with zipfile.ZipFile(ann_zip, 'r') as z:
            z.extractall(data_dir)
    
    # Load COCO annotations
    with open(os.path.join(data_dir, "annotations", "instances_val2017.json")) as f:
        coco = json.load(f)
    
    # Build image_id to annotations mapping
    img_to_anns = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Process images
    images_list = []
    annotations = []
    img_dir = os.path.join(data_dir, "val2017")
    
    for i, img_info in enumerate(tqdm(coco["images"][:num_images], desc="Processing")):
        img_path = os.path.join(img_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((640, 480))
        images_list.append(np.array(img))
        
        # Scale annotations
        scale_x = 640 / orig_w
        scale_y = 480 / orig_h
        
        ann = {'image_id': i, 'boxes': [], 'labels': []}
        if img_info["id"] in img_to_anns:
            for coco_ann in img_to_anns[img_info["id"]]:
                x, y, w, h = coco_ann["bbox"]
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + w) * scale_x)
                y2 = int((y + h) * scale_y)
                ann['boxes'].append([x1, y1, x2, y2])
                ann['labels'].append(coco_ann["category_id"])
        annotations.append(ann)
    
    images = np.stack(images_list)

# Save processed data
print(f"Saving {images.shape}...")
np.save(os.path.join(data_dir, "images.npy"), images)

with open(os.path.join(data_dir, "annotations.json"), 'w') as f:
    json.dump(annotations, f)

# Save metadata
metadata = {
    'num_images': len(images),
    'image_shape': list(images.shape[1:]),
    'type': 'real',
    'source': 'coco_val2017'
}
with open(os.path.join(data_dir, "metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Data saved to {data_dir}")
print(f"  Images: {images.shape}")
print(f"  Annotations: {len(annotations)} images with detections")
PYTHON_EOF
    
    echo -e "${GREEN}✓ OpenImages/COCO data ready!${NC}"
}

# ============================================================================
# Main Execution
# ============================================================================

export MODEL_DIR DATA_DIR

check_existing_data

# Setup based on model size
if [ "$SKIP_DOWNLOAD" = false ]; then
    download_model
    case $MODEL_SIZE in
        small)
            NUM_IMAGES=100
            ;;
        sample)
            NUM_IMAGES=1000
            ;;
        full)
            NUM_IMAGES=5000
            ;;
    esac
    
    case $DATA_TYPE in
        real)
            download_openimages $NUM_IMAGES
            ;;
        synthetic|*)
            generate_data $NUM_IMAGES
            ;;
    esac
fi

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              RetinaNet Benchmark Configuration             ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo "  Model Size: $MODEL_SIZE"
echo "  Data Type:  $DATA_TYPE"
echo "  Device:     $DEVICE"
echo "  Offload:    $USE_OFFLOAD"
echo "  Data Dir:   $DATA_DIR"
echo ""

# Build command
CMD="python3 $SCRIPT_DIR/run_retinanet_benchmark.py"
CMD="$CMD --model-dir $MODEL_DIR"
CMD="$CMD --data-dir $DATA_DIR"
CMD="$CMD --device $DEVICE"
CMD="$CMD --model-size $MODEL_SIZE"

[ -n "$BATCH_SIZE" ] && CMD="$CMD --batch-size $BATCH_SIZE"
[ -n "$MAX_EXAMPLES" ] && CMD="$CMD --max-examples $MAX_EXAMPLES"
[ "$USE_OFFLOAD" = true ] && CMD="$CMD --offload"
[ "$MLPERF_MODE" = true ] && CMD="$CMD --mlperf"

exec $CMD