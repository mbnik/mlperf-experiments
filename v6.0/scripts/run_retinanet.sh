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
mlperf_mode = "$MLPERF_MODE" == "true"

# Use 800x800 for MLPerf compliance, 640x480 otherwise
if mlperf_mode:
    image_height, image_width = 800, 800
    print(f"Generating {num_images} synthetic images (800x800 - MLPerf spec)...")
else:
    image_height, image_width = 480, 640
    print(f"Generating {num_images} synthetic images (640x480)...")

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

# Generate annotations (264 MLPerf classes or 91 COCO classes)
num_classes = 264 if mlperf_mode else 91
annotations = []
for i in range(num_images):
    num_objects = np.random.randint(1, 11)
    ann = {'image_id': i, 'boxes': [], 'labels': []}
    for _ in range(num_objects):
        x1, y1 = np.random.randint(0, image_width-50), np.random.randint(0, image_height-50)
        x2, y2 = x1 + np.random.randint(20, 100), y1 + np.random.randint(20, 100)
        ann['boxes'].append([int(x1), int(y1), int(min(x2, image_width)), int(min(y2, image_height))])
        ann['labels'].append(int(np.random.randint(1, num_classes + 1)))
    annotations.append(ann)

with open(os.path.join(data_dir, "annotations.json"), 'w') as f:
    json.dump(annotations, f)

# Save metadata
metadata = {
    'num_images': num_images,
    'image_shape': [image_height, image_width, 3],
    'type': 'synthetic',
    'num_classes': num_classes,
    'mlperf_compliant': mlperf_mode
}
with open(os.path.join(data_dir, "metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

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
    echo -e "${YELLOW}OpenImages V6 MLPerf Validation Set${NC}"
    echo "  - Official MLPerf dataset for RetinaNet"
    echo "  - 264 object classes (MLPerf subset)"
    echo "  - Images resized to 800x800 (MLPerf spec)"
    echo "  - Downloading up to $num_images images"
    echo ""
    
    mkdir -p "$DATA_DIR"
    
    python3 << PYTHON_EOF
import os
import sys
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import pandas as pd

data_dir = "$DATA_DIR"
num_images = $num_images
mlperf_mode = "$MLPERF_MODE" == "true"

# MLPerf official 264 classes for RetinaNet
MLPERF_CLASSES = [
    'Airplane', 'Antelope', 'Apple', 'Backpack', 'Balloon', 'Banana',
    'Barrel', 'Baseball bat', 'Baseball glove', 'Bee', 'Beer', 'Bench', 'Bicycle',
    'Bicycle helmet', 'Bicycle wheel', 'Billboard', 'Book', 'Bookcase', 'Boot',
    'Bottle', 'Bowl', 'Bowling equipment', 'Box', 'Boy', 'Brassiere', 'Bread',
    'Broccoli', 'Bronze sculpture', 'Bull', 'Bus', 'Bust', 'Butterfly', 'Cabinetry',
    'Cake', 'Camel', 'Camera', 'Candle', 'Candy', 'Cannon', 'Canoe', 'Carrot', 'Cart',
    'Castle', 'Cat', 'Cattle', 'Cello', 'Chair', 'Cheese', 'Chest of drawers', 'Chicken',
    'Christmas tree', 'Coat', 'Cocktail', 'Coffee', 'Coffee cup', 'Coffee table', 'Coin',
    'Common sunflower', 'Computer keyboard', 'Computer monitor', 'Convenience store',
    'Cookie', 'Countertop', 'Cowboy hat', 'Crab', 'Crocodile', 'Cucumber', 'Cupboard',
    'Curtain', 'Deer', 'Desk', 'Dinosaur', 'Dog', 'Doll', 'Dolphin', 'Door', 'Dragonfly',
    'Drawer', 'Dress', 'Drum', 'Duck', 'Eagle', 'Earrings', 'Egg (Food)', 'Elephant',
    'Falcon', 'Fedora', 'Flag', 'Flowerpot', 'Football', 'Football helmet', 'Fork',
    'Fountain', 'French fries', 'French horn', 'Frog', 'Giraffe', 'Girl', 'Glasses',
    'Goat', 'Goggles', 'Goldfish', 'Gondola', 'Goose', 'Grape', 'Grapefruit', 'Guitar',
    'Hamburger', 'Handbag', 'Harbor seal', 'Headphones', 'Helicopter', 'High heels',
    'Hiking equipment', 'Horse', 'House', 'Houseplant', 'Human arm', 'Human beard',
    'Human body', 'Human ear', 'Human eye', 'Human face', 'Human foot', 'Human hair',
    'Human hand', 'Human head', 'Human leg', 'Human mouth', 'Human nose', 'Ice cream',
    'Jacket', 'Jeans', 'Jellyfish', 'Juice', 'Kitchen & dining room table', 'Kite',
    'Lamp', 'Lantern', 'Laptop', 'Lavender (Plant)', 'Lemon', 'Light bulb', 'Lighthouse',
    'Lily', 'Lion', 'Lipstick', 'Lizard', 'Man', 'Maple', 'Microphone', 'Mirror',
    'Mixing bowl', 'Mobile phone', 'Monkey', 'Motorcycle', 'Muffin', 'Mug', 'Mule',
    'Mushroom', 'Musical keyboard', 'Necklace', 'Nightstand', 'Office building',
    'Orange', 'Owl', 'Oyster', 'Paddle', 'Palm tree', 'Parachute', 'Parrot', 'Pen',
    'Penguin', 'Personal flotation device', 'Piano', 'Picture frame', 'Pig', 'Pillow',
    'Pizza', 'Plate', 'Platter', 'Porch', 'Poster', 'Pumpkin', 'Rabbit', 'Rifle',
    'Roller skates', 'Rose', 'Salad', 'Sandal', 'Saucer', 'Saxophone', 'Scarf', 'Sea lion',
    'Sea turtle', 'Sheep', 'Shelf', 'Shirt', 'Shorts', 'Shrimp', 'Sink', 'Skateboard',
    'Ski', 'Skull', 'Skyscraper', 'Snake', 'Sock', 'Sofa bed', 'Sparrow', 'Spider', 'Spoon',
    'Sports uniform', 'Squirrel', 'Stairs', 'Stool', 'Strawberry', 'Street light',
    'Studio couch', 'Suit', 'Sun hat', 'Sunglasses', 'Surfboard', 'Sushi', 'Swan',
    'Swimming pool', 'Swimwear', 'Tank', 'Tap', 'Taxi', 'Tea', 'Teddy bear', 'Television',
    'Tent', 'Tie', 'Tiger', 'Tin can', 'Tire', 'Toilet', 'Tomato', 'Tortoise', 'Tower',
    'Traffic light', 'Train', 'Tripod', 'Truck', 'Trumpet', 'Umbrella', 'Van', 'Vase',
    'Vehicle registration plate', 'Violin', 'Wall clock', 'Waste container', 'Watch',
    'Whale', 'Wheel', 'Wheelchair', 'Whiteboard', 'Window', 'Wine', 'Wine glass', 'Woman',
    'Zebra', 'Zucchini'
]

# Create class name to ID mapping
class_to_id = {name: i+1 for i, name in enumerate(MLPERF_CLASSES)}

print("Setting up OpenImages V6 MLPerf dataset...")

# Download class descriptions
map_classes_url = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
map_classes_file = os.path.join(data_dir, "class-descriptions-boxable.csv")

if not os.path.exists(map_classes_file):
    print("Downloading class descriptions...")
    urllib.request.urlretrieve(map_classes_url, map_classes_file)

# Download validation annotations
ann_url = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
ann_file = os.path.join(data_dir, "validation-annotations-bbox.csv")

if not os.path.exists(ann_file):
    print("Downloading validation annotations (~50MB)...")
    urllib.request.urlretrieve(ann_url, ann_file)

# Load class mapping
print("Loading class mappings...")
class_map = {}
with open(map_classes_file) as f:
    for line in f:
        parts = line.strip().split(',', 1)
        if len(parts) == 2:
            class_map[parts[0]] = parts[1]

# Find MLPerf class codes
mlperf_class_codes = {}
for code, name in class_map.items():
    if name in MLPERF_CLASSES:
        mlperf_class_codes[code] = name

print(f"Found {len(mlperf_class_codes)} MLPerf classes in OpenImages")

# Load annotations and filter by MLPerf classes
print("Loading annotations...")
ann_df = pd.read_csv(ann_file)

# Filter to MLPerf classes only
ann_df = ann_df[ann_df['LabelName'].isin(mlperf_class_codes.keys())]
print(f"Filtered to {len(ann_df)} annotations in MLPerf classes")

# Get unique images
unique_images = ann_df['ImageID'].unique()
print(f"Found {len(unique_images)} images with MLPerf class annotations")

# Limit to requested number
selected_images = unique_images[:num_images]
print(f"Selecting {len(selected_images)} images for download")

# Create images directory
images_dir = os.path.join(data_dir, "validation", "data")
os.makedirs(images_dir, exist_ok=True)

# Download function
import boto3
from botocore import UNSIGNED
from botocore.config import Config

def download_image(image_id):
    output_path = os.path.join(images_dir, f"{image_id}.jpg")
    if os.path.exists(output_path):
        return output_path
    
    try:
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        s3.download_file('open-images-dataset', f'validation/{image_id}.jpg', output_path)
        return output_path
    except Exception as e:
        # Fallback to direct URL
        try:
            url = f"https://storage.googleapis.com/openimages/validation/{image_id}.jpg"
            urllib.request.urlretrieve(url, output_path)
            return output_path
        except:
            return None

# Download images in parallel
print(f"Downloading {len(selected_images)} images from OpenImages...")
successful = 0
failed = 0

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(tqdm(executor.map(download_image, selected_images), 
                       total=len(selected_images), desc="Downloading"))
    successful = sum(1 for r in results if r is not None)
    failed = sum(1 for r in results if r is None)

print(f"Downloaded: {successful}, Failed: {failed}")

# Process images and create annotations in MLPerf format
print("Processing images and creating annotations...")

images_list = []
annotations = []
mlperf_annotations = {"images": [], "categories": [], "annotations": []}

# Add categories
for i, name in enumerate(MLPERF_CLASSES):
    mlperf_annotations["categories"].append({"id": i+1, "name": name})

ann_id = 1
for idx, image_id in enumerate(tqdm(selected_images, desc="Processing")):
    img_path = os.path.join(images_dir, f"{image_id}.jpg")
    if not os.path.exists(img_path):
        continue
    
    try:
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        
        # MLPerf requires 800x800 resized images
        img_resized = img.resize((800, 800))
        images_list.append(np.array(img_resized))
        
        # Scale factors
        scale_x = 800 / orig_w
        scale_y = 800 / orig_h
        
        # Get annotations for this image
        img_anns = ann_df[ann_df['ImageID'] == image_id]
        
        ann = {'image_id': idx, 'boxes': [], 'labels': []}
        
        # Add to COCO-style annotations
        mlperf_annotations["images"].append({
            "id": idx,
            "file_name": f"{image_id}.jpg",
            "width": 800,
            "height": 800
        })
        
        for _, row in img_anns.iterrows():
            class_name = mlperf_class_codes.get(row['LabelName'])
            if not class_name:
                continue
            
            # OpenImages uses normalized coordinates
            x1 = int(row['XMin'] * orig_w * scale_x)
            y1 = int(row['YMin'] * orig_h * scale_y)
            x2 = int(row['XMax'] * orig_w * scale_x)
            y2 = int(row['YMax'] * orig_h * scale_y)
            
            label_id = class_to_id[class_name]
            ann['boxes'].append([x1, y1, x2, y2])
            ann['labels'].append(label_id)
            
            # COCO format annotation
            mlperf_annotations["annotations"].append({
                "id": ann_id,
                "image_id": idx,
                "category_id": label_id,
                "bbox": [x1, y1, x2-x1, y2-y1],
                "area": (x2-x1) * (y2-y1),
                "iscrowd": 0
            })
            ann_id += 1
        
        annotations.append(ann)
        
    except Exception as e:
        print(f"Error processing {image_id}: {e}")
        continue

if len(images_list) == 0:
    print("ERROR: No images were successfully processed!")
    sys.exit(1)

images = np.stack(images_list)

# Save processed data
print(f"Saving {images.shape}...")
np.save(os.path.join(data_dir, "images.npy"), images)

with open(os.path.join(data_dir, "annotations.json"), 'w') as f:
    json.dump(annotations, f)

# Save COCO-style annotations for MLPerf evaluation
with open(os.path.join(data_dir, "openimages-mlperf.json"), 'w') as f:
    json.dump(mlperf_annotations, f)

# Save metadata
metadata = {
    'num_images': len(images),
    'image_shape': [800, 800, 3],
    'type': 'real',
    'source': 'openimages_v6_mlperf',
    'num_classes': len(MLPERF_CLASSES),
    'mlperf_compliant': True
}
with open(os.path.join(data_dir, "metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ OpenImages V6 MLPerf data saved to {data_dir}")
print(f"  Images: {images.shape}")
print(f"  Annotations: {len(annotations)} images")
print(f"  Classes: {len(MLPERF_CLASSES)} (MLPerf subset)")
print(f"  Format: 800x800 (MLPerf spec)")
PYTHON_EOF
    
    echo -e "${GREEN}✓ OpenImages MLPerf data ready!${NC}"
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

# Detect actual data type based on what Python script will use
ACTUAL_DATA_TYPE="synthetic"
if [ -f "$DATA_DIR/images.npy" ]; then
    ACTUAL_DATA_TYPE="real"
fi

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              RetinaNet Benchmark Configuration             ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo "  Model Size: $MODEL_SIZE"
echo "  Data Type:  $ACTUAL_DATA_TYPE"
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