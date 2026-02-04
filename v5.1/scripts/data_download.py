#!/usr/bin/env python3
"""
MLPerf Data Download Utility

Downloads datasets for MLPerf benchmarks using various methods (wget, curl, urllib).
Designed to work in corporate network environments where some methods may be blocked.

Author: Mehdi Nik
Created: Jan 2026

Usage:
    python data_download.py --benchmark bert
    python data_download.py --benchmark bert --method wget
    python data_download.py --url "https://..." --dest data/file.json
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# Dataset Definitions
# ============================================================================

DATASETS = {
    'bert': {
        'name': 'SQuAD v1.1',
        'description': 'Stanford Question Answering Dataset for BERT QA benchmark',
        'task': 'question-answering',
        'default_dataset': 'squad',  # Default folder name for MLPerf data
        'files': [
            {
                'url': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json',
                'dest': 'dev-v1.1.json',
                'size_mb': 4.7,
                'sha256': None,  # Can add checksums for verification
            }
        ],
        'post_download': None,
    },
    
    'resnet50': {
        'name': 'ImageNet ILSVRC2012',
        'description': 'ImageNet validation set via HuggingFace (ILSVRC/imagenet-1k)',
        'task': 'image-classification',
        'default_dataset': 'imagenet',
        'files': [],  # Uses custom download
        'custom_download': 'download_imagenet',
        'requires_auth': True,
        'auth_instructions': """
ImageNet download requires HuggingFace authentication:

1. Create a HuggingFace account: https://huggingface.co/join
2. Get your token: https://huggingface.co/settings/tokens
3. Accept the dataset terms: https://huggingface.co/datasets/ILSVRC/imagenet-1k
4. Login: huggingface-cli login
   OR set HF_TOKEN environment variable
""",
    },
    
    'retinanet': {
        'name': 'OpenImages v6',
        'description': 'OpenImages MLPerf validation subset (264 classes)',
        'task': 'object-detection',
        'default_dataset': 'openimages',
        'files': [],  # Uses custom download
        'custom_download': 'download_openimages',
    },
    
    '3dunet': {
        'name': 'KiTS19',
        'description': 'Kidney Tumor Segmentation Challenge 2019',
        'task': 'medical-segmentation',
        'default_dataset': 'kits19',
        'files': [],  # Uses git clone + custom downloader
        'custom_download': 'download_kits19',
    },
    
    'whisper': {
        'name': 'LibriSpeech',
        'description': 'LibriSpeech dev-clean and dev-other for ASR',
        'task': 'speech-recognition',
        'default_dataset': 'librispeech',
        'files': [
            {
                'url': 'https://www.openslr.org/resources/12/dev-clean.tar.gz',
                'dest': 'dev-clean.tar.gz',
                'size_mb': 337,
                'extract': True,
            },
            {
                'url': 'https://www.openslr.org/resources/12/dev-other.tar.gz',
                'dest': 'dev-other.tar.gz',
                'size_mb': 314,
                'extract': True,
            },
        ],
        'post_download': 'create_librispeech_manifest',
    },
    
    'sdxl': {
        'name': 'COCO 2014',
        'description': 'COCO 2014 validation images and captions for SDXL text-to-image benchmark',
        'task': 'text-to-image',
        'default_dataset': 'coco-2014',
        'files': [
            {
                'url': 'http://images.cocodataset.org/zips/val2014.zip',
                'dest': 'val2014.zip',
                'size_mb': 6200,
                'extract': True,
                'optional': True,  # Images optional, needed for FID score
            },
            {
                'url': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                'dest': 'annotations_trainval2014.zip',
                'size_mb': 252,
                'extract': True,
            },
        ],
        'post_download': 'extract_coco_captions',
        'note': 'Images (~6.2GB) needed for FID accuracy. Captions (~252MB) required for benchmark.',
    },
    
    'gptj': {
        'name': 'CNN-DailyMail',
        'description': 'CNN-DailyMail summarization dataset',
        'task': 'text-summarization',
        'default_dataset': 'cnn-dailymail',
        'files': [],  # Uses HuggingFace datasets or custom
        'custom_download': 'download_cnn_dailymail',
    },
    
    'llama': {
        'name': 'OpenOrca / CNN-DailyMail',
        'description': 'OpenOrca for 70B, CNN-DailyMail for 8B',
        'task': 'text-generation',
        'default_dataset': 'openorca',
        'files': [],
        'custom_download': 'download_llama_data',
    },
    
    'mixtral': {
        'name': 'MLPerf Mixtral Combined',
        'description': 'OpenOrca + GSM8K + MBXP (15K samples)',
        'task': 'text-generation',
        'default_dataset': 'mixtral-15k',
        'files': [],
        'custom_download': 'download_mixtral_data',
    },
    
    'dlrm': {
        'name': 'Criteo Terabyte',
        'description': 'Criteo click-through rate prediction dataset',
        'task': 'recommendation',
        'default_dataset': 'criteo',
        'files': [],  # Very large real data, needs special handling
        'custom_download': 'download_criteo',
        # Note: download_criteo generates synthetic data matching the format
        # Real Criteo data (~1TB) requires manual download
    },
}

# Download methods in order of preference for corporate networks
DOWNLOAD_METHODS = ['wget', 'curl', 'urllib']


# ============================================================================
# Download Functions
# ============================================================================

def check_command_exists(cmd: str) -> bool:
    """Check if a command exists in PATH."""
    return shutil.which(cmd) is not None


def get_available_methods() -> List[str]:
    """Get list of available download methods."""
    available = []
    if check_command_exists('wget'):
        available.append('wget')
    if check_command_exists('curl'):
        available.append('curl')
    available.append('urllib')  # Always available
    return available


def download_with_wget(url: str, dest: Path, resume: bool = True) -> Tuple[bool, str]:
    """Download file using wget."""
    try:
        cmd = ['wget', '-q', '--show-progress']
        if resume:
            cmd.append('-c')  # Continue partial downloads
        cmd.extend(['-O', str(dest), url])
        
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode == 0, ""
    except Exception as e:
        return False, str(e)


def download_with_curl(url: str, dest: Path, resume: bool = True) -> Tuple[bool, str]:
    """Download file using curl."""
    try:
        cmd = ['curl', '-L', '--progress-bar']
        if resume:
            cmd.append('-C')
            cmd.append('-')  # Continue from where it left off
        cmd.extend(['-o', str(dest), url])
        
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode == 0, ""
    except Exception as e:
        return False, str(e)


def download_with_urllib(url: str, dest: Path, resume: bool = False) -> Tuple[bool, str]:
    """Download file using Python urllib."""
    try:
        # Simple progress indicator
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, block_num * block_size * 100 // total_size)
                mb_downloaded = block_num * block_size / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Downloading: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
        
        urllib.request.urlretrieve(url, dest, progress_hook)
        print()  # New line after progress
        return True, ""
    except Exception as e:
        return False, str(e)


def download_file(url: str, dest: Path, method: str = 'auto', 
                  resume: bool = True, verbose: bool = True) -> Tuple[bool, str]:
    """
    Download a file using the specified or auto-detected method.
    
    Args:
        url: URL to download
        dest: Destination path
        method: 'wget', 'curl', 'urllib', or 'auto'
        resume: Whether to resume partial downloads
        verbose: Print progress
        
    Returns:
        Tuple of (success, error_message)
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if method == 'auto':
        methods_to_try = get_available_methods()
    else:
        methods_to_try = [method]
    
    for m in methods_to_try:
        if verbose:
            print(f"  Downloading with {m}...")
        
        if m == 'wget':
            success, error = download_with_wget(url, dest, resume)
        elif m == 'curl':
            success, error = download_with_curl(url, dest, resume)
        elif m == 'urllib':
            success, error = download_with_urllib(url, dest, resume)
        else:
            continue
        
        if success:
            return True, m  # Return method that worked
        elif verbose and method == 'auto':
            print(f"    {m} failed, trying next method...")
    
    return False, f"All download methods failed for {url}"


def extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    """Extract a compressed archive."""
    import tarfile
    import zipfile
    
    archive_str = str(archive_path)
    
    try:
        if archive_str.endswith('.zip'):
            print(f"  Extracting {archive_path.name}...")
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(dest_dir)
            return True
        elif archive_str.endswith('.tar.gz') or archive_str.endswith('.tgz'):
            print(f"  Extracting {archive_path.name}...")
            with tarfile.open(archive_path, 'r:gz') as tf:
                tf.extractall(dest_dir)
            return True
        elif archive_str.endswith('.tar'):
            print(f"  Extracting {archive_path.name}...")
            with tarfile.open(archive_path, 'r') as tf:
                tf.extractall(dest_dir)
            return True
        else:
            print(f"  Unknown archive format: {archive_path.name}")
            return False
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return False


def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """Verify file checksum."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest() == expected_sha256


# ============================================================================
# Custom Download Handlers
# ============================================================================

def download_imagenet(data_dir: Path, method: str = 'auto',
                      num_images: int = 50000) -> Tuple[bool, Dict]:
    """
    Download ImageNet ILSVRC2012 validation set via HuggingFace.
    
    Uses the ILSVRC/imagenet-1k dataset which requires:
    1. HuggingFace account and token
    2. Accepted dataset terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k
    
    Args:
        data_dir: Destination directory
        method: 'hf' (datasets library) or 'wget' (parquet files)
        num_images: Number of images to download (50000 for full validation set)
    
    Returns:
        (success, info_dict)
    """
    val_dir = data_dir / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    existing_images = sum(1 for _ in val_dir.rglob("*.JPEG"))
    if existing_images >= num_images:
        print(f"  ✓ ImageNet already downloaded: {existing_images} images")
        return True, {'downloaded': existing_images, 'skipped': True}
    
    print(f"\n  Downloading ImageNet validation set ({num_images} images)...")
    print("  This requires HuggingFace authentication.")
    print("-" * 50)
    
    # Try HuggingFace datasets library first (preferred)
    try:
        return _download_imagenet_hf(val_dir, num_images)
    except Exception as e:
        print(f"  HuggingFace datasets failed: {e}")
        print("  Trying wget method with parquet files...")
        return _download_imagenet_wget(data_dir, num_images)


def _download_imagenet_hf(val_dir: Path, num_images: int) -> Tuple[bool, Dict]:
    """Download ImageNet using HuggingFace datasets library (streaming)."""
    from datasets import load_dataset
    
    print(f"  Loading ImageNet-1K validation (streaming)...")
    print("  (Requires: huggingface-cli login)")
    
    ds = load_dataset('ILSVRC/imagenet-1k', split='validation', streaming=True)
    
    count = 0
    for item in ds:
        if count >= num_images:
            break
        
        label = item['label']
        label_dir = val_dir / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        img_path = label_dir / f'img_{count}.JPEG'
        item['image'].save(img_path)
        
        count += 1
        if count % 1000 == 0:
            print(f"  Downloaded {count}/{num_images} images...")
    
    print(f"  ✓ Downloaded {count} images to {val_dir}")
    
    return True, {
        'downloaded': count,
        'method': 'huggingface',
        'val_dir': str(val_dir),
    }


def _download_imagenet_wget(data_dir: Path, num_images: int) -> Tuple[bool, Dict]:
    """Download ImageNet via wget from HuggingFace parquet files."""
    import getpass
    import io
    
    try:
        import pyarrow.parquet as pq
        from PIL import Image
    except ImportError:
        print("  ERROR: pyarrow and pillow required for wget method")
        print("  Install with: pip install pyarrow pillow")
        return False, {'error': 'missing dependencies'}
    
    val_dir = data_dir / "val"
    parquet_dir = data_dir / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get HF token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("\n  HuggingFace token required for ImageNet download.")
        print("  Get your token at: https://huggingface.co/settings/tokens")
        print("  Accept terms at: https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        hf_token = getpass.getpass("  Enter HuggingFace token (hf_...): ")
    
    if not hf_token:
        print("  ERROR: No token provided")
        return False, {'error': 'no token'}
    
    # Download parquet shards
    base_url = "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/main/data"
    total_shards = 14
    
    print(f"  Downloading {total_shards} parquet shards...")
    
    for i in range(total_shards):
        shard_num = f"{i:05d}"
        filename = f"validation-{shard_num}-of-00014.parquet"
        filepath = parquet_dir / filename
        
        if filepath.exists():
            print(f"  ✓ Shard {i+1}/{total_shards} exists, skipping")
            continue
        
        url = f"{base_url}/{filename}"
        print(f"  Downloading shard {i+1}/{total_shards}: {filename}")
        
        # Use wget with auth header
        import subprocess
        result = subprocess.run([
            'wget', '-q', '--show-progress',
            '--header', f'Authorization: Bearer {hf_token}',
            url, '-O', str(filepath)
        ], capture_output=False)
        
        if result.returncode != 0:
            print(f"  ERROR: Failed to download {filename}")
            filepath.unlink(missing_ok=True)
            return False, {'error': f'download failed: {filename}'}
    
    # Extract images from parquet files
    print(f"\n  Extracting images from parquet files...")
    
    parquet_files = sorted(parquet_dir.glob('validation-*.parquet'))
    count = 0
    
    for pq_file in parquet_files:
        if count >= num_images:
            break
        
        print(f"  Processing {pq_file.name}...")
        table = pq.read_table(pq_file)
        
        for i in range(len(table)):
            if count >= num_images:
                break
            
            row = table.slice(i, 1).to_pydict()
            label = row['label'][0]
            img_bytes = row['image'][0]['bytes']
            
            label_dir = val_dir / str(label)
            label_dir.mkdir(parents=True, exist_ok=True)
            
            img = Image.open(io.BytesIO(img_bytes))
            img_path = label_dir / f'img_{count}.JPEG'
            img.save(img_path, 'JPEG')
            
            count += 1
            if count % 1000 == 0:
                print(f"  Extracted {count}/{num_images} images...")
    
    print(f"  ✓ Extracted {count} images to {val_dir}")
    
    return True, {
        'downloaded': count,
        'method': 'wget+parquet',
        'val_dir': str(val_dir),
    }


def download_openimages(data_dir: Path, method: str = 'auto',
                        num_images: int = 24781) -> Tuple[bool, Dict]:
    """
    Download OpenImages V6 MLPerf validation subset.
    
    MLPerf uses 264 object classes from OpenImages for RetinaNet.
    The full validation set has ~24,781 images with these classes.
    
    Args:
        data_dir: Destination directory
        method: Download method (auto, wget, curl)
        num_images: Number of images to download (24781 for full MLPerf set)
    
    Returns:
        (success, info_dict)
    """
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    
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
    
    images_dir = data_dir / "validation" / "data"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    existing_images = list(images_dir.glob("*.jpg"))
    if len(existing_images) >= num_images:
        print(f"  ✓ OpenImages already downloaded: {len(existing_images)} images")
        return True, {'downloaded': len(existing_images), 'skipped': True}
    
    print(f"\n  Downloading OpenImages V6 MLPerf subset ({num_images} images)...")
    print("  264 MLPerf object classes")
    print("-" * 50)
    
    # Download class descriptions
    class_map_url = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
    class_map_file = data_dir / "class-descriptions-boxable.csv"
    
    if not class_map_file.exists():
        print("  Downloading class descriptions...")
        urllib.request.urlretrieve(class_map_url, class_map_file)
    
    # Download validation annotations
    ann_url = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
    ann_file = data_dir / "validation-annotations-bbox.csv"
    
    if not ann_file.exists():
        print("  Downloading validation annotations (~50MB)...")
        urllib.request.urlretrieve(ann_url, ann_file)
    
    # Load class mapping
    print("  Loading class mappings...")
    class_map = {}
    with open(class_map_file) as f:
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                class_map[parts[0]] = parts[1]
    
    # Find MLPerf class codes
    mlperf_class_codes = {}
    for code, name in class_map.items():
        if name in MLPERF_CLASSES:
            mlperf_class_codes[code] = name
    
    print(f"  Found {len(mlperf_class_codes)} MLPerf classes in OpenImages")
    
    # Load annotations and filter by MLPerf classes
    print("  Loading annotations...")
    try:
        import pandas as pd
        ann_df = pd.read_csv(ann_file)
        ann_df = ann_df[ann_df['LabelName'].isin(mlperf_class_codes.keys())]
        unique_images = ann_df['ImageID'].unique()
    except ImportError:
        # Fallback without pandas
        print("  (pandas not available, using basic CSV parsing)")
        unique_images = set()
        with open(ann_file) as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(',')
                if len(parts) > 2 and parts[2] in mlperf_class_codes:
                    unique_images.add(parts[0])
        unique_images = list(unique_images)
    
    print(f"  Found {len(unique_images)} images with MLPerf class annotations")
    
    # Limit to requested number
    selected_images = unique_images[:num_images]
    print(f"  Selecting {len(selected_images)} images for download")
    
    # Download function
    def download_image(image_id):
        output_path = images_dir / f"{image_id}.jpg"
        if output_path.exists():
            return str(output_path)
        
        # Try direct URL first
        url = f"https://storage.googleapis.com/openimages/validation/{image_id}.jpg"
        try:
            urllib.request.urlretrieve(url, output_path)
            return str(output_path)
        except Exception:
            pass
        
        # Try S3 if available
        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config
            s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
            s3.download_file('open-images-dataset', f'validation/{image_id}.jpg', str(output_path))
            return str(output_path)
        except Exception:
            return None
    
    # Download images in parallel
    print(f"  Downloading {len(selected_images)} images from OpenImages...")
    successful = 0
    failed = 0
    
    try:
        from tqdm import tqdm
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(tqdm(executor.map(download_image, selected_images),
                               total=len(selected_images), desc="  Downloading"))
    except ImportError:
        # Fallback without tqdm
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = []
            for i, result in enumerate(executor.map(download_image, selected_images)):
                results.append(result)
                if (i + 1) % 100 == 0:
                    print(f"    Downloaded {i + 1}/{len(selected_images)} images...")
    
    successful = sum(1 for r in results if r is not None)
    failed = sum(1 for r in results if r is None)
    
    print(f"  Downloaded: {successful}, Failed: {failed}")
    
    # Save filtered annotations for MLPerf classes
    mlperf_ann_file = data_dir / "mlperf-annotations.csv"
    if not mlperf_ann_file.exists():
        try:
            import pandas as pd
            ann_df = pd.read_csv(ann_file)
            mlperf_df = ann_df[ann_df['LabelName'].isin(mlperf_class_codes.keys())]
            mlperf_df = mlperf_df[mlperf_df['ImageID'].isin(selected_images)]
            mlperf_df.to_csv(mlperf_ann_file, index=False)
            print(f"  Saved MLPerf annotations: {mlperf_ann_file}")
        except:
            pass
    
    return successful > 0, {
        'downloaded': successful,
        'failed': failed,
        'total_images': len(selected_images),
        'method': 'openimages_v6',
        'images_dir': str(images_dir),
    }


def download_kits19(data_dir: Path, method: str = 'auto', 
                    num_cases: int = 43) -> Tuple[bool, Dict]:
    """
    Download KiTS19 dataset cases using the official kits19 repository.
    
    Clones the official kits19 repo and uses their get_imaging.py downloader.
    Then preprocesses NIfTI to numpy arrays for benchmarking.
    """
    import urllib.request
    
    KITS19_GIT_URL = "https://github.com/neheller/kits19.git"
    
    repo_path = data_dir / "kits19_repo"
    raw_path = data_dir / "raw"
    
    print(f"\n  Downloading KiTS19 cases (0 to {num_cases-1})...")
    print(f"  Using official kits19 repository downloader")
    print("-" * 50)
    
    downloaded = 0
    failed = []
    
    try:
        # Step 1: Clone or update kits19 repo
        if not repo_path.exists():
            print("  Cloning kits19 repository...", end=" ", flush=True)
            result = subprocess.run(
                ["git", "clone", "--depth", "1", KITS19_GIT_URL, str(repo_path)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"FAILED: {result.stderr}")
                return False, {'error': 'git clone failed'}
            print("OK")
        else:
            print("  Repository already exists")
        
        # Step 2: Install requirements
        req_file = repo_path / "requirements.txt"
        if req_file.exists():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
                capture_output=True
            )
        
        # Step 3: Download cases using official get_imaging.py
        downloader = repo_path / "starter_code" / "get_imaging.py"
        
        for case_num in range(num_cases):
            case_name = f"case_{case_num:05d}"
            dest_case = raw_path / case_name
            src_case = repo_path / "data" / case_name
            imaging_file = dest_case / "imaging.nii.gz"
            seg_file = dest_case / "segmentation.nii.gz"
            
            # Skip if already exists
            if imaging_file.exists() and seg_file.exists():
                print(f"  {case_name}: Already exists, skipping")
                downloaded += 1
                continue
            
            print(f"  {case_name}: Downloading...", end=" ", flush=True)
            
            if downloader.exists():
                # Use official downloader
                result = subprocess.run(
                    [sys.executable, str(downloader), str(case_num)],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0 and src_case.exists():
                    # Copy from repo to our raw folder
                    raw_path.mkdir(parents=True, exist_ok=True)
                    if dest_case.exists():
                        shutil.rmtree(dest_case)
                    shutil.copytree(src_case, dest_case)
                    print("OK")
                    downloaded += 1
                    continue
            
            # Fallback: try direct download with alternate URL format
            print("trying alternate...", end=" ", flush=True)
            success = _download_kits19_case_direct(case_num, raw_path)
            if success:
                print("OK")
                downloaded += 1
            else:
                print("FAILED")
                failed.append(case_name)
        
    except Exception as e:
        print(f"  Error: {e}")
        return False, {'error': str(e)}
    
    # Preprocess to numpy arrays if any downloaded
    if downloaded > 0:
        print("\n  Preprocessing NIfTI files to numpy arrays...")
        preprocess_success = preprocess_kits19(data_dir, target_shape=(128, 128, 128))
        if not preprocess_success:
            print("  ⚠️  Preprocessing failed - benchmark will use synthetic data")
    
    info = {
        'downloaded': downloaded,
        'failed': failed,
        'total_cases': num_cases,
    }
    
    return len(failed) == 0 or downloaded > 0, info


def _download_kits19_case_direct(case_num: int, raw_path: Path) -> bool:
    """
    Download a single KiTS19 case directly (fallback method).
    
    Tries multiple URL formats since the hosting may change.
    """
    import urllib.request
    
    case_name = f"case_{case_num:05d}"
    case_path = raw_path / case_name
    case_path.mkdir(parents=True, exist_ok=True)
    
    # Try different URL formats
    url_formats = [
        # Format 1: master_XXXXX naming
        ("https://kits19.sfo2.digitaloceanspaces.com/master_{:05d}.nii.gz".format(case_num),
         "https://kits19.sfo2.digitaloceanspaces.com/master_{:05d}_seg.nii.gz".format(case_num)),
        # Format 2: case_XXXXX folder naming  
        ("https://kits19.sfo2.digitaloceanspaces.com/{}/imaging.nii.gz".format(case_name),
         "https://kits19.sfo2.digitaloceanspaces.com/{}/segmentation.nii.gz".format(case_name)),
    ]
    
    for img_url, seg_url in url_formats:
        try:
            img_dest = case_path / "imaging.nii.gz"
            seg_dest = case_path / "segmentation.nii.gz"
            
            if not img_dest.exists():
                urllib.request.urlretrieve(img_url, img_dest)
            if not seg_dest.exists():
                urllib.request.urlretrieve(seg_url, seg_dest)
            
            # Verify files are valid (not XML errors)
            with open(img_dest, 'rb') as f:
                header = f.read(2)
                if header != b'\x1f\x8b':  # gzip magic number
                    img_dest.unlink()
                    continue
            return True
        except Exception:
            continue
    
    return False


def preprocess_kits19(data_dir: Path, target_shape: Tuple[int, int, int] = (128, 128, 128)) -> bool:
    """
    Preprocess raw KiTS19 NIfTI cases into numpy arrays for benchmarking.
    
    Args:
        data_dir: Path to KiTS19 data directory (contains 'raw' subfolder)
        target_shape: Target volume shape (D, H, W)
    
    Returns:
        True if successful
    """
    try:
        import nibabel as nib
        from scipy import ndimage
    except ImportError:
        print("  Error: nibabel and scipy required for preprocessing")
        print("  Install with: pip install nibabel scipy")
        return False
    
    raw_path = data_dir / "raw"
    if not raw_path.exists():
        print(f"  Error: Raw data not found at {raw_path}")
        return False
    
    # Find all cases
    cases = sorted([d for d in raw_path.iterdir() if d.is_dir() and d.name.startswith("case_")])
    if not cases:
        print("  Error: No cases found to preprocess")
        return False
    
    print(f"  Processing {len(cases)} cases to shape {target_shape}...")
    
    volumes = []
    labels = []
    
    for i, case_dir in enumerate(cases):
        imaging_file = case_dir / "imaging.nii.gz"
        seg_file = case_dir / "segmentation.nii.gz"
        
        if not imaging_file.exists() or not seg_file.exists():
            print(f"  {case_dir.name}: Missing files, skipping")
            continue
        
        try:
            # Load NIfTI files
            img_nii = nib.load(str(imaging_file))
            seg_nii = nib.load(str(seg_file))
            
            img_data = img_nii.get_fdata()
            seg_data = seg_nii.get_fdata()
            
            # Calculate zoom factors
            current_shape = img_data.shape
            zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]
            
            # Resample
            img_resampled = ndimage.zoom(img_data, zoom_factors, order=1)  # Linear for image
            seg_resampled = ndimage.zoom(seg_data, zoom_factors, order=0)  # Nearest for labels
            
            # Normalize image to [0, 1]
            img_min, img_max = img_resampled.min(), img_resampled.max()
            if img_max > img_min:
                img_resampled = (img_resampled - img_min) / (img_max - img_min)
            
            # Add channel dimension
            img_resampled = img_resampled[np.newaxis, ...]
            
            volumes.append(img_resampled.astype(np.float32))
            labels.append(seg_resampled.astype(np.int64))
            
            if (i + 1) % 10 == 0 or i == len(cases) - 1:
                print(f"  Processed {i + 1}/{len(cases)} cases")
                
        except Exception as e:
            print(f"  {case_dir.name}: Error - {e}")
            continue
    
    if not volumes:
        print("  Error: No volumes successfully processed")
        return False
    
    # Stack and save
    volumes_array = np.stack(volumes, axis=0)
    labels_array = np.stack(labels, axis=0)
    
    volumes_path = data_dir / "volumes.npy"
    labels_path = data_dir / "labels.npy"
    
    np.save(volumes_path, volumes_array)
    np.save(labels_path, labels_array)
    
    print(f"  ✓ Saved {len(volumes)} volumes to {volumes_path}")
    print(f"    Shape: {volumes_array.shape}")
    
    return True


def download_cnn_dailymail(data_dir: Path, method: str = 'auto') -> Tuple[bool, Dict]:
    """
    Download CNN-DailyMail dataset using HuggingFace datasets.
    
    This is the official MLPerf dataset for text summarization (GPT-J).
    Test set contains ~11,490 articles with summaries.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = data_dir / "test.json"
    
    # Check if already downloaded
    if test_file.exists():
        print(f"  Dataset already exists: {test_file}")
        with open(test_file) as f:
            data = json.load(f)
        return True, {'samples': len(data), 'existing': True}
    
    try:
        from datasets import load_dataset
        print("  Downloading CNN-DailyMail from HuggingFace...")
        print("    - News articles with summaries")
        print("    - Test set: ~11,490 articles")
        
        # Load the dataset
        dataset = load_dataset("cnn_dailymail", "3.0.0", trust_remote_code=True)
        
        # Save test set
        test_data = []
        for item in dataset["test"]:
            test_data.append({
                "article": item["article"],
                "highlights": item["highlights"],
                "id": item["id"]
            })
        
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        print(f"  ✓ Saved {len(test_data)} test articles to {test_file}")
        
        # Also save a small validation subset
        val_data = []
        for item in list(dataset["validation"])[:1000]:
            val_data.append({
                "article": item["article"],
                "highlights": item["highlights"],
                "id": item["id"]
            })
        
        val_file = data_dir / "validation.json"
        with open(val_file, 'w') as f:
            json.dump(val_data, f)
        
        print(f"  ✓ Saved {len(val_data)} validation articles to {val_file}")
        
        # Create metadata
        metadata = {
            'type': 'mlperf',
            'samples': len(test_data),
            'validation_samples': len(val_data),
            'mlperf_compliant': True,
            'format': 'cnn-dailymail',
            'source': 'huggingface/cnn_dailymail',
            'version': '3.0.0',
        }
        metadata_file = data_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True, {'samples': len(test_data)}
        
    except ImportError:
        print("  Error: 'datasets' library not installed")
        print("  Install with: pip install datasets")
        return False, {'error': 'datasets library not installed'}
    
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False, {'error': str(e)}


def download_mixtral_data(data_dir: Path, method: str = 'auto') -> Tuple[bool, Dict]:
    """Download Mixtral combined dataset (OpenOrca + GSM8K + MBXP).
    
    MLPerf Mixtral-8x7B uses official preprocessed pkl file with 15K samples.
    Downloads from MLCommons storage and converts to JSON.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("  Downloading MLPerf Mixtral dataset...")
    
    # Official MLPerf Mixtral dataset URL
    pkl_url = "https://inference.mlcommons-storage.org/mixtral_8x7b/09292024_mixtral_15k_mintoken2_v1.pkl"
    pkl_file = data_dir / "09292024_mixtral_15k_mintoken2_v1.pkl"
    
    # Download pkl file
    if not pkl_file.exists():
        print(f"  Downloading from MLCommons storage (~100MB)...")
        success, result = download_file(pkl_url, pkl_file, method)
        if not success:
            print(f"  Error downloading pkl: {result.get('error', 'Unknown')}")
            return False, result
    else:
        print(f"  Using cached pkl file: {pkl_file}")
    
    # Convert pkl to JSON
    print("  Processing pkl file...")
    try:
        import pickle
        
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"    Dataset type: {type(data)}")
        
        test_data = []
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    test_data.append({
                        "id": str(len(test_data)),
                        "question": item.get("input", item.get("question", item.get("prompt", ""))),
                        "response": item.get("output", item.get("response", item.get("target", ""))),
                        "source": item.get("source", "unknown"),
                    })
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    test_data.append({
                        "id": str(len(test_data)),
                        "question": str(item[0]),
                        "response": str(item[1]),
                        "source": "combined",
                    })
        elif isinstance(data, dict):
            # Handle dict format (input_ids, prompts, etc.)
            if 'input_ids' in data or 'prompts' in data:
                prompts = data.get('prompts', data.get('input', []))
                responses = data.get('responses', data.get('output', []))
                sources = data.get('sources', ['unknown'] * len(prompts))
                for i, (p, r) in enumerate(zip(prompts, responses)):
                    test_data.append({
                        "id": str(i),
                        "question": str(p),
                        "response": str(r),
                        "source": sources[i] if i < len(sources) else "unknown",
                    })
            else:
                for k, v in data.items():
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                test_data.append({
                                    "id": str(len(test_data)),
                                    "question": item.get("input", item.get("question", "")),
                                    "response": item.get("output", item.get("response", "")),
                                    "source": k,
                                })
        
        # Fallback to pandas if parsing failed
        if not test_data:
            print("    Trying pandas fallback...")
            try:
                import pandas as pd
                df = pd.read_pickle(pkl_file)
                print(f"    DataFrame shape: {df.shape}, columns: {list(df.columns)}")
                for idx, row in df.iterrows():
                    test_data.append({
                        "id": str(idx),
                        "question": str(row.get('input', row.get('prompt', row.iloc[0] if len(row) > 0 else ''))),
                        "response": str(row.get('output', row.get('target', row.iloc[1] if len(row) > 1 else ''))),
                        "source": str(row.get('source', 'combined')),
                    })
            except Exception as e:
                print(f"    Pandas fallback failed: {e}")
        
        if not test_data:
            print("  Error: Could not parse pkl file")
            return False, {'error': 'Could not parse pkl file'}
        
        # Save test.json
        test_file = data_dir / "test.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"  Saved {len(test_data)} samples to {test_file}")
        
        # Count categories
        categories = {}
        for item in test_data:
            cat = item.get('source', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        metadata = {
            'type': 'mlperf',
            'dataset': 'Mixtral-Combined',
            'samples': len(test_data),
            'categories': categories,
            'mlperf_compliant': len(test_data) >= 15000,
            'task': 'text-generation',
            'format': 'mixtral',
            'source': 'mlcommons-storage',
        }
        
        metadata_file = data_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True, metadata
        
    except Exception as e:
        print(f"  Error processing pkl: {e}")
        import traceback
        traceback.print_exc()
        return False, {'error': str(e)}


def download_llama_data(data_dir: Path, method: str = 'auto') -> Tuple[bool, Dict]:
    """Download OpenOrca data for Llama benchmarks.
    
    MLPerf Llama2-70B uses OpenOrca dataset (GPT-4 filtered subset).
    Downloads from HuggingFace and processes into test.json format.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("  Downloading OpenOrca dataset for Llama...")
    
    try:
        from datasets import load_dataset
        
        # Download OpenOrca GPT-4 subset (1M entries, but we'll sample)
        print("  Loading OpenOrca from HuggingFace...")
        dataset = load_dataset("Open-Orca/OpenOrca", split="train")
        
        # Filter to GPT-4 responses for quality (like MLPerf does)
        print("  Filtering GPT-4 responses...")
        gpt4_data = dataset.filter(lambda x: 'gpt4' in x.get('id', '').lower() or 
                                              'gpt-4' in x.get('id', '').lower())
        
        if len(gpt4_data) < 100:
            # Fallback to all data if GPT-4 filter too strict
            print(f"  Only {len(gpt4_data)} GPT-4 samples found, using full dataset")
            gpt4_data = dataset
        
        # Sample to reasonable size (MLPerf uses 24576)
        max_samples = min(25000, len(gpt4_data))
        
        # Convert to our format
        print(f"  Processing {max_samples} samples...")
        test_data = []
        
        for i, item in enumerate(gpt4_data):
            if i >= max_samples:
                break
            
            # OpenOrca format: system_prompt, question, response
            system = item.get('system_prompt', '')
            question = item.get('question', '')
            response = item.get('response', '')
            
            # Format as Llama prompt
            if system:
                full_prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{question} [/INST]"
            else:
                full_prompt = f"<s>[INST] {question} [/INST]"
            
            test_data.append({
                'id': str(i),
                'question': full_prompt,
                'response': response,
                'system_prompt': system,
                'original_question': question,
            })
        
        # Save test.json
        test_file = data_dir / "test.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"  Saved {len(test_data)} samples to {test_file}")
        
        # Create metadata
        metadata = {
            'type': 'mlperf',
            'dataset': 'OpenOrca',
            'samples': len(test_data),
            'mlperf_compliant': len(test_data) >= 24576,
            'task': 'text-generation',
            'format': 'openorca',
        }
        
        metadata_file = data_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True, metadata
        
    except ImportError:
        print("  Error: 'datasets' package required. Install with: pip install datasets")
        return False, {'error': 'datasets package not installed'}
    except Exception as e:
        print(f"  Error downloading OpenOrca: {e}")
        return False, {'error': str(e)}


def download_criteo(data_dir: Path, method: str = 'auto') -> Tuple[bool, Dict]:
    """Download preprocessed Criteo dataset for DLRM benchmarks.
    
    MLPerf DLRM-v2 uses preprocessed Criteo Terabyte dataset from MLCommons storage.
    This is the official ~100GB preprocessed dataset (Day 23 validation data).
    
    For smaller testing, use data_gen.py to create synthetic data.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("  Downloading MLPerf DLRM preprocessed dataset...")
    print("  Source: MLCommons inference storage")
    print("  Size: ~100GB (preprocessed Criteo Day 23)")
    print("")
    
    # MLCommons R2 storage URL for preprocessed dataset
    dataset_uri = "https://inference.mlcommons-storage.org/metadata/dlrm-v2-preprocessed-dataset.uri"
    
    # Check disk space
    try:
        import shutil
        free_gb = shutil.disk_usage(data_dir).free / (1024**3)
        if free_gb < 120:
            print(f"  Warning: Only {free_gb:.1f}GB free, need ~120GB for dataset")
            print("  Consider using synthetic data (data_gen.py) for testing")
    except Exception:
        pass
    
    # Download using MLCommons R2 downloader (bash script)
    print("  Using MLCommons R2 downloader...")
    print(f"  Target directory: {data_dir}")
    print("")
    
    import subprocess
    import os
    
    # Change to data directory and run the downloader
    original_dir = os.getcwd()
    os.chdir(data_dir)
    
    try:
        # The MLCommons R2 downloader script
        cmd = f'bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) "{dataset_uri}"'
        
        print(f"  Running: {cmd}")
        print("  This will take a while (~100GB download)...")
        print("")
        
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=False,  # Show progress
        )
        
        os.chdir(original_dir)
        
        if result.returncode != 0:
            print(f"  Download failed with exit code: {result.returncode}")
            return False, {'error': f'Download failed with exit code {result.returncode}'}
        
        # MLCommons downloader creates a subdirectory - find and move files
        subdir = data_dir / 'dlrm_preprocessed'
        if subdir.exists() and subdir.is_dir():
            print(f"  Moving files from {subdir} to {data_dir}...")
            import shutil
            for f in subdir.iterdir():
                dest = data_dir / f.name
                if not dest.exists():
                    shutil.move(str(f), str(dest))
                    print(f"    Moved: {f.name}")
            # Remove empty subdirectory
            try:
                subdir.rmdir()
            except OSError:
                pass  # Not empty, leave it
        
        # Verify download by checking for expected files
        found_files = list(data_dir.glob('*.npy')) + list(data_dir.glob('*.npz')) + list(data_dir.glob('*.bin'))
        
        if not found_files:
            print("  Warning: No data files found after download")
            print("  The download may have failed or files are in a subdirectory")
            # Check subdirectories
            for subdir in data_dir.iterdir():
                if subdir.is_dir():
                    sub_files = list(subdir.glob('*'))
                    if sub_files:
                        print(f"  Found files in {subdir}: {[f.name for f in sub_files[:5]]}")
                        found_files = sub_files
        
        # Create metadata
        metadata = {
            'type': 'mlperf',
            'dataset': 'criteo-preprocessed',
            'source': 'mlcommons-storage',
            'task': 'recommendation',
            'format': 'preprocessed',
            'mlperf_compliant': True,
            'files': [f.name for f in found_files],
        }
        
        # Count samples from the dense file (smaller, faster to load than sparse)
        try:
            import numpy as np
            dense_file = data_dir / 'day_23_dense.npy'
            if dense_file.exists():
                dense = np.load(dense_file)
                metadata['samples'] = len(dense)
                print(f"  Dataset contains {metadata['samples']:,} samples")
            else:
                # Fallback to sparse file if dense doesn't exist
                sparse_file = data_dir / 'day_23_sparse_multi_hot.npz'
                if sparse_file.exists():
                    print("  Counting samples from sparse file (this may take a while)...")
                    with np.load(sparse_file, allow_pickle=True) as data:
                        keys = list(data.keys())
                        if keys:
                            arr = data[keys[0]]
                            metadata['samples'] = arr.shape[0] if hasattr(arr, 'shape') else len(arr)
                            print(f"  Dataset contains {metadata['samples']:,} samples")
        except Exception as e:
            print(f"  Could not count samples: {e}")
        
        metadata_file = data_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Download complete")
        return True, metadata
        
    except Exception as e:
        os.chdir(original_dir)
        print(f"  Error during download: {e}")
        import traceback
        traceback.print_exc()
        return False, {'error': str(e)}


def create_librispeech_manifest(data_dir: Path) -> bool:
    """Create manifest file for LibriSpeech after extraction."""
    
    manifest = []
    
    for subset in ['dev-clean', 'dev-other']:
        subset_dir = data_dir / 'LibriSpeech' / subset
        if not subset_dir.exists():
            print(f"  Subset not found: {subset_dir}")
            continue
        
        # Find all .flac files
        for flac_file in subset_dir.rglob('*.flac'):
            # LibriSpeech structure: speaker_id/chapter_id/speaker_id-chapter_id-utterance_id.flac
            # Transcript file: speaker_id/chapter_id/speaker_id-chapter_id.trans.txt
            parts = flac_file.stem.split('-')
            if len(parts) >= 2:
                speaker_id = parts[0]
                chapter_id = parts[1]
                trans_file = flac_file.parent / f"{speaker_id}-{chapter_id}.trans.txt"
                
                if trans_file.exists():
                    with open(trans_file) as f:
                        for line in f:
                            line_parts = line.strip().split(' ', 1)
                            if len(line_parts) == 2 and line_parts[0] == flac_file.stem:
                                manifest.append({
                                    'audio_path': str(flac_file),
                                    'text': line_parts[1],
                                    'duration': None,
                                })
                                break
    
    manifest_path = data_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  Created manifest with {len(manifest)} samples")
    return True


def extract_coco_captions(data_dir: Path) -> bool:
    """Extract COCO captions to a simple format for SDXL benchmark."""
    annotations_file = data_dir / 'annotations' / 'captions_val2014.json'
    
    if not annotations_file.exists():
        print(f"  Annotations file not found: {annotations_file}")
        return False
    
    with open(annotations_file) as f:
        data = json.load(f)
    
    # Extract unique captions (one per image for MLPerf)
    # MLPerf uses 5000 samples
    image_to_caption = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_to_caption:
            image_to_caption[img_id] = ann['caption']
        if len(image_to_caption) >= 5000:
            break
    
    captions = list(image_to_caption.values())
    
    # Save captions as simple list (compatible with run_sdxl_benchmark.py)
    captions_file = data_dir / 'captions.json'
    with open(captions_file, 'w') as f:
        json.dump(captions, f, indent=2)
    
    print(f"  Created captions.json with {len(captions)} captions")
    
    # Count images if available
    images_dir = data_dir / 'val2014'
    num_images = 0
    if images_dir.exists():
        num_images = len(list(images_dir.glob('*.jpg')))
    
    # Create metadata
    metadata = {
        'type': 'mlperf',
        'samples': len(captions),
        'images': num_images,
        'mlperf_compliant': True,
        'format': 'coco-2014',
        'source': 'COCO 2014 validation set',
        'captions_file': 'captions.json',
        'images_dir': 'val2014' if num_images > 0 else None,
        'annotations_dir': 'annotations',
        'note': f'COCO 2014 with {len(captions)} captions' + (f' and {num_images} images for FID.' if num_images > 0 else '. Images not downloaded.'),
    }
    metadata_file = data_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Created metadata.json")
    return True


# ============================================================================
# Main Download Function
# ============================================================================

def get_default_data_dir(benchmark: str) -> Path:
    """Get the default data directory for a benchmark's MLPerf data."""
    if benchmark not in DATASETS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    config = DATASETS[benchmark]
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    # New structure: data/{benchmark}/{default_dataset}/
    default_dataset = config.get('default_dataset', benchmark)
    return project_dir / 'data' / benchmark / default_dataset


def download_benchmark_data(benchmark: str, data_dir: Optional[Path] = None,
                           method: str = 'auto', force: bool = False,
                           verbose: bool = True) -> Tuple[bool, Dict]:
    """
    Download data for a specific benchmark.
    
    Args:
        benchmark: Benchmark name (bert, resnet50, etc.)
        data_dir: Override default data directory (if None, uses data/{benchmark}/{default_dataset}/)
        method: Download method (wget, curl, urllib, auto)
        force: Re-download even if exists
        verbose: Print progress
        
    Returns:
        Tuple of (success, info_dict)
    """
    if benchmark not in DATASETS:
        return False, {'error': f"Unknown benchmark: {benchmark}"}
    
    config = DATASETS[benchmark]
    
    # Determine data directory - new structure: data/{benchmark}/{dataset}/
    if data_dir is None:
        data_dir = get_default_data_dir(benchmark)
    
    data_dir = Path(data_dir)
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"DOWNLOADING: {config['name']}")
        print("=" * 60)
        print(f"  Benchmark:   {benchmark}")
        print(f"  Task:        {config['task']}")
        print(f"  Destination: {data_dir}")
        print("=" * 60)
    
    # Check for manual download requirement
    if config.get('manual_download') and not config.get('files'):
        if verbose:
            print(config.get('instructions', 'Manual download required.'))
        return True, {'manual_download': True, 'instructions': config.get('instructions')}
    
    # Check for custom download handler
    if config.get('custom_download'):
        handler_name = config['custom_download']
        handler = globals().get(handler_name)
        if handler:
            return handler(data_dir, method)
        else:
            return False, {'error': f"Custom handler not found: {handler_name}"}
    
    # Standard file downloads
    data_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    failed_files = []
    
    for file_info in config.get('files', []):
        url = file_info['url']
        dest = data_dir / file_info['dest']
        size_mb = file_info.get('size_mb', '?')
        
        # Skip if exists and not forcing
        if dest.exists() and not force:
            if verbose:
                print(f"\n  {file_info['dest']}: Already exists, skipping")
            downloaded_files.append(str(dest))
            continue
        
        if verbose:
            print(f"\n  {file_info['dest']} ({size_mb} MB)")
        
        success, result = download_file(url, dest, method, verbose=verbose)
        
        if success:
            downloaded_files.append(str(dest))
            
            # Extract if needed
            if file_info.get('extract'):
                extract_archive(dest, data_dir)
            
            # Verify checksum if provided
            if file_info.get('sha256'):
                if not verify_checksum(dest, file_info['sha256']):
                    if verbose:
                        print(f"    WARNING: Checksum mismatch!")
        else:
            failed_files.append({'file': file_info['dest'], 'error': result})
    
    # Run post-download handler if specified
    if config.get('post_download') and not failed_files:
        handler_name = config['post_download']
        handler = globals().get(handler_name)
        if handler:
            handler(data_dir)
    
    info = {
        'benchmark': benchmark,
        'data_dir': str(data_dir),
        'downloaded': downloaded_files,
        'failed': failed_files,
    }
    
    success = len(failed_files) == 0
    
    if verbose:
        print("\n" + "=" * 60)
        if success:
            print("DOWNLOAD COMPLETE")
        else:
            print("DOWNLOAD FAILED")
            for f in failed_files:
                print(f"  - {f['file']}: {f['error']}")
        print("=" * 60)
    
    return success, info


def check_benchmark_data(benchmark: str, data_dir: Optional[Path] = None) -> Dict:
    """Check the status of benchmark data."""
    if benchmark not in DATASETS:
        return {'error': f"Unknown benchmark: {benchmark}"}
    
    config = DATASETS[benchmark]
    
    if data_dir is None:
        data_dir = get_default_data_dir(benchmark)
    
    data_dir = Path(data_dir)
    
    status = {
        'benchmark': benchmark,
        'name': config['name'],
        'data_dir': str(data_dir),
        'exists': data_dir.exists(),
        'files': {},
        'ready': False,
    }
    
    if config.get('manual_download'):
        status['manual_download'] = True
    
    # Check each expected file
    for file_info in config.get('files', []):
        dest = data_dir / file_info['dest']
        file_status = {
            'exists': dest.exists(),
            'size_mb': dest.stat().st_size / (1024*1024) if dest.exists() else 0,
            'expected_mb': file_info.get('size_mb', '?'),
        }
        status['files'][file_info['dest']] = file_status
    
    # Check for metadata
    meta_file = data_dir / 'metadata.json'
    status['has_metadata'] = meta_file.exists()
    
    # Determine if ready
    if config.get('files'):
        status['ready'] = all(
            (data_dir / f['dest']).exists() 
            for f in config['files']
        )
    elif config.get('manual_download'):
        # Check for expected structure
        status['ready'] = data_dir.exists() and any(data_dir.iterdir()) if data_dir.exists() else False
    
    return status


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Download MLPerf benchmark datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --benchmark bert                    # Download SQuAD for BERT
  %(prog)s --benchmark bert --method wget      # Force wget method
  %(prog)s --benchmark all --check             # Check status of all
  %(prog)s --url URL --dest FILE               # Download arbitrary file
  
Available benchmarks:
  bert, resnet50, retinanet, 3dunet, whisper, sdxl, gptj, llama, mixtral, dlrm
"""
    )
    
    parser.add_argument('--benchmark', '-b', type=str,
                       help='Benchmark name or "all"')
    parser.add_argument('--data-dir', '-d', type=str,
                       help='Override default data directory')
    parser.add_argument('--method', '-m', type=str, default='auto',
                       choices=['auto', 'wget', 'curl', 'urllib'],
                       help='Download method (default: auto)')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Re-download even if exists')
    parser.add_argument('--check', '-c', action='store_true',
                       help='Check data status without downloading')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available benchmarks')
    
    # Direct download mode
    parser.add_argument('--url', type=str,
                       help='Direct URL to download')
    parser.add_argument('--dest', type=str,
                       help='Destination path for direct download')
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        print("\nAvailable Benchmarks:")
        print("-" * 60)
        for name, config in DATASETS.items():
            manual = " (manual)" if config.get('manual_download') else ""
            print(f"  {name:12} - {config['name']}{manual}")
        print()
        return 0
    
    # Direct download mode
    if args.url and args.dest:
        success, result = download_file(args.url, Path(args.dest), args.method)
        return 0 if success else 1
    
    # Benchmark mode
    if not args.benchmark:
        parser.print_help()
        return 1
    
    benchmarks = list(DATASETS.keys()) if args.benchmark == 'all' else [args.benchmark]
    
    # Check mode
    if args.check:
        print("\n" + "=" * 60)
        print("DATA STATUS CHECK")
        print("=" * 60)
        
        for benchmark in benchmarks:
            status = check_benchmark_data(benchmark, 
                                         Path(args.data_dir) if args.data_dir else None)
            icon = "✓" if status.get('ready') else "✗"
            manual = " (manual download)" if status.get('manual_download') else ""
            print(f"\n  {icon} {benchmark:12} - {status.get('name', 'Unknown')}{manual}")
            print(f"    Directory: {status['data_dir']}")
            print(f"    Exists:    {status['exists']}")
            if status.get('files'):
                for fname, fstatus in status['files'].items():
                    ficon = "✓" if fstatus['exists'] else "✗"
                    print(f"    {ficon} {fname}")
        
        print("\n" + "=" * 60)
        return 0
    
    # Download mode
    all_success = True
    for benchmark in benchmarks:
        data_dir = Path(args.data_dir) if args.data_dir else None
        success, info = download_benchmark_data(
            benchmark, data_dir, args.method, args.force
        )
        if not success:
            all_success = False
    
    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())
