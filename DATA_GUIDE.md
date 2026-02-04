# MLPerf Benchmark Data Guide

This guide explains data management for MLPerf benchmarks, including real datasets, synthetic data generation, and the data preparation workflow.

## Quick Summary

| Benchmark  | Real Dataset         | Size    | Synthetic Available | Auto-Download |
|------------|---------------------|---------|:-------------------:|:-------------:|
| BERT       | SQuAD v1.1          | 4.7 MB  | ✓                   | ✅            |
| ResNet50   | ImageNet            | 140 GB  | ✓                   | ❌ (license)  |
| RetinaNet  | OpenImages          | ~2 GB   | ✓                   | ✅            |
| 3D-UNet    | KiTS19              | 24 GB   | ✓                   | ✅            |
| DLRM       | Criteo Terabyte     | ~150 GB | ✓                   | ✅            |
| GPT-J      | CNN-DailyMail       | ~300 MB | ✓                   | ✅            |
| Llama      | OpenOrca            | ~4 GB   | ✓                   | ✅            |
| Mixtral    | OpenOrca / GSM8K    | ~4 GB   | ✓                   | ✅            |
| SDXL       | COCO 2014 Captions  | ~250 MB | ✓                   | ✅            |
| Whisper    | LibriSpeech         | 350 MB  | ✓                   | ✅            |

---

## Data Management Scripts

The v5.1 scripts provide three data management tools:

```
v5.1/scripts/
├── data_prepare.py    # Orchestrator - auto-selects real or synthetic
├── data_download.py   # Downloads real MLPerf datasets
└── data_gen.py        # Generates synthetic test data
```

### data_prepare.py (Recommended)

Central orchestrator that automatically prepares the right data:

```bash
cd v5.1/scripts

# Auto mode: tries real data, falls back to synthetic
python data_prepare.py --benchmark bert

# Force real data download
python data_prepare.py --benchmark bert --real

# Generate synthetic data with custom name
python data_prepare.py --benchmark bert --synthetic --name my-test-data

# Check status of all benchmarks
python data_prepare.py --check

# Force re-preparation
python data_prepare.py --benchmark whisper --real --force
```

### data_download.py

Downloads real MLPerf-compliant datasets:

```bash
# Download specific benchmark data
python data_download.py bert
python data_download.py whisper
python data_download.py llama

# Check what's available
python data_download.py --list

# Force re-download
python data_download.py bert --force
```

### data_gen.py

Generates synthetic data for quick testing:

```bash
# Generate synthetic data
python data_gen.py bert --num-samples 500
python data_gen.py whisper --num-samples 100
python data_gen.py mixtral --num-samples 20

# List available generators
python data_gen.py --list
```

---

## Using Data with benchmark.py

The unified `benchmark.py` works with both real and synthetic data:

```bash
cd v5.1/scripts

# List available datasets for each benchmark
python benchmark.py --list

# Run with real data (auto-downloaded if needed)
python benchmark.py -b bert --dataset squad -n 100 --mlperf

# Run with synthetic data
python benchmark.py -b bert --dataset synthetic -n 50

# Run with specific synthetic dataset
python benchmark.py -b bert --dataset my-custom-data -n 100
```

---

## Per-Benchmark Details

### BERT (Question Answering)

```bash
# Download real data
python data_download.py bert

# Or generate synthetic
python data_gen.py bert --num-samples 1000

# Run benchmark
python benchmark.py -b bert --dataset squad -n 100 --mlperf
```

- **Dataset**: SQuAD v1.1 validation set
- **Size**: 4.7 MB
- **Samples**: 10,833 question-answer pairs
- **Path**: `v5.1/data/squad/`

---

### ResNet50 (Image Classification)

```bash
# Generate synthetic (ImageNet requires manual download)
python data_gen.py resnet50 --num-samples 1000

# Run with synthetic
python benchmark.py -b resnet50 --dataset synthetic -n 100
```

- **Dataset**: ImageNet ILSVRC2012 validation set
- **Size**: ~140 GB
- **Samples**: 50,000 images
- **Auto-Download**: ❌ Requires manual registration at https://image-net.org
- **Path**: `v5.1/data/imagenet/`

---

### RetinaNet (Object Detection)

```bash
# Download real data
python data_download.py retinanet

# Run benchmark
python benchmark.py -b retinanet --dataset openimages -n 100 --mlperf
```

- **Dataset**: OpenImages validation set
- **Size**: ~2 GB (sample), ~5 GB (full)
- **Samples**: 5,000-41,620 images
- **Path**: `v5.1/data/openimages/`

---

### 3D-UNet (Medical Segmentation)

```bash
# Download real data
python data_download.py 3dunet

# Run benchmark
python benchmark.py -b 3dunet --dataset kits19 -n 20 --mlperf
```

- **Dataset**: KiTS19 (Kidney Tumor Segmentation Challenge)
- **Size**: 24 GB preprocessed
- **Samples**: 210 3D CT volumes with segmentation masks
- **Path**: `v5.1/data/kits19/`

---

### DLRM (Recommendation)

```bash
# Download real Criteo data (large!)
python data_download.py dlrm

# Run with sample size
python benchmark.py -b dlrm --dataset criteo -n 1000 --mlperf
```

- **Dataset**: Criteo Terabyte Click Logs (Day 23)
- **Size**: ~150 GB preprocessed
- **Samples**: Millions of ad click records
- **Path**: `v5.1/data/criteo/`

---

### GPT-J (Text Summarization)

```bash
# Download real data
python data_download.py gptj

# Run benchmark
python benchmark.py -b gptj --dataset cnn-dailymail -n 10 --4bit --mlperf
```

- **Dataset**: CNN-DailyMail test set
- **Size**: ~300 MB
- **Samples**: 11,490 news articles
- **Path**: `v5.1/data/cnn-dailymail/`

---

### Llama (Text Generation)

```bash
# Download real data
python data_download.py llama

# Run benchmark
python benchmark.py -b llama --dataset openorca -n 10 --4bit --mlperf
```

- **Dataset**: OpenOrca
- **Size**: ~4 GB
- **Samples**: 24,576 instruction-response pairs
- **Path**: `v5.1/data/openorca/`

---

### Mixtral (Text Generation - MoE)

```bash
# Download real data
python data_download.py mixtral

# Generate synthetic for quick testing
python data_gen.py mixtral --num-samples 10

# Run benchmark (requires --offload on <48GB VRAM)
python benchmark.py -b mixtral --dataset mixtral-15k --mlperf-quick --offload
```

- **Dataset**: GSM8K + OpenOrca (MLPerf reference dataset)
- **Size**: ~4 GB
- **Samples**: 15,000 samples
- **MLPerf Settings**: min_new_tokens=2, max_new_tokens=1024
- **Path**: `v5.1/data/mixtral/`

---

### SDXL (Image Generation)

```bash
# Download real data
python data_download.py sdxl

# Run benchmark
python benchmark.py -b sdxl --dataset coco-2014 -n 5 --offload --mlperf
```

- **Dataset**: COCO 2014 validation captions
- **Size**: ~250 MB
- **Samples**: 40,504 image descriptions
- **Path**: `v5.1/data/coco-2014/`

---

### Whisper (Speech Recognition)

```bash
# Download real data
python data_download.py whisper

# Run benchmark
python benchmark.py -b whisper --dataset librispeech -n 50 --mlperf
```

- **Dataset**: LibriSpeech test-clean
- **Size**: ~350 MB
- **Samples**: 2,620 audio clips (read English speech)
- **Path**: `v5.1/data/librispeech/`

---

## Metadata Files

Each prepared dataset includes a `metadata.json` file that benchmark.py uses to auto-detect data type:

```json
{
  "name": "SQuAD v1.1",
  "benchmark": "bert",
  "type": "mlperf",
  "task": "question-answering",
  "samples": 10833,
  "mlperf_compliant": true,
  "created": "2026-01-29T15:30:00",
  "generator": "data_prepare.py"
}
```

---

## Storage Requirements

| Level | Storage Needed | Description |
|-------|----------------|-------------|
| Minimal | ~15 GB | Models only, synthetic data |
| Standard | ~30 GB | Common datasets (BERT, Whisper, SDXL) |
| Full | ~60 GB | All real data except Criteo/KiTS19 |
| Complete | ~250 GB | All datasets including Criteo/KiTS19 |

---

## Data Directory Structure

```
v5.1/data/
├── squad/              # BERT (4.7 MB)
│   ├── dev-v1.1.json
│   └── metadata.json
├── imagenet/           # ResNet50 (manual download)
├── openimages/         # RetinaNet (~2 GB)
├── kits19/             # 3D-UNet (24 GB)
├── criteo/             # DLRM (~150 GB)
├── cnn-dailymail/      # GPT-J (~300 MB)
├── openorca/           # Llama (~4 GB)
├── mixtral/            # Mixtral (~4 GB)
│   └── mixtral-15k/    # MLPerf reference dataset
├── coco-2014/          # SDXL (~250 MB)
└── librispeech/        # Whisper (350 MB)
```

---

## HuggingFace Authentication

Some datasets and models require HuggingFace authentication:

```bash
# Set token as environment variable
export HF_TOKEN=your_token_here

# Or login via CLI
pip install huggingface_hub
huggingface-cli login
```

**Required for:**
- Llama models (Meta approval needed)
- ImageNet dataset (if using HuggingFace source)

---

## Troubleshooting

### Download Fails

```bash
# Check available download methods
python data_download.py --list

# Try different method
python data_download.py whisper --method wget
python data_download.py whisper --method curl
```

### Not Enough Space

```bash
# Use synthetic data instead
python data_gen.py bert --num-samples 100
python benchmark.py -b bert --dataset synthetic -n 100
```

### Data Not Found

```bash
# Check current status
python data_prepare.py --check

# Re-prepare data
python data_prepare.py --benchmark bert --force
```

---

*Last updated: February 4, 2026*
*Author: Mehdi Nik*
