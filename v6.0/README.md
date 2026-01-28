# MLPerf Inference v6.0 Setup

This folder contains MLPerf Inference v6.0 benchmark setup and runner scripts.

## Prerequisites

### System Requirements

- **OS:** Ubuntu 24.04 LTS (tested)
- **Kernel:** 6.8.x
- **Python:** 3.10 (required)

## Quick Start

```bash
# 0. Create and activate conda environment (if not already done)
conda create --name mlperf python=3.10 -y
conda activate mlperf

# 1. Run setup
./setup.sh --all

# 2. Check status
./setup.sh --status

# 3. Run a benchmark
./scripts/run_benchmark.sh bert --gpu --samples=100
```

## Setup Options

```bash
./setup.sh --clone         # Clone mlcommons/inference at v6.0 tag
./setup.sh --loadgen       # Install LoadGen library
./setup.sh --deps          # Install Python dependencies
./setup.sh --all           # Do all of the above
./setup.sh --status        # Show current setup status
```

## Clean Options

```bash
./setup.sh --clean              # Clean ALL folders (with confirmation)
./setup.sh --clean-inference    # Remove only inference/
./setup.sh --clean-data         # Remove only data/
./setup.sh --clean-models       # Remove only models/
./setup.sh --clean-results      # Remove only results/
./setup.sh --clean-cache        # Remove data/ + models/ (large files)
```

## Supported Benchmarks (v6.0)

| Benchmark | Model               | Dataset     | Script            |
|-----------|---------------------|-------------|-------------------|
| BERT      | bert-large-uncased  | SQuAD v1.1  | `run_bert.sh`     |
| ResNet50  | ResNet-50           | ImageNet    | `run_resnet50.sh` |
| 3D-UNet   | 3D-UNet             | KiTS19      | `run_3dunet.sh`   |
| DLRM-v3   | DLRM-v3             | Criteo      | `run_dlrm.sh`     |
| Llama     | Llama 2/3           | OpenOrca    | `run_llama.sh`    |
| Mixtral   | Mixtral-8x7B        | OpenOrca    | `run_mixtral.sh`  |
| SDXL      | Stable Diffusion XL | COCO        | `run_sdxl.sh`     |
| Whisper   | Whisper             | LibriSpeech | `run_whisper.sh`  |

**Note:** RetinaNet and GPT-J are NOT included in v6.0 (removed from official benchmarks).

## Changes from v5.1

- Removed: RetinaNet (object detection)
- Removed: GPT-J 6B
- Updated: DLRM-v2 → DLRM-v3

## Usage Examples

```bash
# Quick test with synthetic data
./scripts/run_benchmark.sh bert --gpu --samples=5

# Full benchmark with real data
./scripts/run_benchmark.sh bert --gpu --data=real --samples=1000

# Official MLPerf mode
./scripts/run_benchmark.sh bert --mlperf

# LLM with quantization
./scripts/run_benchmark.sh mixtral --gpu --4bit --offload

# Llama with specific model
./scripts/run_benchmark.sh llama llama3-8b --gpu --offload
```

## Folder Structure

```
v6.0/
├── setup.sh              # Setup and clean script
├── README.md             # This file
├── inference/            # MLCommons inference repo (cloned)
├── scripts/              # Benchmark runner scripts
│   ├── run_benchmark.sh  # Master runner
│   ├── run_bert.sh       
│   ├── run_resnet50.sh   
│   ├── run_3dunet.sh     
│   ├── run_dlrm.sh       
│   ├── run_llama.sh      
│   ├── run_mixtral.sh    
│   ├── run_sdxl.sh       
│   └── run_whisper.sh    
├── data/                 # Downloaded datasets
├── models/               # Downloaded model weights
└── results/              # Benchmark results (JSON)
```

## Requirements

- Python 3.8+
- CUDA 11.x+ (for GPU)
- 16GB+ RAM (32GB+ for LLMs)
- GPU: 8GB+ VRAM (24GB+ for LLMs without quantization)

## MLPerf Compliance

Use `--mlperf` flag for official MLPerf settings:
- Automatically downloads real datasets
- Uses official sample counts and settings
- Results are comparable to official MLPerf submissions

```bash
./scripts/run_benchmark.sh bert --mlperf
./scripts/run_benchmark.sh llama --mlperf --4bit --offload
```
