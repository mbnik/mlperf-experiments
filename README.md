# MLPerf Experiments

Simplified MLPerf Inference benchmark suite with easy setup and CLI.

## Overview

This project provides an easy-to-use interface for running MLPerf Inference benchmarks. It wraps the official MLPerf specifications with simplified commands, automatic data downloading, and consistent output formatting.

**Current Version:** v5.1 (MLPerf Inference v5.1.1)

## Features

- **Unified CLI** - Single `benchmark.py` script for all 10 benchmarks
- **Auto-download** - Datasets and models download automatically when needed
- **Synthetic Data** - Generate test data for quick validation
- **Quantization** - 4-bit and 8-bit support for LLMs (reduced VRAM)
- **GPU Offloading** - Run large models on limited GPU memory
- **MLPerf Mode** - `--mlperf` flag for official-compliant settings
- **Consistent Output** - JSON results with standardized metrics

## Quick Start

```bash
# Create and activate conda environment (Python 3.10 required)
conda create --name mlperf python=3.10 -y
conda activate mlperf

# Setup
cd v5.1
./setup.sh --all

# Check status
./setup.sh --status

# List available datasets
python scripts/benchmark.py --list

# Run a benchmark
python scripts/benchmark.py -b bert --dataset squad -n 100 --mlperf
```

## Project Structure

```
mlperf-experiments/
├── v5.1/                         # MLPerf Inference v5.1.1
│   ├── setup.sh                  # Setup and clean script
│   ├── README.md                 # v5.1 documentation
│   ├── inference/                # MLCommons inference repo (cloned)
│   ├── scripts/
│   │   ├── benchmark.py          # Unified benchmark runner (all 10 benchmarks)
│   │   ├── data_download.py      # Dataset downloader
│   │   ├── data_gen.py           # Synthetic data generator
│   │   └── data_prepare.py       # Data preparation utilities
│   ├── data/                     # Downloaded/generated datasets
│   ├── models/                   # Downloaded model weights
│   └── results/                  # Benchmark results (JSON)
│
├── BENCHMARKS.md                 # Detailed benchmark documentation
├── DATA_GUIDE.md                 # Dataset download guide
└── README.md                     # This file
```

## Supported Benchmarks (10 total)

| Benchmark | Model               | Task                 | Dataset       |
|-----------|---------------------|----------------------|---------------|
| BERT      | BERT-Large          | Question Answering   | SQuAD v1.1    |
| ResNet50  | ResNet-50           | Image Classification | ImageNet      |
| RetinaNet | RetinaNet           | Object Detection     | OpenImages    |
| 3D-UNet   | 3D-UNet             | Medical Segmentation | KiTS19        |
| DLRM-v2   | DLRM-v2             | Recommendation       | Criteo        |
| GPT-J     | GPT-J 6B            | Text Summarization   | CNN-DailyMail |
| Llama     | Llama 2/3           | Text Generation      | OpenOrca      |
| Mixtral   | Mixtral-8x7B        | Text Generation      | GSM8K/OpenOrca|
| SDXL      | Stable Diffusion XL | Image Generation     | COCO-2014     |
| Whisper   | Whisper Large       | Speech Recognition   | LibriSpeech   |

## Setup Commands

```bash
cd v5.1

# Setup options
./setup.sh --clone         # Clone mlcommons/inference at v5.1.1 tag
./setup.sh --loadgen       # Install LoadGen library
./setup.sh --deps          # Install Python dependencies
./setup.sh --all           # Do all of the above
./setup.sh --status        # Show current setup status

# Clean options
./setup.sh --clean              # Clean ALL folders (with confirmation)
./setup.sh --clean-inference    # Remove only inference/
./setup.sh --clean-data         # Remove only data/
./setup.sh --clean-models       # Remove only models/
./setup.sh --clean-results      # Remove only results/
./setup.sh --clean-cache        # Remove data/ + models/ (large files)
```

## Usage Examples

```bash
cd v5.1/scripts

# List available datasets
python benchmark.py --list

# Quick test with synthetic data
python benchmark.py -b bert --dataset synthetic -n 50 --mlperf-quick

# Full benchmark with real data
python benchmark.py -b bert --dataset squad -n 1000 --mlperf

# Whisper benchmark
python benchmark.py -b whisper --dataset librispeech -n 50 --mlperf --target-qps 2

# LLM with quantization and offloading
python benchmark.py -b llama --dataset openorca -n 10 --4bit --offload
python benchmark.py -b mixtral --dataset mixtral-15k --mlperf-quick --offload

# Generate synthetic data
python data_gen.py bert --num-samples 500
python data_gen.py mixtral --num-samples 20

# Download real datasets
python data_download.py bert
python data_download.py whisper
```

## Common Options

| Option              | Description                           |
|---------------------|---------------------------------------|
| `-b, --benchmark`   | Benchmark name (required)             |
| `--dataset`         | Dataset name (use --list to see all)  |
| `-n, --max-samples` | Number of samples to process          |
| `--mlperf`          | Full MLPerf mode (10min duration)     |
| `--mlperf-quick`    | Quick test mode (60s duration)        |
| `--target-qps`      | Target queries per second             |
| `--device`          | Device: cuda, cpu (default: cuda)     |
| `--list`            | List available datasets               |

### LLM Options (Llama, Mixtral, GPT-J)

| Option      | Description                 | VRAM Reduction |
|-------------|-----------------------------|----------------|
| `--offload` | GPU + CPU memory offloading | Varies         |
| `--4bit`    | 4-bit quantization          | ~4x            |
| `--8bit`    | 8-bit quantization          | ~2x            |

### VRAM Requirements

| Model      | FP16    | 8-bit   | 4-bit   | 4-bit+offload |
|------------|---------|---------|---------|---------------|
| GPT-J 6B   | ~12GB   | ~6GB    | ~4GB    | ~2GB          |
| Llama 70B  | ~140GB  | ~70GB   | ~35GB   | ~8GB          |
| Mixtral    | ~90GB   | ~48GB   | ~24GB   | ~8GB          |
| SDXL       | ~7GB    | -       | -       | ~4GB          |

## Requirements

### Tested Environment

- **OS:** Ubuntu 24.04 LTS
- **Kernel:** 6.8.x
- **Python:** 3.10 (required)

### Hardware Requirements

- CUDA 11.x+ (for GPU)
- 16GB+ RAM (32GB+ for LLMs)
- GPU: 8GB+ VRAM (24GB+ for LLMs without quantization)

> **Note:** This project has only been tested on Ubuntu 24.04. Other Linux distributions may work but are not officially supported.

### HuggingFace Token (for gated models)

```bash
export HF_TOKEN=your_token_here
# Or login via CLI
huggingface-cli login
```

## Output

Results are saved to `v5.1/results/<benchmark>/` as JSON:

```json
{
  "model": "bert-large",
  "device": "cuda",
  "throughput_samples_per_sec": 172.95,
  "avg_latency_ms": 5.78,
  "mlperf_mode": true,
  "mlperf_compliant": true
}
```

## Disclaimer

This repository is **NOT** an official MLPerf implementation. It contains personal tooling and scripts to run MLPerf workloads.

This software is provided "as is" without warranty of any kind, express or implied. The author assumes no responsibility for errors, omissions, or damages arising from its use.

Benchmark results generated by this software are for development, testing, and educational purposes only and should not be cited as official MLPerf performance metrics.

For official MLPerf benchmarks, see: https://mlcommons.org/benchmarks/inference/

## Citation

```bibtex
@software{mlperf_experiments,
  author = {Nik, Mehdi},
  title = {MLPerf Experiments: Simplified MLPerf Inference Benchmark Suite},
  year = {2026},
  url = {https://github.com/mbnik/mlperf-experiments}
}
```

## Author

**Mehdi Nik** - [GitHub](https://github.com/mbnik)

---

All rights reserved. If you use or reference this work, please provide attribution to the original author.
