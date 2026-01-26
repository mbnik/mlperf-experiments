# MLPerf Experiments

Simplified MLPerf Inference benchmark suite with easy setup and CLI.

## Overview

This project provides an easy-to-use interface for running MLPerf Inference benchmarks. It wraps the official MLPerf specifications with simplified commands, automatic data downloading, and consistent output formatting.

**Supported Versions:**
- **v5.1** - MLPerf Inference v5.1.1 (stable)
- **v6.0** - MLPerf Inference v6.0.0pre (preview)

## Features

- **Version Isolation** - Each MLPerf version has its own complete environment
- **Simple CLI** - Unified command interface across all benchmarks
- **Auto-download** - Datasets and models download automatically when needed
- **Quantization** - 4-bit and 8-bit support for LLMs (reduced VRAM)
- **GPU Offloading** - Run large models on limited GPU memory
- **MLPerf Mode** - `--mlperf` flag for official-compliant settings
- **Consistent Output** - JSON results with standardized metrics

## Quick Start

```bash
# Create and activate conda environment (Python 3.10 required)
conda create --name mlperf python=3.10 -y
conda activate mlperf

# Choose a version and run setup
cd v5.1
./setup.sh --all

# Check status
./setup.sh --status

# Run a benchmark
./scripts/run_benchmark.sh bert --gpu --samples=100
```

## Project Structure

```
mlperf-experiments/
├── v5.1/                         # MLPerf Inference v5.1.1
│   ├── setup.sh                  # Setup and clean script
│   ├── README.md                 # v5.1 documentation
│   ├── inference/                # MLCommons inference repo (cloned)
│   ├── scripts/                  # Benchmark runner scripts
│   ├── data/                     # Downloaded datasets
│   ├── models/                   # Downloaded model weights
│   └── results/                  # Benchmark results (JSON)
│
├── v6.0/                         # MLPerf Inference v6.0.0pre
│   ├── setup.sh                  # Setup and clean script
│   ├── README.md                 # v6.0 documentation
│   ├── inference/                # MLCommons inference repo (cloned)
│   ├── scripts/                  # Benchmark runner scripts
│   ├── data/                     # Downloaded datasets
│   ├── models/                   # Downloaded model weights
│   └── results/                  # Benchmark results (JSON)
│
├── BENCHMARKS.md                 # Detailed benchmark documentation
├── DATA_GUIDE.md                 # Dataset download guide
└── README.md                     # This file
```

## Supported Benchmarks

### v5.1 Benchmarks (10 total)

| Benchmark | Model               | Dataset       |
|-----------|---------------------|---------------|
| BERT      | bert-large-uncased  | SQuAD v1.1    |
| ResNet50  | ResNet-50           | ImageNet      |
| RetinaNet | RetinaNet           | OpenImages    |
| 3D-UNet   | 3D-UNet             | KiTS19        |
| DLRM-v2   | DLRM-v2             | Criteo        |
| GPT-J     | GPT-J 6B            | CNN-DailyMail |
| Llama     | Llama 2/3           | OpenOrca      |
| Mixtral   | Mixtral-8x7B        | OpenOrca      |
| SDXL      | Stable Diffusion XL | COCO-2014     |
| Whisper   | Whisper Large       | LibriSpeech   |

### v6.0 Benchmarks (8 total)

| Benchmark | Model               | Dataset     |
|-----------|---------------------|-------------|
| BERT      | bert-large-uncased  | SQuAD v1.1  |
| ResNet50  | ResNet-50           | ImageNet    |
| 3D-UNet   | 3D-UNet             | KiTS19      |
| DLRM-v3   | DLRM-v3             | Criteo      |
| Llama     | Llama 2/3           | OpenOrca    |
| Mixtral   | Mixtral-8x7B        | OpenOrca    |
| SDXL      | Stable Diffusion XL | COCO-2014   |
| Whisper   | Whisper Large       | LibriSpeech |

**Note:** RetinaNet and GPT-J were removed in v6.0.

## Setup Commands

Each version folder has its own `setup.sh` with these options:

```bash
# Setup options
./setup.sh --clone         # Clone mlcommons/inference at correct tag
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
# Quick test with synthetic data
./scripts/run_benchmark.sh bert --gpu --samples=5

# Full benchmark with real data
./scripts/run_benchmark.sh resnet50 --gpu --data=real --samples=1000

# Official MLPerf mode
./scripts/run_benchmark.sh bert --mlperf

# LLM with quantization
./scripts/run_benchmark.sh mixtral --gpu --4bit --offload

# Llama with specific model size
./scripts/run_benchmark.sh llama llama3-8b --gpu --4bit
```

## Common Options

| Option             | Description                            |
|--------------------|----------------------------------------|
| `--gpu`            | Run on GPU (default)                   |
| `--cpu`            | Run on CPU only                        |
| `--samples=N`      | Number of samples to process           |
| `--data=synthetic` | Use synthetic data (fast, no download) |
| `--data=real`      | Use real dataset (downloads if needed) |
| `--mlperf`         | Use official MLPerf settings           |
| `-h, --help`       | Show help message                      |

### LLM Options (Llama, Mixtral, GPT-J)

| Option      | Description                 | VRAM Reduction |
|-------------|-----------------------------|----------------|
| `--offload` | GPU + CPU memory offloading | Varies         |
| `--4bit`    | 4-bit quantization          | ~4x            |
| `--8bit`    | 8-bit quantization          | ~2x            |

## Requirements

- Python 3.8+
- CUDA 11.x+ (for GPU)
- 16GB+ RAM (32GB+ for LLMs)
- GPU: 8GB+ VRAM (24GB+ for LLMs without quantization)

### HuggingFace Token (for gated models)

```bash
export HF_TOKEN=your_token_here
# Or login via CLI
huggingface-cli login
```

## Output

Results are saved to `<version>/results/<benchmark>/` as JSON:

```json
{
  "model": "bert-large",
  "device": "cuda",
  "throughput_samples_per_sec": 172.95,
  "avg_latency_ms": 5.78,
  "mlperf_mode": false,
  "mlperf_compliant": false
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
