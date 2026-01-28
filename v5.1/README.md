# MLPerf Inference v5.1 Benchmarks

This folder contains benchmark scripts for **MLPerf Inference v5.1**.

## Prerequisites

### System Requirements

- **OS:** Ubuntu 24.04 LTS (tested)
- **Kernel:** 6.8.x
- **Python:** 3.10 (required)

```bash
# Create and activate conda environment (Python 3.10 required)
conda create --name mlperf python=3.10 -y
conda activate mlperf
```

## Setup

```bash
# Full setup (clone inference repo, install LoadGen and dependencies)
./setup.sh --all

# Or step by step
./setup.sh --clone      # Clone MLCommons inference repo at v5.1.1 tag
./setup.sh --loadgen    # Install LoadGen library
./setup.sh --deps       # Install Python dependencies

# Check current status
./setup.sh --status
```

## Clean

```bash
# Clean everything (with confirmation prompt)
./setup.sh --clean

# Clean specific folders
./setup.sh --clean-inference    # Remove inference/ only
./setup.sh --clean-data         # Remove data/ only
./setup.sh --clean-models       # Remove models/ only
./setup.sh --clean-results      # Remove results/ only
./setup.sh --clean-cache        # Remove data/ + models/ (large files)
```

## Supported Benchmarks

| Benchmark | Model               | Task                 | Dataset               |
|-----------|---------------------|----------------------|-----------------------|
| BERT      | BERT-Large          | Question Answering   | SQuAD v1.1            |
| ResNet50  | ResNet-50 v1.5      | Image Classification | ImageNet 2012         |
| RetinaNet | RetinaNet 800x800   | Object Detection     | OpenImages            |
| 3D-UNet   | 3D-UNet             | Medical Segmentation | KiTS19                |
| DLRM      | DLRM-v2             | Recommendation       | Criteo Terabyte       |
| GPT-J     | GPT-J 6B            | Text Summarization   | CNN-DailyMail         |
| Llama     | Llama 2-70B         | Text Generation      | OpenOrca              |
| Mixtral   | Mixtral-8x7B        | Text Generation      | OpenOrca, MBXP, GSM8K |
| SDXL      | Stable Diffusion XL | Image Generation     | COCO 2014             |
| Whisper   | Whisper Large       | Speech Recognition   | LibriSpeech           |

## Quick Start

```bash
# Run any benchmark
./scripts/run_benchmark.sh <benchmark> [options]

# Examples
./scripts/run_benchmark.sh bert --gpu --samples=100
./scripts/run_benchmark.sh llama --gpu --4bit --offload
./scripts/run_benchmark.sh mixtral --4bit --offload --mlperf

# MLPerf compliant mode
./scripts/run_benchmark.sh bert --mlperf
```

## Folder Structure

```
v5.1/
├── inference/     # MLCommons inference repo (v5.1.1 tag)
├── scripts/       # Benchmark runner scripts
├── data/          # Downloaded datasets
├── models/        # Downloaded models
├── results/       # Benchmark results
└── README.md
```

## Common Options

| Option             | Description                  |
|--------------------|------------------------------|
| `--gpu`            | Run on GPU (default)         |
| `--cpu`            | Run on CPU only              |
| `--offload`        | GPU + CPU memory offloading  |
| `--samples=N`      | Number of samples to process |
| `--mlperf`         | Use official MLPerf settings |
| `--data=synthetic` | Use synthetic data (fast)    |
| `--data=real`      | Use real dataset             |

## LLM Options (GPT-J, Llama, Mixtral)

| Option   | Description        | VRAM Reduction |
|----------|--------------------|----------------|
| `--4bit` | 4-bit quantization | ~4x            |
| `--8bit` | 8-bit quantization | ~2x            |

## MLPerf v5.1 Reference

- **Tag**: v5.1.1
- **Submission Deadline**: August 1, 2025
- **Official Documentation**: https://docs.mlcommons.org/inference/benchmarks/

## Notes

- **RetinaNet** and **GPT-J** are included in v5.1 but were removed in v6.0
- **Mixtral-8x7B** is a large MoE model - use `--4bit --offload` for consumer GPUs
- Results saved to `results/<benchmark>/` as JSON files

## Author

**Mehdi Nik** - [GitHub](https://github.com/mbnik)
