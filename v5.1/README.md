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

All benchmarks are run through the unified `benchmark.py` script:

```bash
cd scripts/

# List available datasets
python benchmark.py --list

# Run benchmarks
python benchmark.py -b <benchmark> --dataset <dataset> [options]

# Examples
python benchmark.py -b bert --dataset squad -n 100 --mlperf
python benchmark.py -b whisper --dataset librispeech -n 50 --mlperf --target-qps 2
python benchmark.py -b llama --dataset openorca -n 10 --4bit --offload
python benchmark.py -b mixtral --dataset mixtral-15k --mlperf-quick --offload

# Quick test mode (60s duration, reduced samples)
python benchmark.py -b bert --dataset synthetic --mlperf-quick

# Full MLPerf mode (10min duration)
python benchmark.py -b bert --dataset squad --mlperf
```

## Data Management

```bash
cd scripts/

# Download real MLPerf datasets
python data_download.py <benchmark>
python data_download.py bert           # SQuAD v1.1
python data_download.py whisper        # LibriSpeech
python data_download.py dlrm           # Criteo (~100GB)

# Generate synthetic data for testing
python data_gen.py <benchmark> --num-samples 100
python data_gen.py bert --num-samples 500
python data_gen.py mixtral --num-samples 20
```

## Folder Structure

```
v5.1/
├── inference/           # MLCommons inference repo (v5.1.1 tag)
├── scripts/
│   ├── benchmark.py     # Unified benchmark runner (all 10 benchmarks)
│   ├── data_download.py # Dataset downloader
│   ├── data_gen.py      # Synthetic data generator
│   └── data_prepare.py  # Data preparation utilities
├── data/                # Downloaded/generated datasets
├── models/              # Downloaded models
├── results/             # Benchmark results (JSON)
└── README.md
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

## Memory Options (for large models)

| Option     | Description                | Use Case                    |
|------------|----------------------------|-----------------------------|
| `--4bit`   | 4-bit quantization         | ~4x VRAM reduction          |
| `--8bit`   | 8-bit quantization         | ~2x VRAM reduction          |
| `--offload`| CPU memory offloading      | When GPU VRAM insufficient  |

### VRAM Requirements

| Model      | FP16    | 8-bit   | 4-bit   | 4-bit+offload |
|------------|---------|---------|---------|---------------|
| GPT-J 6B   | ~12GB   | ~6GB    | ~4GB    | ~2GB          |
| Llama 70B  | ~140GB  | ~70GB   | ~35GB   | ~8GB          |
| Mixtral    | ~90GB   | ~48GB   | ~24GB   | ~8GB          |
| SDXL       | ~7GB    | -       | -       | ~4GB          |

## MLPerf v5.1 Reference

- **Tag**: v5.1.1
- **Submission Deadline**: August 1, 2025
- **Official Documentation**: https://docs.mlcommons.org/inference/benchmarks/

## Notes

- **RetinaNet** and **GPT-J** are included in v5.1 but were removed in v6.0
- **Mixtral-8x7B** is a large MoE model - use `--offload` for consumer GPUs (very slow)
- **Mixtral quick mode** uses 1 sample, 4 warmup tokens for faster testing
- Results saved to `results/<benchmark>/` as JSON files

## Author

**Mehdi Nik** - [GitHub](https://github.com/mbnik)
