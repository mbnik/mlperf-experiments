# MLPerf Inference Benchmark Scripts

## Overview

This document provides detailed benchmark documentation for MLPerf Inference v5.1. All benchmarks are accessed through a unified `benchmark.py` script.

> **Note**: This repository contains personal tooling and scripts for running MLPerf workloads. It is NOT an official MLPerf implementation.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Benchmark Test Results](#benchmark-test-results)
- [Feature Matrix](#feature-matrix)
- [MLPerf Compliance Mode](#mlperf-compliance-mode)
- [Memory Requirements](#memory-requirements)
- [Data Sources](#data-sources)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
cd v5.1/scripts

# List all available benchmarks and datasets
python benchmark.py --list

# Run any benchmark
python benchmark.py -b bert --dataset squad -n 100 --mlperf

# Run with CPU offloading (for limited VRAM)
python benchmark.py -b llama --dataset openorca -n 10 --offload

# Run LLMs with 4-bit quantization
python benchmark.py -b llama --dataset openorca -n 10 --4bit
```

---

## Benchmark Test Results

### Test Environment

| Component   | Specification                            |
|-------------|------------------------------------------|
| **GPU**     | NVIDIA GeForce RTX 3080 Ti (12GB GDDR6X) |
| **CPU**     | Intel Xeon (Dell Precision 7920)         |
| **RAM**     | 64GB+ DDR4                               |
| **OS**      | Ubuntu 24.04 LTS                         |
| **Kernel**  | 6.8.x                                    |
| **Python**  | 3.10 (required)                          |
| **PyTorch** | 2.x with CUDA 12.x                       |

### v5.1 Results Summary

All benchmarks tested with `--mlperf` flag using real MLPerf datasets (January 2026).

| Benchmark        | Dataset          | Status  | Throughput         | Latency | Mode    |
|------------------|------------------|:-------:|--------------------|---------|---------|
| **BERT**         | SQuAD v1.1       | ✅ PASS | 175.59 samples/sec | 5.7ms   | GPU     |
| **ResNet50**     | ImageNet-1K      | ✅ PASS | 255.16 images/sec  | 3.9ms   | GPU     |
| **RetinaNet**    | COCO 2017        | ✅ PASS | 29.71 images/sec   | 168ms   | GPU     |
| **3D-UNet**      | KiTS19           | ✅ PASS | 10.12 volumes/sec  | 129ms   | GPU     |
| **DLRM-v2**      | Criteo Terabyte  | ✅ PASS | 34,565 samples/sec | 2.5ms   | GPU     |
| **GPT-J**        | CNN-DailyMail    | ✅ PASS | 14.19 tok/sec      | 9.0s    | 4-bit   |
| **Llama 3.1 8B** | CNN-DailyMail    | ✅ PASS | 0.44 tok/sec       | 2337s   | offload |
| **Mixtral-8x7B** | OpenOrca         | ✅ PASS | 0.03 tok/sec       | 3586s   | offload |
| **SDXL**         | COCO Captions    | ✅ PASS | 4.9 images/min     | 12.2s   | offload |
| **Whisper**      | LibriSpeech      | ✅ PASS | 119.3x realtime    | -       | GPU     |

### Detailed Results

<details>
<summary><b>Vision Models</b></summary>

```
BERT (Question Answering)
├── Dataset: SQuAD v1.1 (dev-v1.1.json)
├── Throughput: 175.59 samples/sec
├── Latency: 5.7ms average
├── Mode: GPU (full)
└── Command: python benchmark.py -b bert --dataset squad -n 100 --mlperf

ResNet50 (Image Classification)
├── Dataset: ImageNet-1K (1000 validation images)
├── Throughput: 255.16 images/sec
├── Top-1 Accuracy: 20.0% (on subset)
├── Mode: GPU (full)
└── Command: python benchmark.py -b resnet50 --dataset imagenet -n 100 --mlperf

RetinaNet (Object Detection)
├── Dataset: COCO 2017 validation
├── Throughput: 29.71 images/sec
├── Avg Detections: 161.7/image
├── Mode: GPU (full)
└── Command: python benchmark.py -b retinanet --dataset openimages -n 100 --mlperf

3D-UNet (Medical Segmentation)
├── Dataset: KiTS19 (20 cases)
├── Throughput: 10.12 volumes/sec
├── Dice Score: 0.007 (K+T mean, random init)
├── Mode: GPU (full)
└── Command: python benchmark.py -b 3dunet --dataset kits19 -n 20 --mlperf
```

</details>

<details>
<summary><b>Language Models</b></summary>

```
GPT-J 6B (Text Generation)
├── Dataset: CNN-DailyMail
├── Throughput: 14.19 tokens/sec
├── ROUGE-L: 0.214
├── Mode: GPU + 4-bit quantization
└── Command: python benchmark.py -b gptj --dataset cnn-dailymail -n 10 --4bit --mlperf

Llama 3.1 8B (Text Generation)
├── Dataset: CNN-DailyMail
├── Throughput: 0.44 tokens/sec
├── Mode: GPU + CPU offload (FP16)
├── Note: Slow due to 12GB VRAM limitation
└── Command: python benchmark.py -b llama --dataset openorca -n 5 --offload --mlperf

Mixtral-8x7B (Text Generation - MoE)
├── Dataset: OpenOrca
├── Throughput: 0.03 tokens/sec
├── ROUGE-L: 0.538
├── Mode: GPU + CPU offload (FP16)
├── Note: Very slow - 93GB model on 12GB GPU
└── Command: python benchmark.py -b mixtral --dataset mixtral-15k -n 1 --offload --mlperf-quick
```

</details>

<details>
<summary><b>Other Models</b></summary>

```
DLRM-v2 (Recommendation)
├── Dataset: Criteo Terabyte (Day 23)
├── Throughput: 34,565 samples/sec
├── Accuracy: 94.18%
├── AUC-ROC: 0.8419
├── Mode: GPU (sample model)
└── Command: python benchmark.py -b dlrm --dataset criteo -n 1000 --mlperf

SDXL (Image Generation)
├── Dataset: COCO 2014 Captions
├── Throughput: 4.9 images/min
├── Latency: 12.2 sec/image
├── Mode: GPU + CPU offload
└── Command: python benchmark.py -b sdxl --dataset coco-2014 -n 5 --offload --mlperf

Whisper Large-v3 (Speech Recognition)
├── Dataset: LibriSpeech test-clean
├── Speed: 119.3x realtime
├── WER: 2.8%
├── Mode: GPU (full)
└── Command: python benchmark.py -b whisper --dataset librispeech -n 50 --mlperf
```

</details>

---

## Feature Matrix

| Benchmark | --device cuda | --device cpu | --offload | --4bit | --8bit | --mlperf | --mlperf-quick |
|-----------|:-------------:|:------------:|:---------:|:------:|:------:|:--------:|:--------------:|
| BERT      | ✓             | ✓            | ✓         | -      | -      | ✓        | ✓              |
| ResNet50  | ✓             | ✓            | ✓         | -      | -      | ✓        | ✓              |
| RetinaNet | ✓             | ✓            | ✓         | -      | -      | ✓        | ✓              |
| 3D-UNet   | ✓             | ✓            | ✓         | -      | -      | ✓        | ✓              |
| DLRM      | ✓             | ✓            | ✓         | -      | -      | ✓        | ✓              |
| GPT-J     | ✓             | ✓            | ✓         | ✓      | ✓      | ✓        | ✓              |
| Llama     | ✓             | ✓            | ✓         | ✓      | ✓      | ✓        | ✓              |
| Mixtral   | ✓             | ✓            | ✓         | ✓      | ✓      | ✓        | ✓              |
| SDXL      | ✓             | ✓            | ✓         | -      | -      | ✓        | ✓              |
| Whisper   | ✓             | ✓            | ✓         | -      | -      | ✓        | ✓              |

All 10 benchmarks use a unified Python script: `v5.1/scripts/benchmark.py`

## Script Architecture

The unified benchmark runner consolidates all benchmarks into a single script:

```
v5.1/scripts/
├── benchmark.py          # Unified benchmark runner (all 10 benchmarks)
├── data_download.py      # Dataset downloader
├── data_gen.py           # Synthetic data generator
└── data_prepare.py       # Data preparation utilities
```

**benchmark.py** handles:
- Model loading and configuration
- Dataset loading (synthetic or real)
- Benchmark execution with LoadGen
- Results JSON output with `mlperf_mode` and `mlperf_compliant` fields
- Consistent error handling and OOM guidance
- Quantization and offloading options

## MLPerf Compliance Mode

All benchmarks support `--mlperf` and `--mlperf-quick` flags for MLPerf-compliant settings:

```bash
# Run with official MLPerf settings (10 minute duration)
python benchmark.py -b bert --dataset squad -n 1000 --mlperf

# Quick test mode (60 second duration)
python benchmark.py -b bert --dataset squad -n 100 --mlperf-quick

# Mixtral with quick mode (1 sample, 4 warmup tokens)
python benchmark.py -b mixtral --dataset mixtral-15k --mlperf-quick --offload
```

### What --mlperf Does

| Setting    | Without --mlperf | With --mlperf              | With --mlperf-quick |
|------------|------------------|----------------------------|---------------------|
| Duration   | 60 seconds       | 600 seconds (10 min)       | 60 seconds          |
| LLM tokens | Default          | 1024 (Llama), 128 (GPT-J)  | Same                |
| SDXL steps | Default          | 20 (official)              | 20                  |
| Results    | For quick testing| Comparable to official     | Quick validation    |

### Official MLPerf Settings

| Benchmark | Dataset       | Target          |
|-----------|---------------|-----------------|
| Llama     | OpenORCA      | ROUGE-1 ≥ 44.38 |
| GPT-J     | CNN-DailyMail | ROUGE-1 ≥ 42.94 |
| BERT      | SQuAD v1.1    | F1 ≥ 89.97%     |
| ResNet50  | ImageNet      | Top-1 ≥ 76.46%  |
| RetinaNet | OpenImages    | mAP ≥ 37.55%    |
| 3D-UNet   | KiTS19        | Dice ≥ 86.1%    |
| DLRM      | Criteo        | AUC ≥ 80.25%    |
| SDXL      | COCO-2014     | CLIP ≥ 31.68    |
| Whisper   | LibriSpeech   | WER ≤ 4.3%      |
| Mixtral   | GSM8K/OpenOrca| GSM8K ≥ 73.78%  |

## Common Options

All benchmarks support these standardized options:

| Option              | Description                              |
|---------------------|------------------------------------------|
| `-b, --benchmark`   | Benchmark name (required)                |
| `--dataset`         | Dataset name (use --list to see all)     |
| `-n, --max-samples` | Number of samples to process             |
| `--device`          | Device: cuda, cpu (default: cuda)        |
| `--mlperf`          | Full MLPerf mode (10min duration)        |
| `--mlperf-quick`    | Quick test mode (60s duration)           |
| `--target-qps`      | Target queries per second                |
| `--list`            | List available benchmarks and datasets   |
| `-h, --help`        | Show help message                        |

## LLM-Specific Options

Language models (GPT-J, Llama, Mixtral) support quantization and offloading:

| Option      | Description                 | VRAM Reduction |
|-------------|-----------------------------|----------------|
| `--offload` | GPU + CPU memory offloading | Varies         |
| `--4bit`    | 4-bit quantization          | ~4x            |
| `--8bit`    | 8-bit quantization          | ~2x            |

### Llama Model Variants

```bash
python benchmark.py -b llama --dataset openorca -n 10 --4bit
```

| Model        | Size | Full VRAM | 4-bit VRAM |
|--------------|------|-----------|------------|
| Llama 3.1 8B | 8B   | ~16GB     | ~4GB       |
| Llama 2 70B  | 70B  | ~140GB    | ~35GB      |

**Note:** Llama models require HuggingFace authentication:
```bash
export HF_TOKEN=your_token_here
```

## Benchmark Details

### 3D-UNet (Medical Image Segmentation)
```bash
python benchmark.py -b 3dunet --dataset kits19 -n 20 --mlperf
```
- **Data**: KiTS19 kidney tumor dataset
- **Real data size**: ~150-500MB per case (210 cases total)

### DLRM (Recommendation)
```bash
python benchmark.py -b dlrm --dataset criteo -n 1000 --mlperf
```
- **Data**: Criteo Terabyte dataset
- **Real data size**: ~1TB (downloads day_23 for validation)

### GPT-J (Text Summarization)
```bash
python benchmark.py -b gptj --dataset cnn-dailymail -n 10 --4bit --mlperf
```
- **Model**: EleutherAI/gpt-j-6b (~24GB full, ~6GB 4-bit)
- **Data**: CNN-DailyMail dataset

### Mixtral (Text Generation)
```bash
python benchmark.py -b mixtral --dataset mixtral-15k -n 5 --offload --mlperf-quick
```
- **Model**: mistralai/Mixtral-8x7B-Instruct-v0.1 (~93GB)
- **Data**: GSM8K + OpenOrca (MLPerf reference)
- **MLPerf Settings**: min_new_tokens=2, max_new_tokens=1024

### RetinaNet (Object Detection)
```bash
python benchmark.py -b retinanet --dataset openimages -n 100 --mlperf
```
- **Data**: OpenImages validation subset (~2GB)

### SDXL (Image Generation)
```bash
python benchmark.py -b sdxl --dataset coco-2014 -n 5 --offload --mlperf
```
- **Model**: stabilityai/stable-diffusion-xl-base-1.0
- **Data**: COCO-2014 captions for prompts

### Whisper (Speech Recognition)
```bash
python benchmark.py -b whisper --dataset librispeech -n 50 --mlperf
```
- **Model**: openai/whisper-large-v3
- **Data**: LibriSpeech test-clean (~350MB)

## Memory Requirements

### GPU VRAM Requirements

| Benchmark  | Minimum VRAM | Recommended  |
|------------|--------------|--------------|
| BERT       | 4GB          | 8GB          |
| ResNet50   | 4GB          | 8GB          |
| RetinaNet  | 4GB          | 8GB          |
| 3D-UNet    | 4GB          | 8GB          |
| DLRM       | 4GB (sample) | 24GB+ (full) |
| GPT-J      | 6GB (4-bit)  | 24GB (full)  |
| Llama 8B   | 4GB (4-bit)  | 16GB (full)  |
| Mixtral    | 24GB (4-bit) | 93GB (full)  |
| SDXL       | 8GB          | 12GB+        |
| Whisper    | 4GB          | 8GB          |

### Using --offload for Limited VRAM

The `--offload` option enables GPU+CPU memory offloading:
- Keeps compute on GPU for speed
- Offloads some layers/weights to system RAM
- Allows running larger models on smaller GPUs

```bash
# Run Mixtral on 12GB GPU
python benchmark.py -b mixtral --dataset mixtral-15k -n 1 --offload
```

---

## Memory Offloading Deep Dive

The `--offload` flag works differently depending on the benchmark type.

### Offloading Strategies by Benchmark

| Benchmark     | Offload Method                         | What Goes to CPU            | What Stays on GPU      |
|---------------|----------------------------------------|-----------------------------|------------------------|
| **BERT**      | HuggingFace `device_map="auto"`        | Automatically balanced      | Automatically balanced |
| **ResNet50**  | Batch size reduction only              | Nothing (model fits)        | Entire model           |
| **RetinaNet** | Batch size reduction only              | Nothing (model fits)        | Entire model           |
| **3D-UNet**   | Batch size reduction only              | Nothing (model fits)        | Entire model           |
| **DLRM**      | Manual layer split                     | 26 embedding tables (~97GB) | MLP layers (~50MB)     |
| **GPT-J**     | HuggingFace `device_map="auto"`        | Automatically balanced      | Automatically balanced |
| **Llama**     | HuggingFace `device_map="auto"`        | Automatically balanced      | Automatically balanced |
| **Mixtral**   | HuggingFace `device_map="auto"`        | Automatically balanced      | Automatically balanced |
| **SDXL**      | Diffusers `enable_model_cpu_offload()` | Idle components             | Active component       |
| **Whisper**   | HuggingFace `device_map="auto"`        | Automatically balanced      | Automatically balanced |

### LLMs: Automatic Layer Distribution

For transformer models (Llama, Mixtral, GPT-J, Whisper, BERT), HuggingFace's `device_map="auto"` automatically distributes layers:

```python
# How it works internally:
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",      # HuggingFace analyzes layer sizes
    torch_dtype=torch.float16,
)
```

**Automatic distribution example (Mixtral on 12GB GPU):**
```
┌─────────────────────────────────────────────────────────────┐
│  🖥️  GPU (12GB used):                                       │
│      ├── First few transformer layers                       │
│      └── Active expert heads                                │
│                                                             │
│  💾 CPU (80GB+ used):                                       │
│      └── Remaining layers and experts                       │
└─────────────────────────────────────────────────────────────┘
```

**Pros:** No manual tuning needed  
**Cons:** Slower than quantization due to CPU↔GPU transfers

### SDXL: Sequential Component Offloading

Stable Diffusion XL uses Diffusers' built-in offloading:

```python
pipe.enable_model_cpu_offload()
```

**How it works:**
```
┌─────────────────────────────────────────────────────────────┐
│  Time →                                                     │
│                                                             │
│  Step 1: Text encoding                                      │
│      🖥️  GPU: CLIP text encoder                             │
│      💾 CPU: UNet, VAE (idle)                               │
│                                                             │
│  Step 2: Diffusion (50 steps)                               │
│      🖥️  GPU: UNet                                          │
│      💾 CPU: CLIP, VAE (idle)                               │
│                                                             │
│  Step 3: Decode latents                                     │
│      🖥️  GPU: VAE decoder                                   │
│      💾 CPU: CLIP, UNet (idle)                              │
└─────────────────────────────────────────────────────────────┘
```

**Result:** ~3GB VRAM instead of ~6.5GB (at cost of speed)

### Performance Comparison

| Model        | Mode        | VRAM  | Speed |
|--------------|-------------|-------|-------|
| Llama 8B     | GPU (FP16)  | 16GB  | 100%  |
| Llama 8B     | 4-bit quant | 4GB   | ~80%  |
| Llama 8B     | --offload   | 8GB   | ~20%  |
| Mixtral 8x7B | GPU (FP16)  | 93GB  | 100%  |
| Mixtral 8x7B | 4-bit quant | 24GB  | ~80%  |
| Mixtral 8x7B | --offload   | 12GB  | ~5%   |
| DLRM full    | GPU only    | 97GB+ | 100%  |
| DLRM full    | --offload   | 1GB   | ~60%  |

**Recommendation:**
1. **If model fits in VRAM**: Use GPU only (fastest)
2. **If close to VRAM limit**: Use quantization for LLMs
3. **If significantly over VRAM**: Use --offload (slowest but works)

## Output

Results are saved to `v5.1/results/<benchmark>/` directory:
- JSON files with detailed metrics
- Throughput (samples/sec or tokens/sec)
- Latency statistics
- Accuracy scores (when applicable)

Example output:
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

---

## Data Sources

### Automatic Downloads

These datasets download automatically or can be generated synthetically:

| Dataset                | Source                     | Size   | Benchmark    |
|------------------------|----------------------------|--------|--------------|
| SQuAD v1.1             | rajpurkar.github.io        | 35MB   | BERT         |
| KiTS19                 | github.com/neheller/kits19 | ~500MB | 3D-UNet      |
| LibriSpeech test-clean | openslr.org                | ~350MB | Whisper      |
| COCO 2014 Captions     | images.cocodataset.org     | ~250MB | SDXL         |
| COCO 2017              | images.cocodataset.org     | ~1GB   | RetinaNet    |
| CNN-DailyMail          | HuggingFace                | ~300MB | GPT-J, Llama |
| OpenOrca               | HuggingFace                | ~4GB   | Mixtral      |
| Criteo Terabyte        | mlcommons-storage.org      | ~150GB | DLRM         |

### Gated/Manual Downloads

These require authentication or approval:

| Dataset          | Source                                     | Requirements                   |
|------------------|--------------------------------------------|--------------------------------|
| **ImageNet-1K**  | huggingface.co/datasets/ILSVRC/imagenet-1k | HuggingFace account + approval |
| **Llama models** | huggingface.co/meta-llama                  | HuggingFace + Meta approval    |

### Setting Up HuggingFace Access

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login to HuggingFace
huggingface-cli login

# For gated datasets (Llama, ImageNet), request access on HuggingFace website first
```

---

## Troubleshooting

### CUDA Out of Memory

Scripts will fail with a clear error message and suggestions:
```
============================================================
CUDA OUT OF MEMORY!
============================================================
Your GPU doesn't have enough VRAM. Try:
  1. --offload    : Enable GPU+CPU memory offloading
  2. --4bit       : Use 4-bit quantization (~4GB VRAM)
  3. --8bit       : Use 8-bit quantization (~7GB VRAM)
  4. --device cpu : Run on CPU only (very slow)
```

**Solution options:**
1. Use `--offload` for GPU+CPU memory sharing
2. Use `--4bit` or `--8bit` for LLMs
3. Reduce `-n` (max samples)

### Mixtral OOM Handling

Mixtral (~93GB) requires special handling:
```bash
# Use offload mode (required for <48GB VRAM)
python benchmark.py -b mixtral --dataset mixtral-15k -n 1 --offload

# With 4-bit quantization (24GB VRAM)
python benchmark.py -b mixtral --dataset mixtral-15k -n 1 --4bit
```

### Quantization + Offload Incompatibility

**Error:**
```
ValueError: Some modules are dispatched on the CPU or the disk. 
Make sure you have enough GPU RAM to fit the quantized model.
```

**Cause:** bitsandbytes 4-bit/8-bit quantization cannot be combined with CPU offloading.

**Solution:** Use either quantization OR offloading, not both:
```bash
# Choose ONE:
python benchmark.py -b llama --4bit --mlperf      # Quantization only
python benchmark.py -b llama --offload --mlperf   # Offload only (FP16)

# This will NOT work:
python benchmark.py -b llama --4bit --offload     # ERROR!
```

### CUDA Not Available
If CUDA is not available, scripts fail with guidance:
```
CUDA not available. Use --device cpu for CPU-only mode (slow)
```

### Model Download Issues
- Check internet connection
- For Llama: Set `HF_TOKEN` environment variable
- Use synthetic datasets to skip data download

### Slow Performance on CPU
CPU inference is significantly slower. Use GPU when possible:
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Design Philosophy

These scripts follow explicit failure with guidance:
- **No automatic fallback**: Scripts don't silently switch devices
- **Clear error messages**: When things fail, you get actionable suggestions
- **Explicit flags**: Use `--device cuda`, `--device cpu`, or `--offload` to control behavior
- **Predictable**: Same flags produce same behavior across all benchmarks
- **Consistent**: All benchmarks use the same argument parser and output format

### Consistency Guarantees

| Aspect          | Standard                                          |
|-----------------|---------------------------------------------------|
| Script entry    | `python benchmark.py -b <benchmark>`              |
| Error handling  | Try/except with OOM detection and suggestions     |
| JSON output     | Always includes `mlperf_mode`, `mlperf_compliant` |
| Results path    | `v5.1/results/<benchmark>/`                       |
| Exit on CUDA OOM| Clear error message with options                  |

---

*Last updated: February 4, 2026*  
*Tested on: NVIDIA RTX 3080 Ti (12GB VRAM)*  
*Author: Mehdi Nik*
