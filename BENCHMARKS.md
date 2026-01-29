# MLPerf Inference Benchmark Scripts

## Overview

This directory contains benchmark scripts for MLPerf Inference v5.1 and v6.0. All custom benchmarks follow a consistent interface pattern.

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
# Run any benchmark with GPU
./scripts/run_<benchmark>.sh --gpu

# Run with CPU offloading (for limited VRAM)
./scripts/run_<benchmark>.sh --offload

# Run with real data
./scripts/run_<benchmark>.sh --gpu --data=real

# Run LLMs with 4-bit quantization
./scripts/run_llama.sh llama3-8b --gpu --4bit
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
└── Command: ./scripts/run_bert.sh --mlperf --samples=100

ResNet50 (Image Classification)
├── Dataset: ImageNet-1K (1000 validation images)
├── Throughput: 255.16 images/sec
├── Top-1 Accuracy: 20.0% (on subset)
├── Mode: GPU (full)
└── Command: ./scripts/run_resnet50.sh --mlperf --samples=100

RetinaNet (Object Detection)
├── Dataset: COCO 2017 validation
├── Throughput: 29.71 images/sec
├── Avg Detections: 161.7/image
├── Mode: GPU (full)
└── Command: ./scripts/run_retinanet.sh --mlperf --samples=100

3D-UNet (Medical Segmentation)
├── Dataset: KiTS19 (20 cases)
├── Throughput: 10.12 volumes/sec
├── Dice Score: 0.007 (K+T mean, random init)
├── Mode: GPU (full)
└── Command: ./scripts/run_3dunet.sh --mlperf --samples=20
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
└── Command: ./scripts/run_gptj.sh --mlperf --4bit --samples=10

Llama 3.1 8B (Text Generation)
├── Dataset: CNN-DailyMail
├── Throughput: 0.44 tokens/sec
├── Mode: GPU + CPU offload (FP16)
├── Note: Slow due to 12GB VRAM limitation
└── Command: ./scripts/run_llama.sh --mlperf --offload --samples=5

Mixtral-8x7B (Text Generation - MoE)
├── Dataset: OpenOrca
├── Throughput: 0.03 tokens/sec
├── ROUGE-L: 0.538
├── Mode: GPU + CPU offload (FP16)
├── Note: Very slow - 93GB model on 12GB GPU
└── Command: ./scripts/run_mixtral.sh --mlperf --offload --samples=1
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
└── Command: ./scripts/run_dlrm.sh --mlperf --samples=1000

SDXL (Image Generation)
├── Dataset: COCO 2014 Captions
├── Throughput: 4.9 images/min
├── Latency: 12.2 sec/image
├── Mode: GPU + CPU offload
└── Command: ./scripts/run_sdxl.sh --mlperf --offload --samples=5

Whisper Large-v3 (Speech Recognition)
├── Dataset: LibriSpeech test-clean
├── Speed: 119.3x realtime
├── WER: 2.8%
├── Mode: GPU (full)
└── Command: ./scripts/run_whisper.sh --mlperf --samples=50
```

</details>

### v6.0 Benchmark Support

v6.0 removes RetinaNet and GPT-J, updates DLRM to v3.

| Benchmark | v5.1 | v6.0 | Notes           |
|-----------|:----:|:----:|-----------------|
| BERT      | ✅   | ✅   | Same            |
| ResNet50  | ✅   | ✅   | Same            |
| RetinaNet | ✅   | ❌   | Removed in v6.0 |
| 3D-UNet   | ✅   | ✅   | Same            |
| DLRM      | v2   | v3   | Updated model   |
| GPT-J     | ✅   | ❌   | Removed in v6.0 |
| Llama     | ✅   | ✅   | Same            |
| Mixtral   | ✅   | ✅   | Same            |
| SDXL      | ✅   | ✅   | Same            |
| Whisper   | ✅   | ✅   | Same            |

---

## Feature Matrix

| Benchmark | --gpu | --cpu | --offload | --4bit | --8bit | --data | --samples | --mlperf |
|-----------|:-----:|:-----:|:---------:|:------:|:------:|:------:|:---------:|:--------:|
| BERT      | ✓     | ✓     | ✓         | -      | -      | ✓      | ✓         | ✓        |
| ResNet50  | ✓     | ✓     | ✓         | -      | -      | ✓      | ✓         | ✓        |
| RetinaNet | ✓     | ✓     | ✓         | -      | -      | ✓      | ✓         | ✓        |
| 3D-UNet   | ✓     | ✓     | ✓         | -      | -      | ✓      | ✓         | ✓        |
| DLRM      | ✓     | ✓     | ✓         | -      | -      | ✓      | ✓         | ✓        |
| GPT-J     | ✓     | ✓     | ✓         | ✓      | ✓      | ✓      | ✓         | ✓        |
| Llama     | ✓     | ✓     | ✓         | ✓      | ✓      | ✓      | ✓         | ✓        |
| Mistral   | ✓     | ✓     | ✓         | ✓      | ✓      | ✓      | ✓         | ✓        |
| SDXL      | ✓     | ✓     | ✓         | -      | -      | ✓      | ✓         | ✓        |
| Whisper   | ✓     | ✓     | ✓         | -      | -      | ✓      | ✓         | ✓        |

All benchmarks now use a consistent Python + Shell script architecture.

## Script Architecture

Each benchmark follows a consistent two-file pattern:

```
scripts/
├── run_<benchmark>.sh         # Shell wrapper (args, display, data download)
└── run_<benchmark>_benchmark.py  # Python implementation (model, inference)
```

**Shell scripts** handle:
- Argument parsing with consistent options
- Configuration display with color output
- Data downloading when `--data=real`
- MLPerf mode settings and warnings
- Building and executing Python commands

**Python scripts** handle:
- Model loading and configuration
- Dataset loading (synthetic or real)
- Benchmark execution with LoadGen
- Results JSON output with `mlperf_mode` and `mlperf_compliant` fields

### Standardized Variables

All shell scripts use consistent variable names:
- `MAX_EXAMPLES` - Number of samples to process
- `DATA_TYPE` - synthetic or real
- `MLPERF_MODE` - true/false for MLPerf compliance mode
- `DEVICE` - cuda or cpu

## MLPerf Compliance Mode

All benchmarks support the `--mlperf` flag for official MLPerf-compliant settings:

```bash
# Run with official MLPerf settings (auto-downloads real data)
./scripts/run_llama.sh llama3-8b --gpu --4bit --mlperf

# Run all benchmarks with MLPerf settings
./scripts/run_gptj.sh --4bit --mlperf
./scripts/run_bert.sh --gpu --mlperf
./scripts/run_resnet50.sh --gpu --mlperf
```

### What --mlperf Does

| Setting    | Without --mlperf    | With --mlperf             |
|------------|---------------------|---------------------------|
| Data       | synthetic (default) | real (auto-download)      |
| LLM tokens | 128                 | 1024 (Llama), 128 (GPT-J) |
| SDXL steps | 20                  | 20 (official)             |
| Results    | For quick testing   | Comparable to official    |

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

### Synthetic Data Warning

Using `--mlperf --data=synthetic` shows a warning:

```
╔════════════════════════════════════════════════════════════╗
║  ⚠️  SYNTHETIC DATA WITH MLPerf MODE                       ║
╠════════════════════════════════════════════════════════════╣
║  Results are NOT comparable to official MLPerf benchmarks  ║
║  For official comparison, use: --mlperf --data=real        ║
╚════════════════════════════════════════════════════════════╝
```

Results will also be clearly labeled in JSON output:
```json
{
  "mlperf_mode": true,
  "mlperf_compliant": false,  // synthetic data used
  ...
}
```

## Common Options

All benchmarks support these standardized options:

| Option             | Description                                             |
|--------------------|---------------------------------------------------------|
| `--gpu`            | Run on GPU (CUDA) - default                             |
| `--cpu`            | Run on CPU only                                         |
| `--offload`        | GPU + CPU offloading for limited VRAM                   |
| `--data=synthetic` | Use synthetic/generated data (fast, no download)        |
| `--data=real`      | Use real dataset (downloads if needed)                  |
| `--samples=N`      | Number of samples to process (internally: MAX_EXAMPLES) |
| `--mlperf`         | Use official MLPerf settings                            |
| `-h, --help`       | Show help message                                       |

### Master Runner Script

Use `run_benchmark.sh` to run any benchmark:

```bash
# Run any benchmark via the master script
./scripts/run_benchmark.sh bert --gpu --mlperf
./scripts/run_benchmark.sh llama llama3-8b --4bit --mlperf
./scripts/run_benchmark.sh whisper --gpu --samples=50

# Show help with all benchmarks and options
./scripts/run_benchmark.sh --help
```

## LLM-Specific Options

Language models (GPT-J, Llama, Mistral) support quantization:

| Option   | Description        | VRAM Reduction |
|----------|--------------------|----------------|
| `--4bit` | 4-bit quantization | ~4x            |
| `--8bit` | 8-bit quantization | ~2x            |

### Llama Model Variants

```bash
./scripts/run_llama.sh <model> [options]
```

| Model        | Size | Full VRAM | 4-bit VRAM |
|--------------|------|-----------|------------|
| `llama2-7b`  | 7B   | ~14GB     | ~4GB       |
| `llama2-13b` | 13B  | ~26GB     | ~7GB       |
| `llama2-70b` | 70B  | ~140GB    | ~35GB      |
| `llama3-8b`  | 8B   | ~16GB     | ~4GB       |
| `llama3-70b` | 70B  | ~140GB    | ~35GB      |

**Note:** Llama models require HuggingFace authentication:
```bash
export HF_TOKEN=your_token_here
```

## Benchmark Details

### 3D-UNet (Medical Image Segmentation)
```bash
./scripts/run_3dunet.sh --gpu --data=real --cases=50
```
- **Data**: KiTS19 kidney tumor dataset
- **Real data size**: ~150-500MB per case (210 cases total)

### DLRM (Recommendation)
```bash
./scripts/run_dlrm.sh --gpu --size=sample --data=real
```
- **Data**: Criteo Terabyte dataset
- **Real data size**: ~1TB (downloads day_23 for validation)
- **Sizes**: small (debug), sample, full

### GPT-J (Text Summarization)
```bash
./scripts/run_gptj.sh --gpu --4bit --data=real
```
- **Model**: EleutherAI/gpt-j-6b (~24GB full, ~6GB 4-bit)
- **Data**: CNN-DailyMail dataset

### Mistral (Text Generation)
```bash
./scripts/run_mistral.sh --gpu --4bit --data=real
```
- **Model**: mistralai/Mistral-7B-Instruct-v0.2
- **No HuggingFace token required** (open model)
- **Data**: CNN-DailyMail dataset

### RetinaNet (Object Detection)
```bash
./scripts/run_retinanet.sh --gpu --data=real
```
- **Data**: OpenImages validation subset (~2GB)

### SDXL (Image Generation)
```bash
./scripts/run_sdxl.sh --gpu --data=real
```
- **Model**: stabilityai/stable-diffusion-xl-base-1.0
- **Data**: COCO-2014 captions for prompts

### Whisper (Speech Recognition)
```bash
./scripts/run_whisper.sh --gpu --data=real
```
- **Model**: openai/whisper-large-v3
- **Data**: LibriSpeech test-clean (~350MB)

## Memory Requirements

### GPU VRAM Requirements

| Benchmark  | Minimum VRAM | Recommended  |
|------------|--------------|--------------|
| 3D-UNet    | 4GB          | 8GB          |
| DLRM       | 4GB (small)  | 24GB+ (full) |
| GPT-J      | 6GB (4-bit)  | 24GB (full)  |
| Llama3-8B  | 4GB (4-bit)  | 16GB (full)  |
| Mistral-7B | 4GB (4-bit)  | 14GB (full)  |
| RetinaNet  | 4GB          | 8GB          |
| SDXL       | 8GB          | 12GB+        |
| Whisper    | 4GB          | 8GB          |

### Using --offload for Limited VRAM

The `--offload` option enables GPU+CPU memory offloading:
- Keeps compute on GPU for speed
- Offloads some layers/weights to system RAM
- Allows running larger models on smaller GPUs

```bash
# Run 7B model on 6GB GPU
./scripts/run_mistral.sh --offload --samples=10
```

---

## Memory Offloading Deep Dive

The `--offload` flag works differently depending on the benchmark type. Understanding these differences helps optimize performance for your hardware.

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

### DLRM: Manual Hybrid GPU+CPU

DLRM is unique - it manually places specific components on different devices:

```
┌─────────────────────────────────────────────────────────────┐
│                    DLRM Architecture                        │
├─────────────────────────────────────────────────────────────┤
│  🖥️  GPU (~50MB):                                           │
│      ├── Bottom MLP (dense feature processing)              │
│      ├── Top MLP (interaction + final output)               │
│      └── Sigmoid activation                                 │
│                                                             │
│  💾 CPU (~97GB):                                            │
│      └── 26 Embedding Tables                                │
│          ├── Table 0:  40,000,000 rows × 128 dims           │
│          ├── Table 9:  40,000,000 rows × 128 dims           │
│          ├── Table 19: 40,000,000 rows × 128 dims           │
│          ├── Table 20: 40,000,000 rows × 128 dims           │
│          ├── Table 21: 40,000,000 rows × 128 dims           │
│          └── ... (26 tables total)                          │
└─────────────────────────────────────────────────────────────┘
```

**Why this split is optimal for DLRM:**

| Component        | Size  | Operation Type                         | Best Device |
|------------------|-------|----------------------------------------|-------------|
| Embedding tables | ~97GB | Simple lookups (memory-bound)          | CPU         |
| MLP layers       | ~50MB | Matrix multiplications (compute-bound) | GPU         |

- **Embedding lookups are O(1)** - given an index, return a vector. No matrix math, no activations. CPU handles this well.
- **MLPs are compute-intensive** - dense matrix multiplications benefit massively from GPU parallelism.
- **We don't transfer 97GB per batch** - only the looked-up vectors (~26MB per 2048-sample batch).

**Data flow per batch:**
```
┌─────────────────────────────────────────────────────────────┐
│  Per Batch (2048 samples):                                  │
│                                                             │
│  1. CPU: Look up 26 embeddings per sample                   │
│     → Returns: 2048 × 26 × 128 floats = ~26MB               │
│                                                             │
│  2. Transfer ~26MB embeddings → GPU (fast)                  │
│                                                             │
│  3. GPU: MLP computation (the actual heavy lifting)         │
│     → Matrix multiplies, activations, output                │
│                                                             │
│  4. GPU → CPU: Results                                      │
└─────────────────────────────────────────────────────────────┘
```

**This is how Meta runs DLRM in production** - they shard embeddings across CPUs and use GPUs for MLPs. This hybrid approach is documented in their DLRM papers and is the canonical deployment pattern.

**Alternative approaches:**

| Approach                   | Speed | Requirement                         |
|----------------------------|-------|-------------------------------------|
| Full GPU                   | 100%  | 97GB+ VRAM (A100 80GB × 2, or H100) |
| Sample model on GPU        | 100%  | 4GB VRAM (uses smaller embeddings)  |
| **Offload (our approach)** | ~60%  | 1GB VRAM + 100GB RAM                |
| Multi-GPU sharding         | ~90%  | Multiple GPUs                       |

**DLRM benchmark output with --offload:**
```
============================================================
DLRM BENCHMARK SUMMARY
============================================================
Model Size:         full
Device:             cuda
Offload Mode:       ENABLED
----------------------------------------
Memory Distribution:
  🖥️  GPU:
      - Bottom MLP (dense features)
      - Top MLP (interaction + output)
      - Memory: 0.02 GB
  💾 CPU:
      - 26 Embedding tables (12,345,678,901 params)
      - Memory: 97.2 GB
----------------------------------------
```

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

**Automatic distribution example (Llama 8B on 12GB GPU):**
```
┌─────────────────────────────────────────────────────────────┐
│  🖥️  GPU (12GB used):                                       │
│      ├── Layers 0-15 (first half of transformer)            │
│      └── LM head                                            │
│                                                             │
│  💾 CPU (4GB used):                                         │
│      └── Layers 16-31 (second half of transformer)          │
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

### Vision Models: No True Offloading

For ResNet50, RetinaNet, and 3D-UNet, `--offload` doesn't move layers to CPU because:
- Models are small enough to fit on most GPUs
- Moving layers would kill performance for no benefit

**What --offload does instead:**
- ResNet50: Skips FP16 conversion (uses FP32)
- RetinaNet: Reduces default batch size
- 3D-UNet: Reduces default batch size

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

Results are saved to `results/<benchmark>/` directory:
- JSON files with detailed metrics
- Throughput (samples/sec or tokens/sec)
- Latency statistics
- Accuracy scores (when applicable)

---

## Data Sources

### Automatic Downloads

These datasets download automatically when using `--mlperf`:

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
  4. --cpu        : Run on CPU only (very slow)
```

**Solution options:**
1. Use `--offload` for GPU+CPU memory sharing
2. Use `--4bit` or `--8bit` for LLMs
3. Reduce `--samples=N` or `--batch=N`

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
./scripts/run_llama.sh --4bit --mlperf      # Quantization only
./scripts/run_llama.sh --offload --mlperf   # Offload only (FP16)

# This will NOT work:
./scripts/run_llama.sh --4bit --offload     # ERROR!
```

The scripts will detect this and show a helpful error message.

### CUDA Not Available
If CUDA is not available, scripts fail with guidance:
```
CUDA not available. Options:
  1. Use --cpu for CPU-only mode (slow)
  2. Install CUDA and GPU drivers
```

### Model Download Issues
- Check internet connection
- For Llama: Set `HF_TOKEN` environment variable
- Use `--data=synthetic` to skip data download

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
- **Explicit flags**: Use `--gpu`, `--cpu`, or `--offload` to control behavior
- **Predictable**: Same flags produce same behavior across all benchmarks
- **Consistent**: All scripts use the same variable names, color schemes, and output formats

### Consistency Guarantees

| Aspect                | Standard                                          |
|-----------------------|---------------------------------------------------|
| Sample count variable | `MAX_EXAMPLES`                                    |
| Error handling        | `set -e` in all shell scripts                     |
| Color output          | RED, GREEN, YELLOW, CYAN, NC variables            |
| JSON output           | Always includes `mlperf_mode`, `mlperf_compliant` |
| Print summary         | Shows "MLPerf Mode: ENABLED/disabled"             |
| Exit on CUDA OOM      | Clear error message with options                  |

---

*Last updated: January 26, 2026*  
*Tested on: NVIDIA RTX 3080 Ti (12GB VRAM)*  
*Author: Mehdi Nik*
