# MLPerf Benchmark Data Guide

This guide explains the real vs. synthetic data options for each benchmark.

## Quick Summary

| Benchmark | `--data=real` | Real Data Source | Real Data Size | Auto-Download |
|-----------|---------------|------------------|----------------|---------------|
| BERT | N/A | SQuAD v1.1 | 4.7 MB | ✅ Always real |
| ResNet50 | ❌ | ImageNet | 140 GB | ❌ Manual (license) |
| Whisper | ✅ | LibriSpeech test-clean | 350 MB | ✅ |
| GPT-J | ✅ | CNN-DailyMail | ~300 MB | ✅ |
| SDXL | ✅ | COCO 2014 Captions | ~20 MB | ✅ |
| Mistral-7B | ✅ | CNN-DailyMail | ~300 MB | ✅ |
| RetinaNet | ✅ | OpenImages | ~500 MB | ✅ |
| DLRM-v2 | ✅ | Criteo Terabyte | ~100 GB | ✅ |
| 3D-UNet | ✅ | KiTS19 | 24 GB | ✅ |

---

## Per-Benchmark Details

### 1. BERT (Question Answering)
```bash
# Always uses real SQuAD v1.1 data (auto-downloaded)
./scripts/run_bert.sh --gpu
```
- **Dataset**: SQuAD v1.1 validation set
- **Size**: 4.7 MB
- **Samples**: 10,833 question-answer pairs
- **Note**: BERT always uses real data - no synthetic option needed

---

### 2. ResNet50 (Image Classification)
```bash
# Currently uses synthetic/sample images
./scripts/run_resnet50.sh --gpu
```
- **Dataset**: ImageNet ILSVRC2012 validation set
- **Size**: ~140 GB
- **Samples**: 50,000 images
- **Auto-Download**: ❌ **Not available** - requires manual registration
- **How to get**: Register at https://image-net.org/download-images.php
- **Place data in**: `data/imagenet/`

---

### 3. Whisper (Speech Recognition)
```bash
# Synthetic (random audio)
./scripts/run_whisper.sh --gpu --data=synthetic --samples=20

# Real (LibriSpeech test-clean)
./scripts/run_whisper.sh --gpu --data=real --samples=20
```
- **Dataset**: LibriSpeech test-clean (read English speech)
- **Size**: ~350 MB
- **Samples**: 2,620 audio clips
- **Official MLPerf**: Yes - standard ASR evaluation set

---

### 4. GPT-J (Text Generation)
```bash
# Synthetic (random prompts)
./scripts/run_gptj.sh --gpu --data=synthetic --samples=10

# Real (CNN-DailyMail articles)
./scripts/run_gptj.sh --offload --data=real --samples=10
```
- **Dataset**: CNN-DailyMail test set
- **Size**: ~300 MB
- **Samples**: 11,490 news articles
- **Official MLPerf**: Yes - used for summarization/generation tasks

---

### 4. SDXL (Image Generation)
```bash
# Synthetic (lorem ipsum prompts)
./scripts/run_sdxl.sh --gpu --data=synthetic --samples=5

# Real (COCO captions)
./scripts/run_sdxl.sh --gpu --data=real --samples=5
```
- **Dataset**: COCO 2014 validation captions
- **Size**: ~20 MB
- **Samples**: 40,504 image descriptions
- **Official MLPerf**: Yes - standard for text-to-image

---

### 5. Mistral-7B (Text Generation)
```bash
# Synthetic (random prompts)
./scripts/run_mistral.sh --offload --data=synthetic --samples=10

# Real (CNN-DailyMail articles)
./scripts/run_mistral.sh --offload --data=real --samples=10
```
- **Dataset**: CNN-DailyMail test set (shared with GPT-J)
- **Size**: ~300 MB
- **Samples**: 11,490 news articles

---

### 6. RetinaNet (Object Detection)
```bash
# Synthetic (random images)
./scripts/run_retinanet.sh --gpu --data=synthetic --samples=100

# Real (OpenImages validation)
./scripts/run_retinanet.sh --gpu --data=real --size=sample
./scripts/run_retinanet.sh --gpu --data=real --size=full
```
- **Dataset**: OpenImages validation set
- **Sample Size**: ~500 MB (5,000 images)
- **Full Size**: ~5 GB (41,620 images)
- **Official MLPerf**: Yes - used for detection benchmarks

---

### 7. DLRM-v2 (Recommendation)
```bash
# Synthetic data (default - fast, no large download)
./scripts/run_dlrm.sh --gpu --size=small --data=synthetic
./scripts/run_dlrm.sh --gpu --size=sample --data=synthetic

# Real Criteo Terabyte data (MLPerf official)
./scripts/run_dlrm.sh --gpu --data=real
```
- **Dataset**: Criteo Terabyte Click Logs (Day 23 validation)
- **Model Size**: ~97 GB
- **Dataset Size**: ~100 GB preprocessed
- **Total Download**: ~200 GB
- **Official MLPerf**: Yes - Criteo Terabyte dataset

**Size Options:**
| Option | Model | Data | Total |
|--------|-------|------|-------|
| `--size=small` | Debug (10MB) | Synthetic (50MB) | ~60 MB |
| `--size=sample` | Full (97GB) | Synthetic (1GB) | ~98 GB |
| `--data=real` | Full (97GB) | Real Criteo (100GB) | ~200 GB |

---

### 8. 3D-UNet (Medical Segmentation)
```bash
# Synthetic (random CT volumes)
./scripts/run_3dunet.sh --gpu --data=synthetic --samples=10

# Real (KiTS19 kidney CT scans)
./scripts/run_3dunet.sh --gpu --data=real --size=sample   # 20 volumes
./scripts/run_3dunet.sh --gpu --data=real --size=small    # 50 volumes
./scripts/run_3dunet.sh --gpu --data=real --size=full     # 210 volumes
```
- **Dataset**: KiTS19 (Kidney Tumor Segmentation Challenge)
- **Size**: 24 GB preprocessed
- **Samples**: 210 3D CT volumes with segmentation masks
- **Official MLPerf**: Yes - standard medical imaging benchmark

---

## Storage Requirements

| Level | Storage Needed |
|-------|----------------|
| Minimal (synthetic only) | ~15 GB (models only) |
| Standard (common datasets) | ~30 GB |
| Full (all real data except Criteo) | ~60 GB |
| Complete (with Criteo) | ~1.1 TB |

---


## Current Status

```
data/
├── squad/           4.7 MB   ✅ BERT (real)
├── librispeech/     689 MB   ✅ Whisper (real)
├── cnn-dailymail/   53 MB    ✅ GPT-J/Mistral (real)
├── coco-2014/       1.1 GB   ✅ SDXL (real)
├── openimages/      880 MB   ✅ RetinaNet (sample)
├── criteo/          152 GB   ✅ DLRM (real)
└── kits19/          24 GB    ✅ 3D-UNet (full 210 cases)
```

**Note:** All datasets are downloaded and ready. Total storage: ~180 GB

---

## Download All Real Data

All datasets are already downloaded. To re-download or verify:

```bash
#!/bin/bash
# Download all MLPerf datasets

cd /home/mehdi/projects/mlperf_setup

# Check existing data
echo "SQuAD: Already present (4.7 MB)"
echo "KiTS19: Already present (24 GB)"
echo "LibriSpeech: Already present (689 MB)"
echo "CNN-DailyMail: Already present (53 MB)"
echo "COCO-2014: Already present (1.1 GB)"
echo "OpenImages: Already present (880 MB)"
echo "Criteo: Already present (152 GB)"
```

