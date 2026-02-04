# MLPerf Inference v5.1 Compliance Roadmap

> **Document Created**: January 30, 2026  
> **Author**: Mehdi Nik  
> **Purpose**: Step-by-step guide to achieve MLPerf v5.1 compliance

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Assessment](#current-state-assessment)
3. [Gap Analysis](#gap-analysis)
4. [Implementation Roadmap](#implementation-roadmap)
5. [Phase 1: LoadGen Integration](#phase-1-loadgen-integration)
6. [Phase 2: Scenario Support](#phase-2-scenario-support)
7. [Phase 3: Missing Benchmarks](#phase-3-missing-benchmarks)
8. [Phase 4: Accuracy Evaluation](#phase-4-accuracy-evaluation)
9. [Phase 5: Compliance Tests](#phase-5-compliance-tests)
10. [Reference Materials](#reference-materials)

---

## Executive Summary

### Current Compliance Score: ~25%

| Category | Score | Status |
|----------|-------|--------|
| Benchmark Coverage | 67% (10/15) | 🟡 Partial |
| LoadGen Integration | 0% | 🔴 Critical |
| Scenario Support | 0% | 🔴 Critical |
| Accuracy Evaluation | 20% | 🟡 Partial |
| Dataset Compliance | 90% | ✅ Good |
| Documentation | 85% | ✅ Good |

### Target: Official MLPerf v5.1 Submission Readiness

---

## Current State Assessment

### What We Have (✅ Working)

- [x] BERT benchmark script with SQuAD v1.1 dataset support
- [x] ResNet50 benchmark script with ImageNet support
- [x] RetinaNet benchmark script with OpenImages support
- [x] 3D-UNet benchmark script with KiTS19 support
- [x] DLRM-v2 benchmark script with Criteo support
- [x] Whisper benchmark script with LibriSpeech support
- [x] SDXL benchmark script with COCO 2014 support
- [x] Llama benchmark script with OpenOrca + CNN-DailyMail support
- [x] Mixtral benchmark script with OpenOrca support
- [x] GPT-J benchmark script (note: retired from v5.1)
- [x] Synthetic data fallback for all benchmarks
- [x] GPU memory management and cleanup utilities
- [x] Basic timing and throughput measurements
- [x] JSON result output format

### What's Missing (🔴 Critical)

- [ ] MLPerf LoadGen library integration
- [ ] Official scenario modes (Offline, SingleStream, Server, MultiStream)
- [ ] SUT (System Under Test) implementation pattern
- [ ] QSL (Query Sample Library) implementation pattern
- [ ] Official accuracy evaluation scripts
- [ ] Compliance test runners (TEST01, TEST04, TEST05, TEST06)
- [ ] 5 additional benchmarks (Llama3.1-405B, R-GAT, PointPainting, DeepSeek-R1, GPT-OSS-120B)

---

## Gap Analysis

### Benchmark Coverage Matrix

| Benchmark | Implemented | Dataset | LoadGen | Scenarios | Accuracy | Priority |
|-----------|-------------|---------|---------|-----------|----------|----------|
| BERT | ✅ | ✅ SQuAD v1.1 | ❌ | ❌ | ❌ | P1 |
| ResNet50 | ✅ | ✅ ImageNet | ❌ | ❌ | ❌ | P1 |
| RetinaNet | ✅ | ✅ OpenImages | ❌ | ❌ | ❌ | P1 |
| 3D-UNet | ✅ | ✅ KiTS19 | ❌ | ❌ | ❌ | P2 |
| DLRM-v2 | ✅ | ✅ Criteo | ❌ | ❌ | ❌ | P2 |
| Whisper | ✅ | ✅ LibriSpeech | ❌ | ❌ | ❌ | P1 |
| SDXL | ✅ | ✅ COCO 2014 | ❌ | ❌ | ❌ | P2 |
| Llama2-70B | ✅ | ✅ OpenOrca | ❌ | ❌ | ❌ | P1 |
| Llama3.1-8B | ✅ | ✅ CNN-DailyMail | ❌ | ❌ | ❌ | P1 |
| Llama3.1-405B | ❌ | - | ❌ | ❌ | ❌ | P3 |
| Mixtral-8x7B | ✅ | ✅ OpenOrca | ❌ | ❌ | ❌ | P2 |
| R-GAT | ❌ | - | ❌ | ❌ | ❌ | P3 |
| PointPainting | ❌ | - | ❌ | ❌ | ❌ | P3 |
| DeepSeek-R1 | ❌ | - | ❌ | ❌ | ❌ | P3 |
| GPT-OSS-120B | ❌ | - | ❌ | ❌ | ❌ | P3 |

### ⚠️ GPT-J Status

**GPT-J was REMOVED from MLPerf after v4.1** - our v5.1 scripts include it but it's NOT part of the official v5.1 specification. Should be moved to `retired/` or removed.

---

## Implementation Roadmap

### Timeline Overview

```
Phase 1: LoadGen Integration     [Week 1-2]  ████████░░░░░░░░
Phase 2: Scenario Support        [Week 2-3]  ░░░░░░░░████░░░░
Phase 3: Missing Benchmarks      [Week 3-5]  ░░░░░░░░░░░░████
Phase 4: Accuracy Evaluation     [Week 4-5]  ░░░░░░░░░░░░░░██
Phase 5: Compliance Tests        [Week 5-6]  ░░░░░░░░░░░░░░░░
```

---

## Phase 1: LoadGen Integration

### 1.1 Install LoadGen

```bash
# Option A: Install from PyPI
pip install mlperf_loadgen

# Option B: Build from source (recommended for latest)
cd v5.1/inference/loadgen
pip install .
```

**Status**: [ ] Not Started / [ ] In Progress / [ ] Complete

### 1.2 Create Base SUT Class

Create a reusable System Under Test (SUT) base class:

**File**: `v5.1/scripts/utils/base_sut.py`

```python
import mlperf_loadgen as lg
from abc import ABC, abstractmethod

class BaseSUT(ABC):
    """Base System Under Test for MLPerf benchmarks."""
    
    def __init__(self, args):
        self.args = args
        self.qsl = None
        self.sut = None
        
    @abstractmethod
    def load_model(self):
        """Load the model for inference."""
        pass
    
    @abstractmethod
    def issue_queries(self, query_samples):
        """Process queries from LoadGen."""
        pass
    
    def flush_queries(self):
        """Flush pending queries."""
        pass
    
    def construct_sut(self):
        """Construct the SUT with LoadGen."""
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut
    
    def destroy(self):
        """Clean up resources."""
        if self.sut:
            lg.DestroySUT(self.sut)
        if self.qsl:
            lg.DestroyQSL(self.qsl)
```

**Status**: [ ] Not Started / [ ] In Progress / [ ] Complete

### 1.3 Create Base QSL Class

Create a reusable Query Sample Library (QSL) base class:

**File**: `v5.1/scripts/utils/base_qsl.py`

```python
import mlperf_loadgen as lg
from abc import ABC, abstractmethod

class BaseQSL(ABC):
    """Base Query Sample Library for MLPerf benchmarks."""
    
    def __init__(self, dataset_path, total_count, perf_count):
        self.dataset_path = dataset_path
        self.total_count = total_count
        self.perf_count = perf_count
        self.qsl = None
        self.loaded_samples = {}
        
    @abstractmethod
    def load_query_samples(self, sample_list):
        """Load samples into memory."""
        pass
    
    @abstractmethod
    def unload_query_samples(self, sample_list):
        """Unload samples from memory."""
        pass
    
    @abstractmethod
    def get_sample(self, sample_id):
        """Get a loaded sample by ID."""
        pass
    
    def construct_qsl(self):
        """Construct the QSL with LoadGen."""
        self.qsl = lg.ConstructQSL(
            self.total_count,
            self.perf_count,
            self.load_query_samples,
            self.unload_query_samples
        )
        return self.qsl
```

**Status**: [ ] Not Started / [ ] In Progress / [ ] Complete

### 1.4 Refactor First Benchmark (BERT)

Convert `run_bert_benchmark.py` to use LoadGen:

**Tasks**:
- [ ] Create `bert_sut.py` implementing `BaseSUT`
- [ ] Create `squad_qsl.py` implementing `BaseQSL`
- [ ] Update `run_bert_benchmark.py` to use LoadGen flow
- [ ] Test with synthetic data
- [ ] Test with real SQuAD data

**Status**: [ ] Not Started / [ ] In Progress / [ ] Complete

### 1.5 Validate LoadGen Output

Verify correct LoadGen log files are generated:

- [ ] `mlperf_log_summary.txt`
- [ ] `mlperf_log_detail.txt`
- [ ] `mlperf_log_accuracy.json` (accuracy mode)

---

## Phase 2: Scenario Support

### 2.1 Scenario Definitions

| Scenario | Mode | Key Metric | Use Case |
|----------|------|------------|----------|
| **Offline** | Batch all samples | Samples/second | Maximum throughput |
| **SingleStream** | One at a time | 90th %ile latency | Edge devices |
| **MultiStream** | N samples/query | 99th %ile latency | Multi-camera |
| **Server** | Poisson arrival | QPS under latency | Datacenter |

### 2.2 Required Scenarios per Benchmark

| Benchmark | Edge | Datacenter |
|-----------|------|------------|
| BERT | Offline | Server |
| ResNet50 | Offline, SingleStream | - |
| RetinaNet | Offline, MultiStream | Server |
| Whisper | Offline, SingleStream | Server |
| Llama2-70B | - | Offline, Server |
| SDXL | Offline | Server |

### 2.3 Implementation Tasks

- [ ] Add `--scenario` argument to all benchmark scripts
- [ ] Implement Offline scenario runner
- [ ] Implement SingleStream scenario runner
- [ ] Implement Server scenario runner
- [ ] Implement MultiStream scenario runner (if needed)
- [ ] Add scenario-specific configuration loading from `mlperf.conf`

**Status**: [ ] Not Started / [ ] In Progress / [ ] Complete

---

## Phase 3: Missing Benchmarks

### 3.1 Existing Benchmarks - Add LoadGen Support

These benchmarks have working scripts but need LoadGen integration:

#### Llama (run_llama_benchmark.py)
- **Supports**: Llama2-70B, Llama3.1-8B, and other Llama variants
- **Datasets**: ✅ OpenOrca, ✅ CNN-DailyMail
- **Status**: Script works, needs LoadGen refactor

**Tasks**:
- [ ] Refactor `run_llama_benchmark.py` to use LoadGen SUT/QSL pattern
- [ ] Add token latency metrics (TTFT, TPOT) for Server scenario
- [ ] Support edge (5,000 samples) and datacenter (13,368+ samples) modes
- [ ] Implement accuracy evaluation with Rouge scores

#### Mixtral-8x7B (run_mixtral_benchmark.py)
- **Dataset**: OpenOrca, MBXP, GSM8K
- **Samples**: 15,000
- **Status**: Script works, needs LoadGen refactor

**Tasks**:
- [ ] Refactor to use LoadGen SUT/QSL pattern
- [ ] Add multi-dataset evaluation (MBXP, GSM8K)
- [ ] Implement accuracy evaluation

### 3.2 Priority 3 - New Benchmarks (Not Yet Implemented)

| Benchmark | Complexity | Notes |
|-----------|------------|-------|
| Llama3.1-405B | Very High | Requires multi-GPU/node, 8x A100 minimum |
| R-GAT | Medium | Graph neural network on IGBH dataset |
| PointPainting | High | 3D object detection, Waymo Open Dataset |
| DeepSeek-R1 | High | Reasoning benchmark (AIME, MATH500) |
| GPT-OSS-120B | Very High | New in v5.1, large model |

**Note**: These 5 benchmarks are lower priority for initial compliance. Focus on getting LoadGen working with existing 10 benchmarks first.

**Status**: [ ] Not Started / [ ] In Progress / [ ] Complete

---

## Phase 4: Accuracy Evaluation

### 4.1 Official Accuracy Targets

| Benchmark | Metric | Reference Score | 99% Target | 99.9% Target |
|-----------|--------|-----------------|------------|--------------|
| BERT | F1 Score | 90.874% | 89.96% | 90.78% |
| ResNet50 | Top-1 Acc | 76.46% | 75.69% | - |
| RetinaNet | mAP | 37.55% | 37.17% | - |
| 3D-UNet | DICE | 86.17% | 85.30% | 86.08% |
| DLRM-v2 | AUC | 80.31% | 79.50% | 80.23% |
| Whisper | WER | 2.07% | ≤2.09% | - |
| Llama2-70B | Rouge-L | 28.61% | 28.33% | - |
| Mixtral-8x7B | Rouge-L | 30.46% | 30.16% | - |

### 4.2 Accuracy Scripts to Implement

Copy and adapt from official repo:

- [ ] `accuracy_bert.py` - F1 score calculation
- [ ] `accuracy_resnet50.py` - Top-1/Top-5 accuracy
- [ ] `accuracy_retinanet.py` - mAP calculation
- [ ] `accuracy_3dunet.py` - DICE coefficient
- [ ] `accuracy_dlrm.py` - AUC-ROC
- [ ] `accuracy_whisper.py` - Word Error Rate
- [ ] `accuracy_llama.py` - Rouge scores
- [ ] `accuracy_sdxl.py` - CLIP/FID scores

### 4.3 Implementation Pattern

```python
def evaluate_accuracy(mlperf_log_path, reference_data_path):
    """
    Parse mlperf_log_accuracy.json and compute accuracy metrics.
    
    Returns:
        dict: {"metric_name": value, "passed": bool}
    """
    with open(mlperf_log_path) as f:
        predictions = json.load(f)
    
    # Compute metric
    score = compute_metric(predictions, reference_data_path)
    target = get_accuracy_target(benchmark_name)
    
    return {
        "score": score,
        "target": target,
        "passed": score >= target
    }
```

**Status**: [ ] Not Started / [ ] In Progress / [ ] Complete

---

## Phase 5: Compliance Tests

### 5.1 Required Compliance Tests

| Test | Purpose | When Required |
|------|---------|---------------|
| TEST01 | Accuracy check | All submissions |
| TEST04 | Performance stability | Performance runs |
| TEST05 | RNG seed verification | All submissions |
| TEST06 | Token count verification | LLM benchmarks |

### 5.2 TEST01 - Accuracy Verification

Verifies accuracy mode produces valid results within threshold.

```bash
# Run compliance test
python compliance/TEST01/run_verification.py \
    --results_dir /path/to/results \
    --compliance_dir /path/to/compliance/TEST01 \
    --output_dir /path/to/output
```

### 5.3 TEST04 - Performance Stability

Verifies performance is stable across runs.

### 5.4 TEST05 - RNG Seed Verification

Verifies RNG seeds are properly set.

### 5.5 TEST06 - Token Count (LLMs only)

Verifies token counts match expected values.

**Status**: [ ] Not Started / [ ] In Progress / [ ] Complete

---

## Reference Materials

### Official Documentation

- **MLPerf Inference Docs**: https://docs.mlcommons.org/inference/
- **GitHub Repo**: https://github.com/mlcommons/inference
- **v5.1.1 Tag**: https://github.com/mlcommons/inference/releases/tag/v5.1.1
- **Submission Guidelines**: https://github.com/mlcommons/inference/blob/master/Submission_Guidelines.md

### Key Files in Official Repo

```
inference/
├── loadgen/                    # LoadGen library source
├── language/
│   ├── bert/
│   │   ├── pytorch_SUT.py      # Reference SUT
│   │   ├── squad_QSL.py        # Reference QSL
│   │   └── accuracy-squad.py   # Accuracy evaluation
│   ├── llama2-70b/
│   ├── llama3.1-8b/
│   └── mixtral-8x7b/
├── vision/
│   ├── classification_and_detection/
│   └── medical_imaging/
├── speech2text/                # Whisper
├── text_to_image/              # SDXL
├── compliance/                 # Compliance tests
│   ├── TEST01/
│   ├── TEST04/
│   ├── TEST05/
│   └── TEST06/
└── tools/
    └── submission/             # Submission checker
```

### Configuration Files

- `mlperf.conf` - Official MLPerf configuration
- `user.conf` - User overrides (target QPS, etc.)
- `audit.conf` - Compliance test configuration

---

## Progress Tracking

### Weekly Checklist

#### Week 1
- [ ] Install LoadGen library
- [ ] Create base SUT/QSL classes
- [ ] Convert BERT to LoadGen

#### Week 2
- [ ] Add Offline scenario support
- [ ] Add SingleStream scenario support
- [ ] Convert ResNet50 and Whisper to LoadGen

#### Week 3
- [ ] Add Server scenario support
- [ ] Implement Llama2-70B benchmark
- [ ] Implement Llama3.1-8B benchmark

#### Week 4
- [ ] Refactor Mixtral with LoadGen
- [ ] Add accuracy evaluation scripts
- [ ] Run TEST01 compliance

#### Week 5
- [ ] Convert remaining benchmarks
- [ ] Run full compliance suite
- [ ] Documentation and cleanup

#### Week 6
- [ ] Final testing
- [ ] Prepare submission package
- [ ] Review and validation

---

## Notes

### Hardware Requirements

| Benchmark | Min VRAM | Recommended |
|-----------|----------|-------------|
| BERT | 8GB | 16GB |
| ResNet50 | 4GB | 8GB |
| RetinaNet | 8GB | 16GB |
| Whisper | 8GB | 16GB |
| Llama2-70B | 140GB+ | 8x A100 80GB |
| Llama3.1-8B | 16GB | 24GB |
| Mixtral-8x7B | 96GB+ | 4x A100 |
| SDXL | 16GB | 24GB |

### Quantization Options (for limited hardware)

- **4-bit**: ~4x VRAM reduction, slight accuracy loss
- **8-bit**: ~2x VRAM reduction, minimal accuracy loss
- **Note**: Quantized models may not meet accuracy targets

---

## Changelog

| Date | Change |
|------|--------|
| 2026-01-30 | Initial document created |

