#!/bin/bash
# ============================================================================
# MLPerf Benchmark Setup and Runner - Master Runner Script
#
# Author: Mehdi Nik
# Created: Jan 2026
#
# DISCLAIMER:
# This repository is NOT an official MLPerf implementation.
# It contains personal tooling and scripts to run MLPerf workloads.
#
# This software is provided "as is" without warranty of any kind, express or
# implied. The author assumes no responsibility for errors, omissions, or
# damages arising from its use.
#
# All rights reserved. If you use or reference this work, please provide
# attribution to the original author.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_usage() {
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║          MLPerf Inference v5.1 Benchmark Runner            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Usage: $0 <benchmark> [options]"
    echo ""
    echo "Benchmarks:"
    echo "  bert        - BERT-Large (SQuAD v1.1)"
    echo "  resnet50    - ResNet-50 (ImageNet)"
    echo "  retinanet   - RetinaNet (OpenImages)"
    echo "  3dunet      - 3D-UNet (KiTS19)"
    echo "  dlrm        - DLRM-v2 (Criteo)"
    echo "  gptj        - GPT-J 6B (CNN/DailyMail)"
    echo "  llama       - Llama models (various sizes)"
    echo "  mixtral     - Mixtral-8x7B"
    echo "  sdxl        - Stable Diffusion XL"
    echo "  whisper     - Whisper (LibriSpeech)"
    echo ""
    echo "Common Options (all benchmarks):"
    echo "  --gpu           Run on GPU (default)"
    echo "  --cpu           Run on CPU"
    echo "  --samples=N     Number of samples"
    echo "  --mlperf        Use official MLPerf settings"
    echo "  --data=TYPE     Data type: synthetic, real"
    echo "  --download=M    Download method: hf (default), wget"
    echo "  -h, --help      Show help"
    echo ""
    echo "LLM Options (gptj, llama, mixtral):"
    echo "  --offload       GPU+CPU memory offloading"
    echo "  --4bit          4-bit quantization"
    echo "  --8bit          8-bit quantization"
    echo ""
    echo "MLPerf Compliance:"
    echo "  --mlperf              Auto-downloads real data, uses official settings"
    echo "  --mlperf --data=real  Full compliance mode"
    echo ""
    echo "Examples:"
    echo "  $0 bert --gpu --samples=100"
    echo "  $0 bert --mlperf                  # Official MLPerf mode"
    echo "  $0 mixtral --gpu --4bit"
    echo "  $0 llama llama3-8b --gpu --offload"
    echo "  $0 sdxl --gpu --data=real"
    echo ""
}

if [[ $# -lt 1 ]]; then
    print_usage
    exit 1
fi

BENCHMARK="$1"
shift

case "$BENCHMARK" in
    bert)
        "${SCRIPT_DIR}/run_bert.sh" "$@"
        ;;
    resnet50)
        "${SCRIPT_DIR}/run_resnet50.sh" "$@"
        ;;
    retinanet)
        "${SCRIPT_DIR}/run_retinanet.sh" "$@"
        ;;
    gptj)
        "${SCRIPT_DIR}/run_gptj.sh" "$@"
        ;;
    llama|llama2|llama3)
        "${SCRIPT_DIR}/run_llama.sh" "$@"
        ;;
    mixtral)
        "${SCRIPT_DIR}/run_mixtral.sh" "$@"
        ;;
    sdxl)
        "${SCRIPT_DIR}/run_sdxl.sh" "$@"
        ;;
    whisper)
        "${SCRIPT_DIR}/run_whisper.sh" "$@"
        ;;
    3dunet)
        "${SCRIPT_DIR}/run_3dunet.sh" "$@"
        ;;
    dlrm)
        "${SCRIPT_DIR}/run_dlrm.sh" "$@"
        ;;
    --help|-h)
        print_usage
        ;;
    *)
        echo "Error: Unknown benchmark '$BENCHMARK'"
        echo ""
        print_usage
        exit 1
        ;;
esac
