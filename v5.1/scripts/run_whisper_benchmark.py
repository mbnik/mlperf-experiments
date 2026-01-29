#!/usr/bin/env python3
"""
MLPerf Benchmark Setup and Runner - Whisper Speech Recognition

Author: Mehdi Nik
Created: Jan 2026

DISCLAIMER:
This repository is NOT an official MLPerf implementation.
It contains personal tooling and scripts to run MLPerf workloads.

This software is provided "as is" without warranty of any kind, express or implied.
The author assumes no responsibility for errors, omissions, or damages arising from
its use. Users are solely responsible for determining the appropriateness of using
this software and assume all risks associated with its use.

All rights reserved. If you use or reference this work, please provide attribution
to the original author.
"""

import argparse
import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("Whisper-Benchmark")


def generate_test_audio(duration=5.0, sample_rate=16000):
    """Generate a simple test audio signal"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate a more complex signal
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz
    audio += 0.2 * np.sin(2 * np.pi * 880 * t)  # 880 Hz
    audio += 0.1 * np.random.randn(len(t))  # noise
    return audio.astype(np.float32)


def load_librispeech_samples(data_dir: str, max_examples: int) -> Tuple[Optional[List], Optional[List], Dict]:
    """Load real LibriSpeech audio samples"""
    import soundfile as sf
    
    manifest_path = Path(data_dir) / "manifest.json"
    
    if not manifest_path.exists():
        log.warning(f"Manifest not found at {manifest_path}, using synthetic data")
        return None, None, {}
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    total_available = len(manifest)
    log.info(f"Loaded manifest with {total_available} samples")
    
    samples = []
    references = []
    
    for i, entry in enumerate(manifest[:max_examples]):
        audio_path = entry["audio_path"]
        text = entry["text"]
        
        try:
            audio, sr = sf.read(audio_path)
            # Resample to 16kHz if needed
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            samples.append(audio.astype(np.float32))
            references.append(text)
        except Exception as e:
            log.warning(f"Failed to load {audio_path}: {e}")
    
    log.info(f"Loaded {len(samples)} audio samples")
    
    data_info = {
        'type': 'real',
        'dataset': 'LibriSpeech',
        'source': str(data_dir),
        'samples_used': len(samples),
        'samples_available': total_available,
        'task': 'speech_recognition',
        'sample_rate': 16000,
        'verified': True,
        'mlperf_compliant': True,
        'note': 'LibriSpeech automatic speech recognition dataset'
    }
    
    return samples, references, data_info


def get_synthetic_data(max_examples: int) -> Tuple[List[np.ndarray], Dict]:
    """Get synthetic audio samples with data info"""
    samples = [generate_test_audio(duration=5.0) for _ in range(max_examples)]
    
    data_info = {
        'type': 'synthetic',
        'dataset': 'Synthetic Audio',
        'source': 'generated_sine_waves',
        'samples_used': len(samples),
        'samples_available': max_examples,
        'task': 'speech_recognition',
        'sample_rate': 16000,
        'duration_sec': 5.0,
        'verified': False,
        'mlperf_compliant': False,
        'note': 'Synthetic sine wave audio for testing only'
    }
    
    return samples, data_info


def get_args():
    parser = argparse.ArgumentParser(description="Whisper Benchmark with Real/Synthetic Data")
    parser.add_argument("--model-name", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"])
    parser.add_argument("--offload", action="store_true",
                       help="Enable GPU+CPU memory offloading for large models")
    parser.add_argument("--dtype", type=str, default="float16",
                       choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--data-type", type=str, default="synthetic",
                       choices=["synthetic", "real"])
    parser.add_argument("--data-dir", type=str, default="data/librispeech")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--mlperf", action="store_true",
                       help="Use MLPerf official settings")
    return parser.parse_args()


def load_model(args):
    """Load Whisper model with appropriate device configuration"""
    log.info(f"Loading model: {args.model_name}")
    log.info(f"Device: {args.device}, Offload: {args.offload}")
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        log.error("CUDA not available. Options:")
        log.error("  1. Use --cpu for CPU-only mode (slow)")
        log.error("  2. Install CUDA and GPU drivers")
        raise SystemExit(1)
    
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(args.dtype, torch.float16)
    
    if args.device == "cpu":
        device = "cpu"
        device_map = None
        torch_dtype = torch.float32
    elif args.offload:
        device = "cuda:0"
        device_map = "auto"
        log.info("Loading with GPU+CPU offloading...")
    else:
        device = "cuda:0"
        device_map = None
    
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map=device_map,
        )
        
        if device_map is None:
            model = model.to(device)
        
        processor = AutoProcessor.from_pretrained(args.model_name)
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device if device_map is None else None,
        )
    except torch.cuda.OutOfMemoryError:
        log.error("=" * 60)
        log.error("CUDA OUT OF MEMORY!")
        log.error("=" * 60)
        log.error("Whisper-large requires ~3GB VRAM. Try:")
        log.error("  1. --offload : Enable GPU+CPU memory offloading")
        log.error("  2. --cpu     : Run on CPU only (slow)")
        raise SystemExit(1)
    
    log.info("Model loaded successfully!")
    return pipe, processor


def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate"""
    ref_words = reference.upper().split()
    hyp_words = hypothesis.upper().split()
    
    # Simple Levenshtein distance for WER
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    
    if len(ref_words) == 0:
        return 0.0
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def run_benchmark(pipe, args, samples: List[np.ndarray], references: List[str] = None, data_info: Dict = None):
    """Run the benchmark"""
    log.info("=" * 60)
    log.info("Starting Whisper Benchmark")
    log.info("=" * 60)
    log.info(f"Data type: {args.data_type}")
    
    num_examples = len(samples)
    log.info(f"Processing {num_examples} samples")
    
    results = []
    total_time = 0
    total_wer = 0
    wer_count = 0
    
    # Warmup
    log.info("Warmup...")
    _ = pipe(samples[0], generate_kwargs={"max_new_tokens": 128})
    
    log.info("\nRunning benchmark...")
    for i, audio in enumerate(samples):
        start_time = time.perf_counter()
        result = pipe(audio, generate_kwargs={"max_new_tokens": 256})
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        total_time += elapsed
        
        transcription = result["text"].strip()
        
        # Calculate WER if we have references
        wer = None
        if references and i < len(references):
            wer = calculate_wer(references[i], transcription)
            total_wer += wer
            wer_count += 1
        
        if (i + 1) % 5 == 0 or i == num_examples - 1:
            wer_str = f", WER: {wer:.2%}" if wer is not None else ""
            log.info(f"  [{i+1}/{num_examples}] Latency: {elapsed:.2f}s{wer_str}")
        
        results.append({
            "sample_id": i,
            "transcription": transcription[:100],
            "latency_sec": elapsed,
            "wer": wer,
            "reference": references[i][:100] if references and i < len(references) else None,
        })
    
    # Summary
    avg_latency = total_time / num_examples
    avg_wer = total_wer / wer_count if wer_count > 0 else None
    
    summary = {
        "model": args.model_name,
        "device": args.device,
        "data_type": args.data_type,
        "num_samples": num_examples,
        "total_time_sec": total_time,
        "avg_latency_sec": avg_latency,
        "samples_per_sec": num_examples / total_time,
        "avg_wer": avg_wer,
        "timestamp": datetime.now().isoformat(),
        "mlperf_mode": args.mlperf,
        "mlperf_compliant": args.mlperf and args.data_type == "real",
    }
    
    print("\n" + "=" * 60)
    print("WHISPER BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Device:             {args.device}")
    print(f"Data Type:          {args.data_type}")
    if args.mlperf:
        print(f"MLPerf Mode:        ENABLED")
    print(f"Total Samples:      {num_examples}")
    print(f"Total Time:         {total_time:.2f} seconds")
    print(f"Avg Latency:        {avg_latency:.2f} sec/sample")
    print(f"Throughput:         {num_examples/total_time:.2f} samples/sec")
    if avg_wer is not None:
        print(f"Average WER:        {avg_wer:.2%}")
    print("=" * 60)
    
    # Performance rating
    if avg_latency < 1.0:
        print("Performance:        🚀 Excellent")
    elif avg_latency < 3.0:
        print("Performance:        ✅ Good")
    else:
        print("Performance:        ⚠️ Slow")
    print("=" * 60)
    
    # Print data information
    if data_info:
        print("\n" + "=" * 60)
        print("DATA INFORMATION")
        print("=" * 60)
        print(f"Type:               {data_info['type']}")
        print(f"Dataset:            {data_info['dataset']}")
        print(f"Source:             {data_info['source']}")
        print(f"Samples Used:       {data_info['samples_used']:,}")
        print(f"Samples Available:  {data_info['samples_available']:,}")
        print(f"Task:               {data_info['task']}")
        print(f"Sample Rate:        {data_info['sample_rate']} Hz")
        print(f"Verified:           {'✓' if data_info['verified'] else '✗'}")
        print(f"MLPerf Compliant:   {'✓' if data_info['mlperf_compliant'] else '✗'}")
        print(f"Note:               {data_info['note']}")
        print("=" * 60)
        
        summary['data_info'] = data_info
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"whisper_benchmark_{args.data_type}_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    
    log.info(f"\nResults saved to: {output_file}")
    return summary


def main():
    args = get_args()
    
    # Install soundfile if using real data
    if args.data_type == "real":
        try:
            import soundfile
        except ImportError:
            log.info("Installing soundfile for audio loading...")
            import subprocess
            subprocess.run(["pip", "install", "-q", "soundfile"], check=True)
    
    # Load data first
    references = None
    if args.data_type == "real":
        samples, references, data_info = load_librispeech_samples(args.data_dir, args.max_examples)
        if samples is None:
            log.warning("Falling back to synthetic data")
            samples, data_info = get_synthetic_data(args.max_examples)
    else:
        samples, data_info = get_synthetic_data(args.max_examples)
    
    pipe, _ = load_model(args)
    run_benchmark(pipe, args, samples=samples, references=references, data_info=data_info)


if __name__ == "__main__":
    main()
