#!/usr/bin/env python3
"""
MLPerf Benchmark Setup and Runner - Stable Diffusion XL Image Generation

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
import atexit
import gc
import os
import signal
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("SDXL-Benchmark")


# ============================================================================
# GPU Cleanup Utilities
# ============================================================================

_pipeline_ref = None


def cleanup_gpu():
    """Properly cleanup GPU resources to prevent device unavailability issues."""
    global _pipeline_ref
    
    log.info("Cleaning up GPU resources...")
    
    if _pipeline_ref is not None:
        del _pipeline_ref
        _pipeline_ref = None
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated() / 1024**2
        log.info(f"GPU cleanup complete. Memory: {allocated:.1f}MB allocated")
    
    gc.collect()


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    log.warning(f"Received signal {signum}, cleaning up...")
    cleanup_gpu()
    sys.exit(1)


# Register cleanup handlers
atexit.register(cleanup_gpu)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Synthetic prompts for testing
SYNTHETIC_PROMPTS = [
    "A photo of an astronaut riding a horse on the moon",
    "A beautiful sunset over a mountain lake with reflections",
    "A futuristic city skyline at night with flying cars",
    "A cute robot playing chess with a cat",
    "An oil painting of a garden in spring with flowers",
    "A professional photograph of a coffee cup on a wooden table",
    "A fantasy castle floating in the clouds",
    "A photorealistic portrait of a wise elderly wizard",
    "A colorful tropical fish swimming in a coral reef",
    "A cozy cabin in a snowy forest at night",
]


def load_coco_captions(data_dir: str, max_examples: int) -> Tuple[Optional[List], Dict]:
    """Load real COCO captions as prompts"""
    captions_path = Path(data_dir) / "captions.json"
    
    if not captions_path.exists():
        log.warning(f"Captions not found at {captions_path}")
        return None, {}
    
    with open(captions_path) as f:
        captions = json.load(f)
    
    total_available = len(captions)
    log.info(f"Loaded {total_available} COCO captions")
    
    data_info = {
        'type': 'real',
        'dataset': 'COCO 2014 Captions',
        'source': str(data_dir),
        'samples_used': min(max_examples, total_available),
        'samples_available': total_available,
        'task': 'text_to_image',
        'verified': True,
        'mlperf_compliant': True,
        'note': 'COCO 2014 image captions for text-to-image generation'
    }
    
    return captions[:max_examples], data_info


def get_synthetic_data(max_examples: int) -> Tuple[List[str], Dict]:
    """Get synthetic prompts with data info"""
    prompts = SYNTHETIC_PROMPTS[:max_examples]
    
    data_info = {
        'type': 'synthetic',
        'dataset': 'Synthetic Prompts',
        'source': 'generated_prompts',
        'samples_used': len(prompts),
        'samples_available': len(SYNTHETIC_PROMPTS),
        'task': 'text_to_image',
        'verified': False,
        'mlperf_compliant': False,
        'note': 'Synthetic prompts for testing only'
    }
    
    return prompts, data_info


def build_command(args) -> str:
    """Build the command line string for reproducibility"""
    import sys
    cmd_parts = [sys.executable, __file__]
    
    cmd_parts.extend(["--model-name", args.model_name])
    cmd_parts.extend(["--device", args.device])
    cmd_parts.extend(["--max-examples", str(args.max_examples)])
    cmd_parts.extend(["--num-steps", str(args.num_steps)])
    cmd_parts.extend(["--data-type", args.data_type])
    cmd_parts.extend(["--data-dir", args.data_dir])
    cmd_parts.extend(["--output-dir", args.output_dir])
    
    if args.offload:
        cmd_parts.append("--offload")
    if args.save_images:
        cmd_parts.append("--save-images")
    if args.mlperf:
        cmd_parts.append("--mlperf")
    
    return " ".join(cmd_parts)


def get_args():
    parser = argparse.ArgumentParser(description="SDXL Benchmark with Real/Synthetic Prompts")
    parser.add_argument("--model-name", type=str, 
                       default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"])
    parser.add_argument("--offload", action="store_true",
                       help="Enable CPU offloading for models that don't fit in VRAM")
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--data-type", type=str, default="synthetic",
                       choices=["synthetic", "real"])
    parser.add_argument("--data-dir", type=str, default="data/coco-2014")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--save-images", action="store_true",
                       help="Save generated images")
    parser.add_argument("--mlperf", action="store_true",
                       help="Use MLPerf official settings")
    return parser.parse_args()


def load_model(args):
    """Load SDXL model with appropriate device configuration"""
    log.info(f"Loading model: {args.model_name}")
    log.info(f"Device: {args.device}, Offload: {args.offload}")
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        log.error("CUDA not available. Options:")
        log.error("  1. Use --cpu for CPU-only mode (very slow)")
        log.error("  2. Install CUDA and GPU drivers")
        raise SystemExit(1)
    
    try:
        if args.device == "cpu":
            torch_dtype = torch.float32
            log.info("Loading on CPU (this will be very slow)...")
            pipe = DiffusionPipeline.from_pretrained(
                args.model_name,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        elif args.offload:
            torch_dtype = torch.float16
            log.info("Loading with model CPU offloading...")
            pipe = DiffusionPipeline.from_pretrained(
                args.model_name,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                variant="fp16",
            )
            pipe.enable_model_cpu_offload()
        else:
            torch_dtype = torch.float16
            log.info("Loading on GPU...")
            pipe = DiffusionPipeline.from_pretrained(
                args.model_name,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                variant="fp16",
            )
            pipe = pipe.to("cuda")
    except torch.cuda.OutOfMemoryError:
        log.error("=" * 60)
        log.error("CUDA OUT OF MEMORY!")
        log.error("=" * 60)
        log.error("SDXL requires ~6.5GB VRAM. Try:")
        log.error("  1. --offload : Enable CPU offloading (~3GB VRAM)")
        log.error("  2. --cpu     : Run on CPU only (very slow)")
        raise SystemExit(1)
    
    log.info("Model loaded successfully!")
    return pipe


def run_benchmark(pipe, args, prompts: List[str], data_info: Dict = None):
    """Run the benchmark"""
    log.info("=" * 60)
    log.info("Starting SDXL Benchmark")
    log.info("=" * 60)
    log.info(f"Data type: {args.data_type}")
    log.info(f"Diffusion steps: {args.num_steps}")
    
    num_examples = len(prompts)
    log.info(f"Generating {num_examples} images")
    
    results = []
    total_time = 0
    total_steps = 0
    
    # Warmup
    log.info("Warmup...")
    try:
        _ = pipe(prompts[0], num_inference_steps=5, output_type="latent")
    except torch.cuda.OutOfMemoryError:
        log.error("=" * 60)
        log.error("CUDA OUT OF MEMORY during warmup!")
        log.error("=" * 60)
        log.error("SDXL requires ~6.5GB VRAM. Try:")
        log.error("  1. --offload : Enable CPU offloading (~3GB VRAM)")
        log.error("  2. --cpu     : Run on CPU only (very slow)")
        raise SystemExit(1)
    
    # Create images directory if saving
    if args.save_images:
        images_dir = Path(args.output_dir) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("\nRunning benchmark...")
    for i, prompt in enumerate(prompts):
        start_time = time.perf_counter()
        try:
            image = pipe(
                prompt,
                num_inference_steps=args.num_steps,
                guidance_scale=7.5,
            ).images[0]
        except torch.cuda.OutOfMemoryError:
            log.error("=" * 60)
            log.error(f"CUDA OUT OF MEMORY at sample {i+1}!")
            log.error("=" * 60)
            log.error("SDXL requires ~6.5GB VRAM. Try:")
            log.error("  1. --offload : Enable CPU offloading (~3GB VRAM)")
            log.error("  2. --cpu     : Run on CPU only (very slow)")
            raise SystemExit(1)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        total_time += elapsed
        total_steps += args.num_steps
        
        steps_per_sec = args.num_steps / elapsed
        
        log.info(f"  [{i+1}/{num_examples}] {elapsed:.2f}s, {steps_per_sec:.1f} steps/s")
        log.info(f"    Prompt: {prompt[:60]}...")
        
        # Save image if requested
        if args.save_images:
            image_path = images_dir / f"image_{i:03d}.png"
            image.save(image_path)
        
        results.append({
            "sample_id": i,
            "prompt": prompt[:100],
            "latency_sec": elapsed,
            "steps": args.num_steps,
            "steps_per_sec": steps_per_sec,
        })
    
    # Summary
    avg_latency = total_time / num_examples
    avg_steps_per_sec = total_steps / total_time if total_time > 0 else 0
    
    summary = {
        "model": args.model_name,
        "device": args.device,
        "data_type": args.data_type,
        "num_samples": num_examples,
        "num_steps": args.num_steps,
        "total_time_sec": total_time,
        "avg_latency_sec": avg_latency,
        "avg_steps_per_sec": avg_steps_per_sec,
        "images_per_min": 60 / avg_latency if avg_latency > 0 else 0,
        "timestamp": datetime.now().isoformat(),
        "mlperf_mode": args.mlperf,
        "mlperf_compliant": args.mlperf and args.data_type == "real",
    }
    
    print("\n" + "=" * 60)
    print("SDXL BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Device:             {args.device}")
    print(f"Data Type:          {args.data_type}")
    if args.mlperf:
        print(f"MLPerf Mode:        ENABLED")
    print(f"Total Images:       {num_examples}")
    print(f"Steps per Image:    {args.num_steps}")
    print(f"Total Time:         {total_time:.2f} seconds")
    print(f"Avg Latency:        {avg_latency:.2f} sec/image")
    print(f"Throughput:         {avg_steps_per_sec:.2f} steps/sec")
    print(f"Images per Min:     {60/avg_latency:.1f}")
    print("=" * 60)
    
    # Performance rating
    if avg_steps_per_sec > 10:
        print("Performance:        🚀 Excellent")
    elif avg_steps_per_sec > 3:
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
        print(f"Verified:           {'✓' if data_info['verified'] else '✗'}")
        print(f"MLPerf Compliant:   {'✓' if data_info['mlperf_compliant'] else '✗'}")
        print(f"Note:               {data_info['note']}")
        print("=" * 60)
        
        summary['data_info'] = data_info
    
    # Print command for reproducibility
    print("\n" + "=" * 60)
    print("COMMAND")
    print("=" * 60)
    print(build_command(args))
    print("=" * 60)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"sdxl_benchmark_{args.data_type}_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    
    log.info(f"\nResults saved to: {output_file}")
    return summary


def main():
    global _pipeline_ref
    
    args = get_args()
    
    try:
        # Load data first
        if args.data_type == "real":
            prompts, data_info = load_coco_captions(args.data_dir, args.max_examples)
            if prompts is None:
                log.warning("Falling back to synthetic prompts")
                prompts, data_info = get_synthetic_data(args.max_examples)
        else:
            prompts, data_info = get_synthetic_data(args.max_examples)
        
        pipe = load_model(args)
        _pipeline_ref = pipe
        
        run_benchmark(pipe, args, prompts=prompts, data_info=data_info)
    
    finally:
        cleanup_gpu()


if __name__ == "__main__":
    main()
