#!/usr/bin/env python3
"""
MLPerf Data Preparation Orchestrator

Central entry point for preparing data for MLPerf benchmarks.
Orchestrates data download (real) or generation (synthetic) and creates
proper metadata files for benchmark scripts to auto-detect data type.

Author: Mehdi Nik
Created: Jan 2026

Usage:
    python data_prepare.py --benchmark bert                    # Auto: try real, fallback synthetic
    python data_prepare.py --benchmark bert --real             # Download real data
    python data_prepare.py --benchmark bert --synthetic        # Generate synthetic data
    python data_prepare.py --check                             # Check status of all benchmarks
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import our data modules
from data_download import (
    DATASETS,
    download_benchmark_data,
    check_benchmark_data,
    get_available_methods,
    get_default_data_dir,
)
from data_gen import (
    GENERATORS,
    DEFAULT_SAMPLES,
    generate_benchmark_data,
)


# ============================================================================
# Metadata Management
# ============================================================================

def create_metadata(data_dir: Path, benchmark: str, data_type: str,
                    samples: int, info: Dict) -> Path:
    """
    Create metadata.json file for a dataset.
    
    This file is read by benchmark scripts to auto-detect data type.
    """
    metadata = {
        "name": DATASETS.get(benchmark, {}).get('name', benchmark),
        "benchmark": benchmark,
        "type": data_type,  # 'mlperf' or 'synthetic'
        "task": DATASETS.get(benchmark, {}).get('task', 'unknown'),
        "samples": samples,
        "mlperf_compliant": data_type == 'mlperf',
        "created": datetime.now().isoformat(),
        "generator": "data_prepare.py",
        "info": info,
    }
    
    # Add MLPerf-specific info
    mlperf_config = get_mlperf_config(benchmark)
    if mlperf_config:
        metadata["mlperf_config"] = mlperf_config
    
    metadata_file = data_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_file


def load_metadata(data_dir: Path) -> Optional[Dict]:
    """Load metadata.json if it exists."""
    metadata_file = data_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            return json.load(f)
    return None


def get_mlperf_config(benchmark: str) -> Dict:
    """Get MLPerf configuration for a benchmark."""
    configs = {
        'bert': {
            'performance_sample_count': 10833,
            'min_query_count': 10833,
            'target_latency_ms': 130,
        },
        'resnet50': {
            'performance_sample_count': 1024,
            'min_query_count': 24576,
            'target_latency_ms': 15,
        },
        'retinanet': {
            'performance_sample_count': 64,
            'min_query_count': 24576,
            'target_latency_ms': 100,
        },
        '3dunet': {
            'performance_sample_count': 16,
            'min_query_count': 16,
        },
        'whisper': {
            'performance_sample_count': 2513,
            'min_query_count': 100,
        },
        'sdxl': {
            'performance_sample_count': 5000,
            'min_query_count': 5000,
        },
        'gptj': {
            'performance_sample_count': 13368,
            'min_query_count': 13368,
        },
        'llama': {
            'performance_sample_count': 24576,
            'min_query_count': 24576,
        },
        'mixtral': {
            'performance_sample_count': 15000,
            'min_query_count': 15000,
        },
        'dlrm': {
            'performance_sample_count': 204800,
            'min_query_count': 204800,
        },
    }
    return configs.get(benchmark, {})


# ============================================================================
# Data Preparation Functions
# ============================================================================

def prepare_data(benchmark: str, mode: str = 'auto',
                 data_dir: Optional[Path] = None,
                 dataset_name: Optional[str] = None,
                 num_samples: Optional[int] = None,
                 method: str = 'auto',
                 force: bool = False,
                 verbose: bool = True) -> Tuple[bool, Dict]:
    """
    Prepare data for a benchmark.
    
    Args:
        benchmark: Benchmark name
        mode: 'auto', 'mlperf', or 'synthetic'
        data_dir: Override default data directory (full path)
        dataset_name: For synthetic mode, name of the dataset subfolder (REQUIRED)
        num_samples: For synthetic mode, number of samples
        method: Download method (wget, curl, urllib, auto)
        force: Force re-download/regenerate
        verbose: Print progress
        
    Returns:
        Tuple of (success, result_info)
    """
    # Validate benchmark
    if benchmark not in GENERATORS:
        return False, {'error': f"Unknown benchmark: {benchmark}"}
    
    # Determine data directory based on mode
    if data_dir is not None:
        # User specified explicit path
        data_dir = Path(data_dir)
    elif mode == 'mlperf' or mode == 'auto':
        # For mlperf: use data/{benchmark}/{default_dataset}/
        data_dir = get_default_data_dir(benchmark)
    elif mode == 'synthetic':
        # For synthetic: user MUST provide dataset_name
        if not dataset_name:
            return False, {
                'error': 'Synthetic mode requires --name to specify dataset folder name'
            }
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        data_dir = project_dir / 'data' / benchmark / dataset_name
    else:
        # Fallback
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        data_dir = project_dir / 'data' / benchmark / 'default'
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"PREPARING DATA: {benchmark.upper()}")
        print("=" * 70)
        print(f"  Mode:        {mode}")
        print(f"  Directory:   {data_dir}")
        if mode == 'synthetic' and num_samples:
            print(f"  Samples:     {num_samples}")
        print("=" * 70)
    
    # Check existing data
    existing_metadata = load_metadata(data_dir)
    if existing_metadata and not force:
        if verbose:
            print(f"\n  Data already exists:")
            print(f"    Type:    {existing_metadata.get('type', 'unknown')}")
            print(f"    Samples: {existing_metadata.get('samples', 'unknown')}")
            print(f"    MLPerf:  {'✓' if existing_metadata.get('mlperf_compliant') else '✗'}")
            print(f"\n  Use --force to re-prepare")
        return True, {'existing': True, 'metadata': existing_metadata}
    
    success = False
    result_info = {}
    data_type = None
    samples = 0
    
    if mode == 'mlperf' or mode == 'auto':
        # Try to download MLPerf-compliant data
        if verbose:
            print("\n  Attempting to download MLPerf data...")
        
        success, download_info = download_benchmark_data(
            benchmark, data_dir, method, force, verbose
        )
        
        if success and not download_info.get('manual_download') and not download_info.get('instructions_created'):
            data_type = 'mlperf'
            # Count samples from downloaded data
            samples = count_samples(data_dir, benchmark)
            result_info = download_info
            
            if verbose:
                print(f"\n  ✓ MLPerf data ready ({samples} samples)")
        elif mode == 'auto':
            if verbose:
                print("\n  MLPerf data not available, falling back to synthetic...")
            mode = 'synthetic'
    
    if mode == 'synthetic' and not success:
        # Generate synthetic data
        if verbose:
            print("\n  Generating synthetic data...")
        
        success, gen_info = generate_benchmark_data(
            benchmark, data_dir, num_samples, verbose=verbose
        )
        
        if success:
            data_type = 'synthetic'
            samples = gen_info.get('samples', num_samples or DEFAULT_SAMPLES.get(benchmark, 0))
            result_info = gen_info
            
            if verbose:
                print(f"\n  ✓ Synthetic data ready ({samples} samples)")
    
    # Create metadata
    if success and data_type:
        metadata_file = create_metadata(data_dir, benchmark, data_type, samples, result_info)
        if verbose:
            print(f"\n  Created metadata: {metadata_file}")
    
    # Final summary
    if verbose:
        print("\n" + "=" * 70)
        if success:
            print("DATA PREPARATION COMPLETE")
            print("=" * 70)
            print(f"  Benchmark:    {benchmark}")
            print(f"  Type:         {data_type}")
            print(f"  Samples:      {samples}")
            print(f"  MLPerf OK:    {'✓' if data_type == 'mlperf' else '✗ (synthetic)'}")
            print(f"  Location:     {data_dir}")
        else:
            print("DATA PREPARATION FAILED")
            print("=" * 70)
            if result_info.get('error'):
                print(f"  Error: {result_info['error']}")
        print("=" * 70)
    
    return success, {
        'benchmark': benchmark,
        'data_dir': str(data_dir),
        'data_type': data_type,
        'samples': samples,
        'info': result_info,
    }


def count_samples(data_dir: Path, benchmark: str) -> int:
    """Count samples in a data directory based on benchmark type."""
    try:
        if benchmark == 'bert':
            squad_file = data_dir / "dev-v1.1.json"
            if squad_file.exists():
                with open(squad_file) as f:
                    data = json.load(f)
                return sum(len(p['qas']) for article in data['data'] for p in article['paragraphs'])
        
        elif benchmark == 'resnet50':
            val_dir = data_dir / "val"
            if val_dir.exists():
                return sum(1 for _ in val_dir.rglob("*.JPEG"))
            images_file = data_dir / "images.npy"
            if images_file.exists():
                import numpy as np
                return len(np.load(images_file))
        
        elif benchmark == 'retinanet':
            # Check for downloaded images
            images_dir = data_dir / "validation" / "data"
            if images_dir.exists():
                return len(list(images_dir.glob("*.jpg")))
            annotations = data_dir / "annotations.json"
            if annotations.exists():
                with open(annotations) as f:
                    return len(json.load(f))
        
        elif benchmark == '3dunet':
            # Prefer preprocessed volumes.npy over raw cases
            volumes_file = data_dir / "volumes.npy"
            if volumes_file.exists():
                import numpy as np
                return len(np.load(volumes_file))
            raw_dir = data_dir / "raw"
            if raw_dir.exists():
                return len([d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith('case_')])
        
        elif benchmark == 'whisper':
            manifest = data_dir / "manifest.json"
            if manifest.exists():
                with open(manifest) as f:
                    return len(json.load(f))
        
        elif benchmark == 'sdxl':
            captions = data_dir / "captions.json"
            if captions.exists():
                with open(captions) as f:
                    return len(json.load(f))
        
        elif benchmark in ['gptj', 'llama', 'mixtral']:
            test_file = data_dir / "test.json"
            if test_file.exists():
                with open(test_file) as f:
                    return len(json.load(f))
        
        elif benchmark == 'dlrm':
            labels_file = data_dir / "labels.npy"
            if labels_file.exists():
                import numpy as np
                return len(np.load(labels_file))
    
    except Exception:
        pass
    
    return 0


# ============================================================================
# Status Check Functions
# ============================================================================

def check_all_data(verbose: bool = True) -> Dict[str, Dict]:
    """Check status of all benchmark data."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    results = {}
    
    if verbose:
        print("\n" + "=" * 70)
        print("MLPerf DATA STATUS")
        print("=" * 70)
        print(f"{'Benchmark':<12} {'Dataset':<15} {'Type':<10} {'Samples':>10} {'MLPerf':>8}")
        print("-" * 70)
    
    for benchmark in GENERATORS.keys():
        benchmark_dir = project_dir / 'data' / benchmark
        results[benchmark] = {'datasets': {}}
        
        if not benchmark_dir.exists():
            if verbose:
                print(f"{benchmark:<12} {'(no data)':<15} {'---':<10} {'---':>10} {'✗':>8}")
            continue
        
        # Scan all subdirectories (each is a dataset)
        datasets_found = False
        for dataset_dir in sorted(benchmark_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            
            datasets_found = True
            dataset_name = dataset_dir.name
            
            status = {
                'exists': True,
                'data_dir': str(dataset_dir),
                'type': None,
                'samples': 0,
                'mlperf_compliant': False,
                'ready': False,
            }
            
            metadata = load_metadata(dataset_dir)
            if metadata:
                status['type'] = metadata.get('type')
                status['samples'] = metadata.get('samples', 0)
                status['mlperf_compliant'] = metadata.get('mlperf_compliant', False)
                status['ready'] = True
            else:
                # No metadata but directory exists - try to detect
                samples = count_samples(dataset_dir, benchmark)
                if samples > 0:
                    status['samples'] = samples
                    status['ready'] = True
                    status['type'] = 'unknown'
            
            results[benchmark]['datasets'][dataset_name] = status
            
            if verbose:
                type_str = status['type'] or "---"
                samples_str = f"{status['samples']:,}" if status['samples'] else "---"
                mlperf_str = "✓" if status['mlperf_compliant'] else "✗"
                
                print(f"{benchmark:<12} {dataset_name:<15} {type_str:<10} {samples_str:>10} {mlperf_str:>8}")
        
        if not datasets_found:
            if verbose:
                print(f"{benchmark:<12} {'(empty)':<15} {'---':<10} {'---':>10} {'✗':>8}")
    
    if verbose:
        print("=" * 70)
        
        # Summary
        total_benchmarks = len(results)
        benchmarks_with_data = sum(1 for r in results.values() if r['datasets'])
        total_datasets = sum(len(r['datasets']) for r in results.values())
        mlperf_datasets = sum(
            1 for r in results.values() 
            for d in r['datasets'].values() 
            if d.get('mlperf_compliant')
        )
        
        print(f"\nSummary: {benchmarks_with_data}/{total_benchmarks} benchmarks have data")
        print(f"         {total_datasets} total datasets, {mlperf_datasets} MLPerf compliant")
        print("\nCommands:")
        print("  Prepare (mlperf):    python data_prepare.py --benchmark bert --mlperf")
        print("  Prepare (synthetic): python data_prepare.py --benchmark bert --synthetic --name test1")
        print("  Prepare all mlperf:  python data_prepare.py --benchmark all --mlperf")
    
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Prepare data for MLPerf benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --benchmark bert --mlperf              # Download official SQuAD data to data/bert/squad/
  %(prog)s --benchmark bert --synthetic --name test1  # Generate synthetic to data/bert/test1/
  %(prog)s --benchmark bert --synthetic --name test1 -n 5000  # Generate 5000 samples
  %(prog)s --benchmark all --mlperf               # Download all official datasets
  %(prog)s --check                                # Check status of all data
  
Directory Structure:
  data/{benchmark}/{dataset_name}/
  
  For --mlperf:    Uses default dataset name (e.g., squad, imagenet, openimages)
  For --synthetic: Requires --name to specify dataset folder name
  
Modes:
  --mlperf     Download official MLPerf-compliant datasets
  --synthetic  Generate synthetic data for testing (requires --name)
  
Download Methods (for --mlperf):
  --method wget    Use wget (best for corporate networks)
  --method curl    Use curl
  --method urllib  Use Python urllib
  --method auto    Try each until one works (default)

Available benchmarks:
  bert, resnet50, retinanet, 3dunet, whisper, sdxl, gptj, llama, mixtral, dlrm
"""
    )
    
    parser.add_argument('--benchmark', '-b', type=str,
                       help='Benchmark name or "all"')
    parser.add_argument('--mlperf', action='store_true',
                       help='Download official MLPerf-compliant data')
    parser.add_argument('--synthetic', action='store_true',
                       help='Generate synthetic data for testing')
    parser.add_argument('--name', type=str,
                       help='Dataset name (required for --synthetic, creates data/{benchmark}/{name}/)')
    parser.add_argument('--data-dir', '-d', type=str,
                       help='Override default data directory (full path)')
    parser.add_argument('--samples', '-n', type=int,
                       help='Number of samples for synthetic data')
    parser.add_argument('--method', '-m', type=str, default='auto',
                       choices=['auto', 'wget', 'curl', 'urllib'],
                       help='Download method for real data')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force re-prepare even if data exists')
    parser.add_argument('--check', '-c', action='store_true',
                       help='Check data status')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available benchmarks')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # List mode
    if args.list:
        print("\nAvailable Benchmarks:")
        print("-" * 80)
        print(f"  {'Benchmark':<12} {'Dataset Name':<25} {'Default Folder':<15} {'Samples':<10}")
        print("-" * 80)
        for name in GENERATORS:
            dataset_display = DATASETS.get(name, {}).get('name', 'Unknown')
            default_folder = DATASETS.get(name, {}).get('default_dataset', name)
            default_samples = DEFAULT_SAMPLES.get(name, '?')
            print(f"  {name:<12} {dataset_display:<25} {default_folder:<15} {default_samples:<10}")
        print("-" * 80)
        print("\nMLPerf data directory: data/{benchmark}/{default_folder}/")
        print("Synthetic data dir:    data/{benchmark}/{your_name}/")
        print()
        return 0
    
    # Check mode
    if args.check:
        check_all_data(verbose)
        return 0
    
    # Prepare mode
    if not args.benchmark:
        parser.print_help()
        return 1
    
    # Determine mode
    if args.mlperf and args.synthetic:
        print("Error: Cannot specify both --mlperf and --synthetic")
        return 1
    
    if not args.mlperf and not args.synthetic:
        print("Error: Must specify either --mlperf or --synthetic mode")
        print("  Use --mlperf to download official MLPerf data")
        print("  Use --synthetic --name <name> to generate synthetic data")
        return 1
    
    if args.mlperf:
        mode = 'mlperf'
    else:
        mode = 'synthetic'
        # Validate --name is provided for synthetic mode
        if not args.name and not args.data_dir:
            print("Error: --synthetic mode requires --name <dataset_name>")
            print("  Example: --benchmark bert --synthetic --name test1")
            print("  This will create data in: data/bert/test1/")
            return 1
    
    # Prepare benchmarks
    benchmarks = list(GENERATORS.keys()) if args.benchmark == 'all' else [args.benchmark]
    
    all_success = True
    for benchmark in benchmarks:
        data_dir = Path(args.data_dir) if args.data_dir else None
        success, info = prepare_data(
            benchmark=benchmark,
            mode=mode,
            data_dir=data_dir,
            dataset_name=args.name,
            num_samples=args.samples,
            method=args.method,
            force=args.force,
            verbose=verbose,
        )
        if not success:
            all_success = False
            if info.get('error'):
                print(f"Error: {info['error']}")
    
    # Final status check if preparing all
    if args.benchmark == 'all' and verbose:
        print("\n")
        check_all_data(verbose)
    
    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())
