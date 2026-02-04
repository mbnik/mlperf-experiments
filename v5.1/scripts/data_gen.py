#!/usr/bin/env python3
"""
MLPerf Synthetic Data Generator

Generates synthetic/fake data for MLPerf benchmarks for testing purposes.
Useful when real data cannot be downloaded (corporate firewall, disk space, etc.)

Author: Mehdi Nik
Created: Jan 2026

Usage:
    python data_gen.py --benchmark bert --samples 1000
    python data_gen.py --benchmark resnet50 --samples 5000
    python data_gen.py --benchmark all --samples 100
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional imports - some benchmarks need specific libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# Synthetic Data Generators
# ============================================================================

def generate_bert_data(data_dir: Path, num_samples: int = 1000,
                       seed: int = 42) -> Dict:
    """
    Generate synthetic SQuAD-like data for BERT QA benchmark.
    
    Creates fake questions, contexts, and answers that match SQuAD format.
    """
    np.random.seed(seed)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample vocabulary for generating text
    vocab = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "company", "system", "technology", "data", "information", "process",
        "development", "research", "analysis", "management", "service",
        "product", "market", "business", "industry", "organization",
        "computer", "software", "network", "internet", "database", "server",
        "application", "program", "code", "algorithm", "function", "method",
        "model", "learning", "training", "inference", "neural", "deep",
        "artificial", "intelligence", "machine", "benchmark", "performance",
    ]
    
    def random_text(min_words: int, max_words: int) -> str:
        length = np.random.randint(min_words, max_words + 1)
        return " ".join(np.random.choice(vocab, length))
    
    # Generate SQuAD-like structure
    squad_data = {"version": "synthetic-1.0", "data": []}
    
    samples_per_article = max(1, num_samples // 100)
    num_articles = (num_samples + samples_per_article - 1) // samples_per_article
    
    total_qas = 0
    for article_idx in range(num_articles):
        article = {
            "title": f"Synthetic Article {article_idx}",
            "paragraphs": []
        }
        
        # Each article has multiple paragraphs
        num_paragraphs = np.random.randint(1, 4)
        for para_idx in range(num_paragraphs):
            context = random_text(100, 300)
            
            # Each paragraph has multiple Q&A pairs
            qas = []
            num_qas = min(samples_per_article // num_paragraphs + 1, 
                         num_samples - total_qas)
            
            for qa_idx in range(num_qas):
                if total_qas >= num_samples:
                    break
                
                question = random_text(5, 15) + "?"
                
                # Answer is a substring of context
                words = context.split()
                start_idx = np.random.randint(0, max(1, len(words) - 5))
                answer_words = words[start_idx:start_idx + np.random.randint(1, 6)]
                answer_text = " ".join(answer_words)
                answer_start = context.find(answer_text)
                
                qas.append({
                    "id": f"synth_{article_idx}_{para_idx}_{qa_idx}",
                    "question": question,
                    "answers": [
                        {
                            "text": answer_text,
                            "answer_start": answer_start if answer_start >= 0 else 0
                        }
                    ],
                    "is_impossible": False
                })
                total_qas += 1
            
            article["paragraphs"].append({
                "context": context,
                "qas": qas
            })
            
            if total_qas >= num_samples:
                break
        
        squad_data["data"].append(article)
        
        if total_qas >= num_samples:
            break
    
    # Save in SQuAD format
    output_file = data_dir / "dev-v1.1.json"
    with open(output_file, 'w') as f:
        json.dump(squad_data, f, indent=2)
    
    print(f"  Generated {total_qas} QA pairs")
    print(f"  Saved to: {output_file}")
    
    # Save metadata.json
    metadata = {
        'type': 'synthetic',
        'samples': total_qas,
        'mlperf_compliant': False,
        'format': 'squad',
    }
    metadata_file = data_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_file}")
    
    return metadata


def generate_resnet50_data(data_dir: Path, num_samples: int = 1000,
                           image_size: int = 224, seed: int = 42) -> Dict:
    """
    Generate synthetic ImageNet-like images for ResNet50 benchmark.
    
    Creates random RGB images in ImageNet val/<class_id>/ structure.
    """
    from PIL import Image
    
    np.random.seed(seed)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    val_dir = data_dir / "val"
    val_dir.mkdir(exist_ok=True)
    
    print(f"  Generating {num_samples} images ({image_size}x{image_size}) in ImageNet structure...")
    
    # Use 10 classes for synthetic data (spread images across classes)
    num_classes = min(10, num_samples)
    images_per_class = num_samples // num_classes
    remainder = num_samples % num_classes
    
    image_count = 0
    for class_idx in range(num_classes):
        # Create class directory
        class_dir = val_dir / str(class_idx)
        class_dir.mkdir(exist_ok=True)
        
        # Number of images for this class
        n_images = images_per_class + (1 if class_idx < remainder else 0)
        
        for i in range(n_images):
            # Generate random image with some structure (gradients, shapes)
            image = np.random.randint(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
            
            # Add some visual structure (optional, makes images more realistic)
            # Add a gradient based on class
            gradient = np.linspace(0, 50, image_size).reshape(1, -1, 1).astype(np.uint8)
            image = np.clip(image.astype(np.int16) + gradient * (class_idx % 5), 0, 255).astype(np.uint8)
            
            # Save as JPEG
            img = Image.fromarray(image)
            image_file = class_dir / f"img_{image_count:06d}.JPEG"
            img.save(image_file, 'JPEG', quality=95)
            
            image_count += 1
    
    print(f"  Saved {image_count} JPEG images in {val_dir}")
    
    # Save metadata.json
    metadata = {
        'type': 'synthetic',
        'samples': image_count,
        'mlperf_compliant': False,
        'format': 'imagenet',
        'image_size': image_size,
        'num_classes': num_classes,
    }
    metadata_file = data_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_file}")
    
    return metadata


def generate_retinanet_data(data_dir: Path, num_samples: int = 100,
                            image_size: int = 800, seed: int = 42) -> Dict:
    """
    Generate synthetic images for RetinaNet object detection.
    
    Creates random JPEG images in validation/data/ directory (same as OpenImages).
    """
    from PIL import Image
    
    np.random.seed(seed)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    # Use same structure as OpenImages: validation/data/
    images_dir = data_dir / "validation" / "data"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Generating {num_samples} JPEG images ({image_size}x{image_size})...")
    
    # For object detection, save images as JPEG with annotations
    annotations = []
    
    for i in range(num_samples):
        # Generate random image with some structure
        image = np.random.randint(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
        
        # Add some visual structure (rectangles to simulate objects)
        num_shapes = np.random.randint(2, 8)
        for _ in range(num_shapes):
            x1 = np.random.randint(0, image_size - 100)
            y1 = np.random.randint(0, image_size - 100)
            x2 = x1 + np.random.randint(50, 200)
            y2 = y1 + np.random.randint(50, 200)
            x2 = min(x2, image_size)
            y2 = min(y2, image_size)
            color = np.random.randint(0, 256, 3)
            image[y1:y2, x1:x2] = np.clip(image[y1:y2, x1:x2].astype(np.int16) + color // 2, 0, 255).astype(np.uint8)
        
        # Save as JPEG
        img = Image.fromarray(image)
        image_file = images_dir / f"image_{i:06d}.jpg"
        img.save(image_file, 'JPEG', quality=95)
        
        # Generate random bounding boxes (COCO format)
        num_boxes = np.random.randint(1, 10)
        boxes = []
        for _ in range(num_boxes):
            x = np.random.randint(0, image_size - 50)
            y = np.random.randint(0, image_size - 50)
            w = np.random.randint(20, min(100, image_size - x))
            h = np.random.randint(20, min(100, image_size - y))
            category = np.random.randint(1, 91)  # COCO has 90 categories
            boxes.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'category_id': int(category),
            })
        
        annotations.append({
            'image_id': f"image_{i:06d}",
            'file_name': f"image_{i:06d}.jpg",
            'height': image_size,
            'width': image_size,
            'annotations': boxes,
        })
    
    # Save annotations in same location as OpenImages
    annotations_file = data_dir / "validation-annotations-bbox.json"
    with open(annotations_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"  Saved {num_samples} JPEG images to {images_dir}")
    print(f"  Saved annotations: {annotations_file}")
    
    # Save metadata.json
    metadata = {
        'type': 'synthetic',
        'samples': num_samples,
        'mlperf_compliant': False,
        'format': 'openimages',
        'image_size': image_size,
    }
    metadata_file = data_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_file}")
    
    return metadata


def generate_3dunet_data(data_dir: Path, num_samples: int = 10,
                         volume_size: Tuple[int, int, int] = (128, 128, 128),
                         seed: int = 42) -> Dict:
    """
    Generate synthetic 3D medical volumes for 3D-UNet benchmark.
    
    Creates random 3D volumes and segmentation masks.
    """
    np.random.seed(seed)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Generating {num_samples} 3D volumes {volume_size}...")
    print(f"  This may take a while for large volumes...")
    
    volumes = []
    labels = []
    
    for i in range(num_samples):
        print(f"    Volume {i+1}/{num_samples}...", end="\r")
        
        # Generate random volume (simulating CT scan intensity values)
        # Shape: (1, D, H, W) - 1 channel for grayscale CT
        volume = np.random.randn(1, *volume_size).astype(np.float32)
        volume = (volume - volume.min()) / (volume.max() - volume.min())
        
        # Generate random segmentation mask (3 classes: background, kidney, tumor)
        # Shape: (1, D, H, W) - matching volume shape
        label = np.zeros((1, *volume_size), dtype=np.int64)
        
        # Add some random "organ" regions
        for _ in range(np.random.randint(1, 4)):
            cx, cy, cz = [np.random.randint(20, s-20) for s in volume_size]
            rx, ry, rz = [np.random.randint(10, 30) for _ in range(3)]
            
            x, y, z = np.ogrid[:volume_size[0], :volume_size[1], :volume_size[2]]
            mask = ((x-cx)**2/rx**2 + (y-cy)**2/ry**2 + (z-cz)**2/rz**2) <= 1
            label[0, mask] = np.random.randint(1, 3)  # kidney or tumor
        
        volumes.append(volume)
        labels.append(label)
    
    print()  # New line after progress
    
    # Save as numpy arrays
    volumes_array = np.stack(volumes)
    labels_array = np.stack(labels)
    
    volumes_file = data_dir / "volumes.npy"
    labels_file = data_dir / "labels.npy"
    
    np.save(volumes_file, volumes_array)
    np.save(labels_file, labels_array)
    
    print(f"  Saved volumes: {volumes_file} ({volumes_file.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Saved labels: {labels_file}")
    
    # Save metadata.json
    metadata = {
        'type': 'synthetic',
        'samples': num_samples,
        'mlperf_compliant': False,
        'format': 'numpy',
        'volume_size': list(volume_size),
    }
    metadata_file = data_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_file}")
    
    return metadata


def generate_whisper_data(data_dir: Path, num_samples: int = 100,
                          duration_sec: float = 5.0, sample_rate: int = 16000,
                          seed: int = 42) -> Dict:
    """
    Generate synthetic audio data for Whisper ASR benchmark.
    
    Creates random audio signals (sine waves with noise).
    """
    np.random.seed(seed)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Generating {num_samples} audio samples ({duration_sec}s each)...")
    
    audio_samples = []
    transcripts = []
    
    num_samples_per_audio = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples_per_audio)
    
    vocab = ["hello", "world", "this", "is", "a", "test", "audio", "sample",
             "speech", "recognition", "benchmark", "synthetic", "data"]
    
    for i in range(num_samples):
        # Generate audio: combination of sine waves + noise
        freq1 = np.random.uniform(200, 800)
        freq2 = np.random.uniform(400, 1200)
        
        audio = 0.3 * np.sin(2 * np.pi * freq1 * t)
        audio += 0.2 * np.sin(2 * np.pi * freq2 * t)
        audio += 0.1 * np.random.randn(num_samples_per_audio)
        
        audio_samples.append(audio.astype(np.float32))
        
        # Generate random "transcript"
        transcript = " ".join(np.random.choice(vocab, np.random.randint(3, 10)))
        transcripts.append(transcript)
    
    # Save audio samples
    audio_file = data_dir / "audio_samples.npy"
    np.save(audio_file, np.stack(audio_samples))
    
    # Save manifest
    manifest = [
        {"audio_index": i, "text": transcripts[i], "duration": duration_sec}
        for i in range(num_samples)
    ]
    manifest_file = data_dir / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  Saved audio: {audio_file} ({audio_file.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Saved manifest: {manifest_file}")
    
    # Save metadata.json
    metadata = {
        'type': 'synthetic',
        'samples': num_samples,
        'mlperf_compliant': False,
        'format': 'numpy',
        'duration_sec': duration_sec,
        'sample_rate': sample_rate,
    }
    metadata_file = data_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_file}")
    
    return metadata


def generate_sdxl_data(data_dir: Path, num_samples: int = 100,
                       seed: int = 42) -> Dict:
    """
    Generate synthetic prompts for SDXL text-to-image benchmark.
    
    Creates random text prompts compatible with run_sdxl_benchmark.py.
    Format matches COCO captions (simple list of strings).
    """
    np.random.seed(seed)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Generating {num_samples} prompts...")
    
    # Prompt templates - designed to test diverse image generation capabilities
    subjects = [
        "a cat", "a dog", "a robot", "a wizard", "a castle", "a mountain",
        "a forest", "an astronaut", "a dragon", "a city skyline", "a spaceship",
        "a portrait of a woman", "a landscape", "a flower bouquet", "an ocean wave",
        "a vintage car", "a futuristic building", "a medieval knight", "a tropical beach",
        "a snowy village", "a steampunk machine", "a fantasy creature", "a cozy cabin",
        "a busy street market", "a serene lake", "a majestic eagle", "a colorful parrot",
        "a wise old owl", "a cute puppy", "a playful kitten"
    ]
    
    styles = [
        "photorealistic", "oil painting", "watercolor", "digital art",
        "3D render", "pencil sketch", "anime style", "impressionist",
        "surrealist", "minimalist", "baroque", "modern art", "pop art",
        "art nouveau", "cyberpunk", "steampunk", "fantasy art", "concept art",
        "hyperrealistic", "low poly 3D", "pixel art", "vector illustration"
    ]
    
    settings = [
        "at sunset", "in the rain", "under moonlight", "in a garden",
        "on a mountain top", "by the sea", "in outer space", "in a dense forest",
        "at night with stars", "during golden hour sunrise", "in winter snow",
        "in autumn leaves", "in a misty morning", "during a thunderstorm",
        "in a desert oasis", "underwater", "in a meadow of wildflowers"
    ]
    
    details = [
        "highly detailed", "8k resolution", "trending on artstation",
        "dramatic lighting", "cinematic composition", "vibrant colors", "soft focus",
        "sharp focus", "studio lighting", "golden hour lighting", "volumetric fog",
        "intricate details", "professional photography", "award winning",
        "masterpiece", "ultra realistic", "ray tracing", "ambient occlusion"
    ]
    
    prompts = []
    for i in range(num_samples):
        subject = np.random.choice(subjects)
        style = np.random.choice(styles)
        setting = np.random.choice(settings)
        num_details = np.random.randint(2, 5)
        detail_list = ", ".join(np.random.choice(details, num_details, replace=False))
        
        prompt = f"{subject} {setting}, {style}, {detail_list}"
        prompts.append(prompt)
    
    # Save prompts as simple list (compatible with run_sdxl_benchmark.py and COCO format)
    prompts_file = data_dir / "captions.json"
    with open(prompts_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    print(f"  Saved {len(prompts)} prompts to: {prompts_file}")
    
    # Save metadata.json
    metadata = {
        'type': 'synthetic',
        'samples': num_samples,
        'images': 0,
        'mlperf_compliant': False,
        'format': 'captions-only',
        'source': 'synthetic-prompts',
        'captions_file': 'captions.json',
        'note': 'Synthetic prompts for SDXL testing. Not MLPerf compliant.',
    }
    metadata_file = data_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_file}")
    
    return metadata


def generate_gptj_data(data_dir: Path, num_samples: int = 100,
                       seed: int = 42) -> Dict:
    """
    Generate synthetic summarization data for GPT-J benchmark.
    
    Creates random articles and summaries.
    """
    np.random.seed(seed)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Generating {num_samples} articles...")
    
    # Vocabulary for generating text
    vocab = [
        "the", "company", "announced", "today", "that", "it", "will", "be",
        "launching", "a", "new", "product", "in", "market", "next", "quarter",
        "according", "to", "sources", "close", "matter", "development", "has",
        "been", "ongoing", "for", "several", "years", "and", "expected", "to",
        "revolutionize", "industry", "experts", "say", "this", "could", "lead",
        "significant", "changes", "how", "business", "operates", "future",
        "technology", "innovation", "research", "scientists", "discovered",
        "breakthrough", "study", "published", "journal", "findings", "suggest",
    ]
    
    def random_article(min_words: int = 200, max_words: int = 500) -> str:
        length = np.random.randint(min_words, max_words + 1)
        words = np.random.choice(vocab, length)
        # Add some sentence structure
        sentences = []
        i = 0
        while i < len(words):
            sent_len = np.random.randint(8, 20)
            sent_words = words[i:i+sent_len].tolist()
            if sent_words:
                sent_words[0] = sent_words[0].capitalize()
                sentences.append(" ".join(sent_words) + ".")
            i += sent_len
        return " ".join(sentences)
    
    def random_summary(article: str) -> str:
        # Take first few sentences as "summary"
        sentences = article.split(".")[:3]
        return ". ".join(sentences) + "."
    
    data = []
    for i in range(num_samples):
        article = random_article()
        highlights = random_summary(article)
        data.append({
            "id": str(i),
            "article": article,
            "highlights": highlights,
        })
    
    # Save
    output_file = data_dir / "test.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Saved articles: {output_file}")
    
    # Save metadata.json
    metadata = {
        'type': 'synthetic',
        'samples': num_samples,
        'mlperf_compliant': False,
        'format': 'cnn-dailymail',
    }
    metadata_file = data_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_file}")
    
    return metadata


def generate_llama_data(data_dir: Path, num_samples: int = 100,
                        seed: int = 42) -> Dict:
    """
    Generate synthetic instruction data for Llama benchmark.
    
    Creates instruction-response pairs matching MLPerf OpenOrca format.
    Includes pre-formatted Llama chat template tokens.
    """
    np.random.seed(seed)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Generating {num_samples} instruction pairs (OpenOrca format)...")
    
    # System prompts matching MLPerf OpenOrca style
    system_prompts = [
        "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.",
        "You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old.",
        "You are an AI assistant that helps people find information.",
        "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",
        "You are an AI assistant. Provide a detailed answer so user don't need to search outside to understand the answer.",
        "",  # Some prompts have no system prompt
    ]
    
    # Task templates matching MLPerf OpenOrca style
    task_templates = [
        ("Explain the concept of {topic} in detail.", "explanation"),
        ("What are the main differences between {topic1} and {topic2}?", "comparison"),
        ("Summarize the following information about {topic}:\n\n{context}", "summarization"),
        ("Generate a detailed description of {topic}.", "description"),
        ("Answer the following question: What is {topic} and how does it work?", "qa"),
        ("Write a short paragraph about {topic}.", "writing"),
        ("List the key benefits and drawbacks of {topic}.", "analysis"),
        ("Given the following scenario about {topic}, what would be the best approach?\n\nScenario: {context}", "reasoning"),
    ]
    
    topics = [
        "machine learning", "quantum computing", "climate change",
        "artificial intelligence", "renewable energy", "space exploration",
        "blockchain technology", "genetic engineering", "neural networks",
        "sustainable development", "cybersecurity", "data privacy",
        "cloud computing", "internet of things", "virtual reality",
        "autonomous vehicles", "natural language processing", "robotics",
    ]
    
    # Context snippets for summarization/reasoning tasks
    contexts = [
        "The technology has been developing rapidly over the past decade. Researchers have made significant breakthroughs in efficiency and scalability. Industry adoption has increased by 300% in the last five years.",
        "Recent studies show promising results in this field. The global market is expected to grow significantly. Several major companies have announced new initiatives.",
        "The approach involves multiple stages of processing and analysis. Each stage builds upon the results of the previous one. The final output is evaluated against established benchmarks.",
    ]
    
    data = []
    for i in range(num_samples):
        # Select random components
        system_prompt = np.random.choice(system_prompts)
        template, task_type = task_templates[np.random.randint(0, len(task_templates))]
        topic = np.random.choice(topics)
        topic1, topic2 = np.random.choice(topics, size=2, replace=False)
        context = np.random.choice(contexts)
        
        # Build original question
        original_question = template.format(
            topic=topic, 
            topic1=topic1, 
            topic2=topic2,
            context=context
        )
        
        # Build pre-formatted question with Llama chat template (matching MLPerf format)
        if system_prompt:
            question = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{original_question} [/INST]"
        else:
            question = f"<s>[INST] {original_question} [/INST]"
        
        # Generate response (synthetic but reasonable length)
        response_parts = [
            f"Based on the given task about {topic}, here is my response.",
            f"This involves understanding the key concepts and applications of {topic}.",
            f"The main aspects to consider include efficiency, scalability, and practical implementation.",
            f"In conclusion, {topic} represents an important area of development with significant potential.",
        ]
        response = " ".join(response_parts)
        
        data.append({
            "id": str(i),
            "question": question,
            "response": response,
            "system_prompt": system_prompt,
            "original_question": original_question,
        })
    
    output_file = data_dir / "test.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Saved instructions: {output_file}")
    print(f"  Format: MLPerf OpenOrca compatible (with Llama chat template)")
    
    # Save metadata.json
    metadata = {
        'type': 'synthetic',
        'samples': num_samples,
        'mlperf_compliant': False,
        'format': 'openorca',
        'note': 'Synthetic data with pre-formatted Llama chat template tokens'
    }
    metadata_file = data_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_file}")
    
    return metadata


def generate_mixtral_data(data_dir: Path, num_samples: int = 100,
                          seed: int = 42) -> Dict:
    """
    Generate synthetic combined data for Mixtral benchmark.
    
    Creates mix of OpenOrca, GSM8K-like, and code samples in MLPerf format.
    Questions are pre-formatted with <s> [INST] ... [/INST] tokens.
    
    MLPerf Mixtral uses 5-shot chain-of-thought format for math problems.
    """
    np.random.seed(seed)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Generating {num_samples} mixed samples (MLPerf format)...")
    
    # Split: 40% GSM8K, 40% OpenOrca, 20% MBXP (code)
    gsm8k_count = int(num_samples * 0.4)
    openorca_count = int(num_samples * 0.4)
    mbxp_count = num_samples - gsm8k_count - openorca_count
    
    data = []
    
    # =========================================================================
    # GSM8K-style 5-shot chain-of-thought math problems (MLPerf format)
    # =========================================================================
    # 5-shot examples (constant prefix for all GSM8K questions)
    gsm8k_prefix = """<s> [INST] As an expert problem solver solve step by step the following mathematical questions. You will be given 5 examples and a question at the end. Answer the last question in the same format as the example.

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: [/INST] We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6. </s> [INST]

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: [/INST] There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5. </s> [INST]

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: [/INST] Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39. </s> [INST]

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: [/INST] Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8. </s> [INST]

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: [/INST] He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9. </s> [INST]

Q: """
    
    # Math problem templates
    math_problems = [
        ("John has {a} apples. He buys {b} more apples. How many apples does John have now?",
         "John starts with {a} apples and buys {b} more. So he has {a} + {b} = {result} apples. The answer is {result}."),
        ("A store has {a} books. They sell {b} books. How many books are left?",
         "The store had {a} books and sold {b}. So they have {a} - {b} = {result} books left. The answer is {result}."),
        ("Sarah baked {a} cookies for her {b} friends to share equally. How many cookies does each friend get?",
         "Sarah has {a} cookies for {b} friends. Each friend gets {a} / {b} = {result} cookies. The answer is {result}."),
        ("A farmer has {a} chickens. Each chicken lays {b} eggs per day. How many eggs does the farmer collect in one day?",
         "The farmer has {a} chickens, each laying {b} eggs. Total eggs = {a} × {b} = {result}. The answer is {result}."),
        ("Mike runs {a} miles every day. How many miles does he run in {b} days?",
         "Mike runs {a} miles daily for {b} days. Total = {a} × {b} = {result} miles. The answer is {result}."),
        ("A class has {a} students. If {b} students are absent, how many students are present?",
         "The class has {a} students with {b} absent. Present = {a} - {b} = {result} students. The answer is {result}."),
        ("Emma has ${a}. She earns ${b} more. How much money does Emma have now?",
         "Emma had ${a} and earned ${b} more. Total = ${a} + ${b} = ${result}. The answer is {result}."),
        ("A train travels {a} miles per hour. How far does it travel in {b} hours?",
         "The train goes {a} mph for {b} hours. Distance = {a} × {b} = {result} miles. The answer is {result}."),
    ]
    
    for i in range(gsm8k_count):
        template, answer_template = math_problems[i % len(math_problems)]
        
        # Generate appropriate numbers based on operation
        if "sell" in template or "absent" in template:
            a = np.random.randint(20, 100)
            b = np.random.randint(5, a)
            result = a - b
        elif "share" in template:
            b = np.random.randint(2, 8)
            a = b * np.random.randint(3, 15)
            result = a // b
        else:
            a = np.random.randint(5, 50)
            b = np.random.randint(2, 20)
            result = a + b if "+" in answer_template else a * b
        
        question = template.format(a=a, b=b)
        answer = answer_template.format(a=a, b=b, result=result)
        
        # Format in MLPerf style
        formatted_question = gsm8k_prefix + question + "\nA: [/INST]"
        
        data.append({
            "id": str(i),
            "question": formatted_question,
            "response": f"synthetic_gsm8k.{i}",
            "source": "combined",
            "_synthetic_answer": answer,
        })
    
    # =========================================================================
    # OpenOrca-style instruction following
    # =========================================================================
    orca_templates = [
        ("Explain the concept of {topic} in simple terms.",
         "A clear explanation of {topic} covering key aspects and examples."),
        ("What are the main benefits of {topic}?",
         "The key benefits of {topic} include improved efficiency, better outcomes, and accessibility."),
        ("Describe how {topic} works.",
         "Here's how {topic} works: it involves a systematic process that achieves specific goals."),
        ("Compare and contrast {topic1} and {topic2}.",
         "While {topic1} and {topic2} share similarities, they differ in key ways including approach and application."),
        ("Write a brief summary about {topic}.",
         "In summary, {topic} is an important concept that has significant implications in various fields."),
    ]
    
    topics = [
        "machine learning", "climate change", "renewable energy", "artificial intelligence",
        "blockchain technology", "quantum computing", "sustainable agriculture", "space exploration",
        "genetic engineering", "cybersecurity", "electric vehicles", "5G networks",
        "virtual reality", "cloud computing", "data privacy", "autonomous vehicles",
    ]
    
    for i in range(openorca_count):
        template_idx = i % len(orca_templates)
        question_template, answer_template = orca_templates[template_idx]
        
        if "{topic1}" in question_template:
            topic1 = topics[i % len(topics)]
            topic2 = topics[(i + 1) % len(topics)]
            question = question_template.format(topic1=topic1, topic2=topic2)
            answer = answer_template.format(topic1=topic1, topic2=topic2)
        else:
            topic = topics[i % len(topics)]
            question = question_template.format(topic=topic)
            answer = answer_template.format(topic=topic)
        
        # Format in Mixtral instruction format
        formatted_question = f"<s> [INST] You are a helpful AI assistant. {question} [/INST]"
        
        data.append({
            "id": str(gsm8k_count + i),
            "question": formatted_question,
            "response": f"synthetic_orca.{i}",
            "source": "combined",
            "_synthetic_answer": answer,
        })
    
    # =========================================================================
    # MBXP-style code generation
    # =========================================================================
    code_templates = [
        ("Write a Python function to calculate the factorial of a number.",
         "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),
        ("Write a Python function to check if a string is a palindrome.",
         "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]"),
        ("Write a Python function to find the maximum element in a list.",
         "def find_max(lst):\n    if not lst:\n        return None\n    return max(lst)"),
        ("Write a Python function to reverse a string.",
         "def reverse_string(s):\n    return s[::-1]"),
        ("Write a Python function to count vowels in a string.",
         "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"),
        ("Write a Python function to check if a number is prime.",
         "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
        ("Write a Python function to compute the Fibonacci sequence up to n terms.",
         "def fibonacci(n):\n    fib = [0, 1]\n    for i in range(2, n):\n        fib.append(fib[-1] + fib[-2])\n    return fib[:n]"),
        ("Write a Python function to sort a list of numbers.",
         "def sort_list(lst):\n    return sorted(lst)"),
    ]
    
    for i in range(mbxp_count):
        question, answer = code_templates[i % len(code_templates)]
        
        # Format in Mixtral instruction format
        formatted_question = f"<s> [INST] You are an expert programmer. {question} [/INST]"
        
        data.append({
            "id": str(gsm8k_count + openorca_count + i),
            "question": formatted_question,
            "response": f"synthetic_mbxp.{i}",
            "source": "combined",
            "_synthetic_answer": answer,
        })
    
    # Shuffle the data
    np.random.shuffle(data)
    
    # Re-assign IDs after shuffle
    for i, item in enumerate(data):
        item["id"] = str(i)
    
    output_file = data_dir / "test.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Saved {len(data)} mixed samples: {output_file}")
    print(f"    - GSM8K (math): {gsm8k_count} samples")
    print(f"    - OpenOrca (QA): {openorca_count} samples")
    print(f"    - MBXP (code): {mbxp_count} samples")
    
    # Save metadata.json
    metadata = {
        'type': 'synthetic',
        'samples': len(data),
        'mlperf_compliant': False,
        'format': 'mixtral-combined',
        'categories': {
            'gsm8k': gsm8k_count,
            'openorca': openorca_count,
            'mbxp': mbxp_count,
        },
        'note': 'Pre-formatted prompts matching MLPerf Mixtral format',
    }
    metadata_file = data_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_file}")
    
    return metadata


def generate_dlrm_data(data_dir: Path, num_samples: int = 10000,
                       seed: int = 42) -> Dict:
    """
    Generate synthetic Criteo-like data for DLRM benchmark.
    
    Creates data matching the MLPerf DLRM-v2 preprocessed format:
    - day_23_dense.npy: (samples, 13) float32 dense features
    - day_23_sparse_multi_hot.npz: 26 sparse features with multi-hot encoding
    
    The sparse features use the same embedding table sizes and multi-hot widths
    as the official MLPerf preprocessed Criteo dataset.
    """
    np.random.seed(seed)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Generating {num_samples:,} recommendation samples (MLPerf format)...")
    
    # DLRM-v2 configuration matching official MLPerf preprocessed data
    num_dense_features = 13
    num_sparse_features = 26
    
    # Embedding table sizes from MLPerf DLRM-v2 spec (26 tables)
    embedding_sizes = [
        40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63,
        40000000, 3067956, 405282, 10, 2209, 11938, 155, 4,
        976, 14, 40000000, 40000000, 40000000, 590152, 12973, 108, 36
    ]
    
    # Multi-hot widths for each sparse feature (from real data analysis)
    # These match the actual Criteo Day 23 preprocessed structure
    multi_hot_widths = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 100, 100, 3, 1, 1]
    
    # Generate dense features (normalized, like Criteo)
    print("    Generating dense features...")
    dense_features = np.random.randn(num_samples, num_dense_features).astype(np.float32)
    # Apply log transform like real Criteo data processing
    dense_features = np.log1p(np.abs(dense_features))
    
    # Save as day_23_dense.npy (matching real data filename)
    dense_file = data_dir / "day_23_dense.npy"
    np.save(dense_file, dense_features)
    print(f"    Saved: {dense_file.name} ({dense_features.shape})")
    
    # Generate sparse multi-hot features
    print("    Generating sparse multi-hot features...")
    sparse_dict = {}
    for i, (emb_size, mh_width) in enumerate(zip(embedding_sizes, multi_hot_widths)):
        # Generate random indices into embedding table
        # Cap embedding size to avoid memory issues with very large tables
        max_idx = min(emb_size, 1000000)
        indices = np.random.randint(0, max_idx, size=(num_samples, mh_width)).astype(np.int32)
        sparse_dict[str(i)] = indices
        if i < 5 or i >= 21:  # Show first 5 and last 5
            print(f"      Feature {i}: shape={indices.shape}, max_idx={max_idx}")
    
    # Save as day_23_sparse_multi_hot.npz (matching real data filename)
    sparse_file = data_dir / "day_23_sparse_multi_hot.npz"
    np.savez_compressed(sparse_file, **sparse_dict)
    print(f"    Saved: {sparse_file.name} (26 features)")
    
    # Also save labels (for compatibility with some benchmark implementations)
    labels = np.random.randint(0, 2, size=num_samples).astype(np.int32)
    labels_file = data_dir / "day_23_labels.npy"
    np.save(labels_file, labels)
    print(f"    Saved: {labels_file.name} ({labels.shape})")
    
    # Calculate file sizes
    dense_size_mb = dense_file.stat().st_size / (1024 * 1024)
    sparse_size_mb = sparse_file.stat().st_size / (1024 * 1024)
    labels_size_mb = labels_file.stat().st_size / (1024 * 1024)
    
    print(f"    Total size: {dense_size_mb + sparse_size_mb + labels_size_mb:.1f} MB")
    
    # Save metadata.json
    metadata = {
        'type': 'synthetic',
        'samples': num_samples,
        'mlperf_compliant': False,
        'format': 'mlperf-preprocessed',
        'num_dense_features': num_dense_features,
        'num_sparse_features': num_sparse_features,
        'embedding_sizes': embedding_sizes,
        'multi_hot_widths': multi_hot_widths,
        'files': ['day_23_dense.npy', 'day_23_sparse_multi_hot.npz', 'day_23_labels.npy'],
    }
    metadata_file = data_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"    Saved: {metadata_file.name}")
    
    return metadata


# ============================================================================
# Generator Registry
# ============================================================================

GENERATORS = {
    'bert': generate_bert_data,
    'resnet50': generate_resnet50_data,
    'retinanet': generate_retinanet_data,
    '3dunet': generate_3dunet_data,
    'whisper': generate_whisper_data,
    'sdxl': generate_sdxl_data,
    'gptj': generate_gptj_data,
    'llama': generate_llama_data,
    'mixtral': generate_mixtral_data,
    'dlrm': generate_dlrm_data,
}

# Default sample counts for each benchmark
DEFAULT_SAMPLES = {
    'bert': 1000,
    'resnet50': 1000,
    'retinanet': 100,
    '3dunet': 10,
    'whisper': 100,
    'sdxl': 100,
    'gptj': 100,
    'llama': 100,
    'mixtral': 100,
    'dlrm': 10000,
}


# ============================================================================
# Main Generation Function
# ============================================================================

def generate_benchmark_data(benchmark: str, data_dir: Path,
                           num_samples: Optional[int] = None,
                           seed: int = 42, verbose: bool = True) -> Tuple[bool, Dict]:
    """
    Generate synthetic data for a benchmark.
    
    Args:
        benchmark: Benchmark name
        data_dir: Output directory (REQUIRED - user must specify dataset name)
        num_samples: Number of samples (uses default if None)
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Tuple of (success, info_dict)
    """
    if benchmark not in GENERATORS:
        return False, {'error': f"Unknown benchmark: {benchmark}"}
    
    if data_dir is None:
        return False, {'error': 'data_dir is required for synthetic data generation'}
    
    data_dir = Path(data_dir)
    
    # Determine sample count
    if num_samples is None:
        num_samples = DEFAULT_SAMPLES[benchmark]
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"GENERATING SYNTHETIC DATA: {benchmark.upper()}")
        print("=" * 60)
        print(f"  Samples:     {num_samples}")
        print(f"  Destination: {data_dir}")
        print(f"  Seed:        {seed}")
        print("=" * 60)
    
    try:
        generator = GENERATORS[benchmark]
        info = generator(data_dir, num_samples=num_samples, seed=seed)
        
        if verbose:
            print("\n" + "=" * 60)
            print("GENERATION COMPLETE")
            print("=" * 60)
        
        return True, info
        
    except Exception as e:
        if verbose:
            print(f"\nERROR: {e}")
        return False, {'error': str(e)}


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic data for MLPerf benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -b bert --name test1                # Creates data/bert/test1/
  %(prog)s -b bert --name test1 -n 5000        # Generate 5000 samples
  %(prog)s -b gptj --name exp1 -d /tmp/data    # Creates /tmp/data/gptj/exp1/
  
Note: For full data preparation workflow, consider using data_prepare.py instead:
  python data_prepare.py -b bert --synthetic --name test1
  
Available benchmarks:
  bert, resnet50, retinanet, 3dunet, whisper, sdxl, gptj, llama, mixtral, dlrm
"""
    )
    
    parser.add_argument('--benchmark', '-b', type=str,
                       help='Benchmark name or "all"')
    parser.add_argument('--name', type=str,
                       help='Dataset name (creates data/{benchmark}/{name}/)')
    parser.add_argument('--data-dir', '-d', type=str,
                       help='Base data directory (default: ../data)')
    parser.add_argument('--samples', '-n', type=int,
                       help='Number of samples to generate')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available benchmarks')
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        print("\nAvailable Benchmarks:")
        print("-" * 60)
        for name in GENERATORS:
            samples = DEFAULT_SAMPLES[name]
            print(f"  {name:12} - default {samples} samples")
        print()
        return 0
    
    # Check required arguments
    if not args.benchmark:
        parser.print_help()
        print("\nError: --benchmark/-b is required")
        return 1
    
    if not args.name:
        parser.print_help()
        print("\nError: --name is required (e.g., --name test1)")
        print("  This will create: data/{benchmark}/{name}/")
        return 1
    
    # Determine base data directory
    if args.data_dir:
        base_dir = Path(args.data_dir)
    else:
        # Default to ../data
        script_dir = Path(__file__).parent
        base_dir = script_dir.parent / 'data'
    
    # Generate for one or all benchmarks
    benchmarks = list(GENERATORS.keys()) if args.benchmark == 'all' else [args.benchmark]
    
    all_success = True
    for benchmark in benchmarks:
        # Structure: base_dir/{benchmark}/{name}/
        data_dir = base_dir / benchmark / args.name
        
        success, info = generate_benchmark_data(
            benchmark, data_dir, args.samples, args.seed
        )
        if not success:
            print(f"Error: {info.get('error', 'Unknown error')}")
            all_success = False
    
    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())
