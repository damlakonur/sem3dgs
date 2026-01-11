#!/usr/bin/env python3
"""
Preprocess segmentation masks: clip class IDs to 0-9 range with GLOBAL consistent mapping.
Usage: python preprocess_seg_masks.py --input /path/to/seg_masks --output /path/to/output
"""

import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm


def collect_global_classes_with_frequency(input_dir, npy_files):
    """
    First pass: Collect ALL unique classes across ALL images WITH pixel counts.
    This ensures we can map the most FREQUENT classes to dedicated slots.
    """
    from collections import Counter
    class_pixel_counts = Counter()
    
    print("Pass 1: Collecting class frequencies across all images...")
    for npy_file in tqdm(npy_files, desc="Scanning"):
        mask_path = os.path.join(input_dir, npy_file)
        mask = np.load(mask_path)
        unique, counts = np.unique(mask, return_counts=True)
        for cls, cnt in zip(unique, counts):
            class_pixel_counts[cls] += cnt
    
    return class_pixel_counts


def create_global_mapping(class_pixel_counts, max_classes=10):
    """
    Create a GLOBAL consistent mapping from original classes to 0-(max_classes-1).
    
    Strategy (FREQUENCY-BASED):
    - The top (max_classes) most frequent classes get their own dedicated slots 0-(max_classes-1)
    - All other classes are mapped to (max_classes-1) as "other"
    
    This ensures semantically important (frequent) classes are preserved!
    """
    mapping = {}
    
    # Sort classes by frequency (most frequent first)
    sorted_classes = sorted(class_pixel_counts.keys(), 
                           key=lambda c: class_pixel_counts[c], 
                           reverse=True)
    
    # Top N classes get dedicated slots
    top_classes = sorted_classes[:max_classes]
    other_classes = sorted_classes[max_classes:]
    
    # Assign new IDs: most frequent = 0, second most = 1, etc.
    for new_id, original_class in enumerate(top_classes):
        mapping[original_class] = new_id
    
    # All remaining classes map to (max_classes - 1) as "other"
    # But if we already have max_classes top classes, they share the last slot
    for original_class in other_classes:
        mapping[original_class] = max_classes - 1
    
    return mapping, top_classes, other_classes


def apply_mapping(mask, global_mapping):
    """Apply the global mapping to a single mask."""
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    
    for original_class, new_class in global_mapping.items():
        new_mask[mask == original_class] = new_class
    
    # Handle any classes not in the mapping (shouldn't happen, but just in case)
    # Map them to the last class
    unmapped = np.setdiff1d(np.unique(mask), list(global_mapping.keys()))
    if len(unmapped) > 0:
        max_new_class = max(global_mapping.values())
        for c in unmapped:
            new_mask[mask == c] = max_new_class
    
    return new_mask


def process_directory(input_dir, output_dir, max_classes=10, save_png=True, resize=None):
    """Process all .npy files in input directory with GLOBAL consistent FREQUENCY-BASED mapping."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .npy files
    npy_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    
    if len(npy_files) == 0:
        print(f"No .npy files found in {input_dir}")
        return
    
    print(f"Found {len(npy_files)} segmentation masks")
    print(f"Output directory: {output_dir}")
    print(f"Max classes: {max_classes} (0 to {max_classes-1})")
    if resize:
        print(f"Resize to: {resize[0]}x{resize[1]}")
    print()
    
    # PASS 1: Collect all unique classes with FREQUENCY counts globally
    class_pixel_counts = collect_global_classes_with_frequency(input_dir, npy_files)
    total_pixels = sum(class_pixel_counts.values())
    
    print(f"\nFound {len(class_pixel_counts)} unique classes across all images")
    
    # Create GLOBAL FREQUENCY-BASED mapping (same for ALL images!)
    global_mapping, top_classes, other_classes = create_global_mapping(class_pixel_counts, max_classes)
    
    print(f"\n{'='*60}")
    print(f"FREQUENCY-BASED CLASS MAPPING (top {max_classes} most common classes)")
    print(f"{'='*60}")
    print(f"{'New ID':<8} {'Original ID':<12} {'Pixels':<15} {'% of Dataset':<12}")
    print(f"{'-'*60}")
    for new_id, orig_class in enumerate(top_classes):
        pct = 100 * class_pixel_counts[orig_class] / total_pixels
        print(f"{new_id:<8} {orig_class:<12} {class_pixel_counts[orig_class]:<15,} {pct:.2f}%")
    
    if other_classes:
        other_pixels = sum(class_pixel_counts[c] for c in other_classes)
        other_pct = 100 * other_pixels / total_pixels
        print(f"{'-'*60}")
        print(f"Classes merged into '{max_classes-1}' (other): {len(other_classes)} classes, {other_pixels:,} pixels ({other_pct:.2f}%)")
    print(f"{'='*60}\n")
    
    # PASS 2: Apply mapping to all images
    print("Pass 2: Applying consistent mapping to all images...")
    for idx, npy_file in enumerate(tqdm(npy_files, desc="Processing")):
        # Load mask
        mask_path = os.path.join(input_dir, npy_file)
        mask = np.load(mask_path)
        
        # Downsample if requested (BEFORE mapping to preserve details)
        if resize:
            mask_pil = Image.fromarray(mask.astype(np.uint8))
            mask_pil_resized = mask_pil.resize((resize[1], resize[0]), Image.NEAREST)
            mask = np.array(mask_pil_resized)
        
        # Apply GLOBAL mapping
        clipped_mask = apply_mapping(mask, global_mapping)
        
        # Save as .npy
        output_npy = os.path.join(output_dir, npy_file)
        np.save(output_npy, clipped_mask)
        
        # Optionally save as PNG for visualization
        if save_png:
            vis_mask = (clipped_mask.astype(float) / (max_classes - 1) * 255).astype(np.uint8)
            png_file = npy_file.replace('.npy', '.png')
            output_png = os.path.join(output_dir, png_file)
            Image.fromarray(vis_mask).save(output_png)
    
    # Save detailed mapping for reference
    mapping_file = os.path.join(output_dir, "class_mapping.txt")
    with open(mapping_file, 'w') as f:
        f.write("# FREQUENCY-BASED Global Class Mapping\n")
        f.write(f"# Total original classes: {len(class_pixel_counts)}\n")
        f.write(f"# Mapped to: {max_classes} classes (0-{max_classes-1})\n")
        f.write("#\n")
        f.write("# Top classes (by pixel frequency):\n")
        for new_id, orig_class in enumerate(top_classes):
            pct = 100 * class_pixel_counts[orig_class] / total_pixels
            f.write(f"{orig_class} -> {new_id}  # {class_pixel_counts[orig_class]:,} pixels ({pct:.2f}%)\n")
        f.write("#\n")
        f.write(f"# Merged into class {max_classes-1} ({len(other_classes)} classes):\n")
        for orig_class in other_classes[:50]:  # Show first 50
            f.write(f"{orig_class} -> {max_classes-1}\n")
        if len(other_classes) > 50:
            f.write(f"# ... and {len(other_classes) - 50} more classes\n")
    
    print(f"\n✓ Processed {len(npy_files)} masks with CONSISTENT FREQUENCY-BASED mapping")
    print(f"✓ Saved to {output_dir}")
    print(f"✓ Mapping saved to {mapping_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clip segmentation masks to N classes and optionally resize")
    parser.add_argument("--input", type=str, required=True, help="Input directory with .npy masks")
    parser.add_argument("--output", type=str, required=True, help="Output directory for clipped masks")
    parser.add_argument("--max_classes", type=int, default=10, help="Maximum number of classes (default: 10)")
    parser.add_argument("--no_png", action="store_true", help="Don't save PNG visualizations")
    parser.add_argument("--resize", type=str, default=None, 
                       help="Resize to HxW (e.g., '260,389' for 260 height, 389 width)")
    
    args = parser.parse_args()
    
    # Parse resize argument
    resize = None
    if args.resize:
        try:
            h, w = map(int, args.resize.split(','))
            resize = (h, w)
            print(f"Will resize to: {h} x {w}")
        except:
            print(f"Invalid resize format: {args.resize}. Use 'HEIGHT,WIDTH' (e.g., '260,389')")
            exit(1)
    
    process_directory(
        args.input,
        args.output,
        max_classes=args.max_classes,
        save_png=not args.no_png,
        resize=resize
    )

