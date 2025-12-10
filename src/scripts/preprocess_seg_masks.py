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


def collect_global_classes(input_dir, npy_files):
    """
    First pass: Collect ALL unique classes across ALL images.
    This ensures consistent mapping across the entire dataset.
    """
    all_classes = set()
    
    print("Pass 1: Collecting unique classes across all images...")
    for npy_file in tqdm(npy_files, desc="Scanning"):
        mask_path = os.path.join(input_dir, npy_file)
        mask = np.load(mask_path)
        all_classes.update(np.unique(mask).tolist())
    
    return sorted(all_classes)


def create_global_mapping(all_classes, max_classes=10):
    """
    Create a GLOBAL consistent mapping from original classes to 0-(max_classes-1).
    
    Strategy:
    - Class 0 stays as 0 (background)
    - Other classes are mapped to 1-(max_classes-1) based on frequency/order
    - Overflow classes are mapped to (max_classes-1)
    """
    mapping = {}
    
    # Background stays 0
    if 0 in all_classes:
        mapping[0] = 0
        all_classes = [c for c in all_classes if c != 0]
    
    # Map remaining classes
    for i, original_class in enumerate(all_classes):
        if i >= max_classes - 1:  # Reserve class 0 for background
            new_class = max_classes - 1  # Map overflow to last class
        else:
            new_class = i + 1
        mapping[original_class] = new_class
    
    return mapping


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
    """Process all .npy files in input directory with GLOBAL consistent mapping."""
    
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
    
    # PASS 1: Collect all unique classes globally
    all_classes = collect_global_classes(input_dir, npy_files)
    print(f"\nFound {len(all_classes)} unique classes across all images: {all_classes[:20]}{'...' if len(all_classes) > 20 else ''}")
    
    # Create GLOBAL mapping (same for ALL images!)
    global_mapping = create_global_mapping(all_classes, max_classes)
    
    print(f"\nGlobal class mapping (consistent across ALL images):")
    for orig, new in sorted(global_mapping.items())[:15]:
        print(f"  {orig:3d} → {new}")
    if len(global_mapping) > 15:
        print(f"  ... and {len(global_mapping) - 15} more")
    print()
    
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
    
    # Save mapping for reference
    mapping_file = os.path.join(output_dir, "class_mapping.txt")
    with open(mapping_file, 'w') as f:
        f.write("# Global class mapping (original -> new)\n")
        for orig, new in sorted(global_mapping.items()):
            f.write(f"{orig} -> {new}\n")
    
    print(f"\n✓ Processed {len(npy_files)} masks with CONSISTENT global mapping")
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

