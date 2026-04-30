#!/usr/bin/env python3
"""
LISA Traffic Light Dataset → YOLO Format Converter

Converts Supervisly-format JSON annotations to YOLO bounding box format.
Creates train/val/test splits with normalized coordinates.

Usage:
    python scripts/prepare_dataset.py --dataset-dir Dataset --output-dir data/processed --val-ratio 0.1
"""

import json
import os
import sys
import argparse
import random
from pathlib import Path
from PIL import Image
import shutil

# Class mapping from meta.json (in order)
CLASS_NAMES = [
    "go",
    "go forward",
    "go forward traffic light",
    "go left",
    "go left traffic light",
    "go traffic light",
    "stop",
    "stop left",
    "stop left traffic light",
    "stop traffic light",
    "warning",
    "warning left",
    "warning left traffic light",
    "warning traffic light"
]

CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def load_meta(meta_path):
    """Load and verify meta.json"""
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"✓ Loaded meta.json with {len(meta['classes'])} classes")
    return meta


def convert_annotation_to_yolo(ann_file, class_map):
    """
    Convert Supervisly JSON annotation to YOLO format lines.
    Returns list of YOLO lines: "class_id x_center y_center width height"
    """
    with open(ann_file) as f:
        ann_data = json.load(f)
    
    img_h = ann_data["size"]["height"]
    img_w = ann_data["size"]["width"]
    
    yolo_lines = []
    for obj in ann_data.get("objects", []):
        class_title = obj.get("classTitle", "").strip()
        if not class_title or class_title not in class_map:
            print(f"  WARNING: Unknown class '{class_title}' in {ann_file}")
            continue
        
        pts = obj["points"]["exterior"]
        if len(pts) != 2:
            print(f"  WARNING: Expected 2 points, got {len(pts)} in {ann_file}")
            continue
        
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        
        # Normalize to [0, 1]
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        
        x_center = ((x_min + x_max) / 2.0) / img_w
        y_center = ((y_min + y_max) / 2.0) / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h
        
        # Clamp to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        class_id = class_map[class_title]
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)
    
    return yolo_lines


def process_split(split_name, img_dir, ann_dir, out_img_dir, out_lbl_dir, class_map, copy_images=False):
    """
    Process one split (train/test).
    Returns list of processed image names.
    """
    print(f"\nProcessing {split_name}...")
    
    # Find all annotation files
    ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".json")])
    processed = []
    
    for ann_file in ann_files:
        ann_path = os.path.join(ann_dir, ann_file)
        img_name = ann_file.replace(".json", "")  # e.g., "dayClip10--00000.jpg"
        img_path = os.path.join(img_dir, img_name)
        
        if not os.path.exists(img_path):
            # Try without the extra suffix in case naming differs
            img_base = os.path.splitext(ann_file)[0]  # e.g., "dayClip10--00000.jpg"
            img_path_alt = os.path.join(img_dir, img_base)
            if not os.path.exists(img_path_alt):
                print(f"  WARNING: Image not found for {ann_file}")
                continue
            img_path = img_path_alt
        
        try:
            yolo_lines = convert_annotation_to_yolo(ann_path, class_map)
        except Exception as e:
            print(f"  ERROR processing {ann_file}: {e}")
            continue
        
        # Copy or symlink image
        out_img_path = os.path.join(out_img_dir, img_name)
        os.makedirs(out_img_dir, exist_ok=True)
        if not os.path.exists(out_img_path):
            if copy_images:
                shutil.copy2(img_path, out_img_path)
            else:
                os.symlink(os.path.abspath(img_path), out_img_path)
        
        # Write label file
        out_lbl_dir_path = out_lbl_dir
        os.makedirs(out_lbl_dir_path, exist_ok=True)
        lbl_name = img_name + ".txt"  # e.g., "dayClip10--00000.jpg.txt"
        out_lbl_path = os.path.join(out_lbl_dir_path, lbl_name)
        with open(out_lbl_path, "w") as f:
            f.write("\n".join(yolo_lines))
            if yolo_lines:
                f.write("\n")
        
        processed.append(img_name)
    
    print(f"  ✓ Processed {len(processed)} images")
    return processed


def create_val_split(train_imgs, train_out_img, train_out_lbl, val_out_img, val_out_lbl, val_ratio=0.1, copy_images=False):
    """
    Sample n% of train images and move to val split.
    """
    print(f"\nCreating validation split ({int(val_ratio*100)}% of train)...")
    
    n_val = max(1, int(len(train_imgs) * val_ratio))
    random.seed(42)
    val_imgs = set(random.sample(train_imgs, n_val))
    
    os.makedirs(val_out_img, exist_ok=True)
    os.makedirs(val_out_lbl, exist_ok=True)
    
    moved = 0
    for img_name in val_imgs:
        # Move image
        src_img = os.path.join(train_out_img, img_name)
        dst_img = os.path.join(val_out_img, img_name)
        if os.path.exists(src_img):
            if os.path.islink(src_img):
                os.remove(src_img)
                os.symlink(os.readlink(os.path.join(train_out_img, img_name)), dst_img)
            else:
                shutil.move(src_img, dst_img)
        
        # Move label
        lbl_name = img_name + ".txt"
        src_lbl = os.path.join(train_out_lbl, lbl_name)
        dst_lbl = os.path.join(val_out_lbl, lbl_name)
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
        
        moved += 1
    
    print(f"  ✓ Moved {moved} images to validation split")


def create_data_yaml(output_dir, class_names):
    """Create data.yaml for YOLOv8"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    
    lines = [
        "path: data/processed",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "nc: " + str(len(class_names)),
        "names:",
    ]
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")
    
    with open(yaml_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"\n✓ Created {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert LISA dataset to YOLO format")
    parser.add_argument("--dataset-dir", default="Dataset", help="Path to raw LISA Dataset folder")
    parser.add_argument("--output-dir", default="data/processed", help="Path to output processed folder")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Ratio of train to use as validation")
    parser.add_argument("--copy-images", action="store_true", help="Copy images (default: symlink)")
    
    args = parser.parse_args()
    
    # Validate paths
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    # Load meta
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        print(f"ERROR: meta.json not found: {meta_path}")
        sys.exit(1)
    
    meta = load_meta(meta_path)
    
    # Process train split
    train_imgs = process_split(
        "train",
        img_dir=str(dataset_dir / "train" / "img"),
        ann_dir=str(dataset_dir / "train" / "ann"),
        out_img_dir=str(output_dir / "images" / "train"),
        out_lbl_dir=str(output_dir / "labels" / "train"),
        class_map=CLASS_MAP,
        copy_images=args.copy_images
    )
    
    # Process test split
    test_imgs = process_split(
        "test",
        img_dir=str(dataset_dir / "test" / "img"),
        ann_dir=str(dataset_dir / "test" / "ann"),
        out_img_dir=str(output_dir / "images" / "test"),
        out_lbl_dir=str(output_dir / "labels" / "test"),
        class_map=CLASS_MAP,
        copy_images=args.copy_images
    )
    
    # Create val split from train
    create_val_split(
        train_imgs,
        str(output_dir / "images" / "train"),
        str(output_dir / "labels" / "train"),
        str(output_dir / "images" / "val"),
        str(output_dir / "labels" / "val"),
        val_ratio=args.val_ratio,
        copy_images=args.copy_images
    )
    
    # Create data.yaml
    create_data_yaml(str(output_dir), CLASS_NAMES)
    
    # Summary
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    train_count = len([f for f in os.listdir(output_dir / "images" / "train") if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    val_count = len([f for f in os.listdir(output_dir / "images" / "val") if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    test_count = len([f for f in os.listdir(output_dir / "images" / "test") if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    
    print(f"Train: {train_count} images")
    print(f"Val:   {val_count} images")
    print(f"Test:  {test_count} images")
    print(f"Total: {train_count + val_count + test_count} images")
    print("="*60)
    print(f"Ready to train with: yolo detect train data={output_dir}/data.yaml ...")


if __name__ == "__main__":
    main()
