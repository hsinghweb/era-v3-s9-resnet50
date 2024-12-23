import os
import shutil
from typing import Dict

def load_val_annotations(val_anno_path: str) -> Dict[str, str]:
    """Load validation annotations file"""
    img_to_class = {}
    with open(val_anno_path, 'r') as f:
        for line in f:
            img_name, class_id = line.strip().split()
            img_to_class[img_name] = class_id
    return img_to_class

def organize_val_set(val_dir: str, val_anno_path: str):
    """Organize validation images into class folders"""
    annotations = load_val_annotations(val_anno_path)
    
    # Create class directories
    for class_id in set(annotations.values()):
        os.makedirs(os.path.join(val_dir, class_id), exist_ok=True)
    
    # Move images to respective class directories
    for img_name, class_id in annotations.items():
        src_path = os.path.join(val_dir, img_name)
        dst_path = os.path.join(val_dir, class_id, img_name)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir', type=str, required=True,
                       help='Path to validation directory')
    parser.add_argument('--val_anno', type=str, required=True,
                       help='Path to ILSVRC2012_validation_ground_truth.txt')
    args = parser.parse_args()
    
    organize_val_set(args.val_dir, args.val_anno) 