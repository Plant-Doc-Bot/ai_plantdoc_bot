"""
Create a stratified validation split from data/train into data/val.
Moves ~ratio of images per class into a matching subfolder under data/val.

Usage:
  python scripts/make_val_split.py --ratio 0.2 --seed 42
"""
import argparse
import random
from pathlib import Path
import shutil

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def collect_images(class_dir):
    return [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]


def main(train_dir, val_dir, ratio, seed):
    random.seed(seed)
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    val_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for class_path in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        images = collect_images(class_path)
        if not images:
            continue

        random.shuffle(images)
        n_val = max(1, int(len(images) * ratio))
        val_subset = images[:n_val]

        target_dir = val_dir / class_path.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for img_path in val_subset:
            dest = target_dir / img_path.name
            # If a file with the same name already exists, keep original to avoid loss.
            if dest.exists():
                continue
            shutil.move(str(img_path), str(dest))

        summary.append((class_path.name, n_val, len(images) - n_val))

    print("Completed stratified move:")
    for cls, moved, remaining in summary:
        print(f"{cls:35} -> val: {moved:4d} | train: {remaining:4d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a validation split from train data.")
    parser.add_argument("--train_dir", default="data/train", help="Path to train directory.")
    parser.add_argument("--val_dir", default="data/val", help="Destination validation directory.")
    parser.add_argument("--ratio", type=float, default=0.2, help="Fraction of each class to move to val.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    main(args.train_dir, args.val_dir, args.ratio, args.seed)
