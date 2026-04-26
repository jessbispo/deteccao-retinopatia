import os
import argparse
import shutil
import csv
import random

def prepare_data(args):
    """
    Organizes images into a directory structure for training and validation,
    maintaining class proportions (stratified split).
    """
    
    # 1. Load data from CSV
    print(f"Reading CSV: {args.csv_path}")
    data = []
    try:
        with open(args.csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                id_code = row['id_code']
                diagnosis = int(row['diagnosis'])
                # Binary mapping: 0 -> normal, >=1 -> retinopatia
                label = "normal" if diagnosis == 0 else "retinopatia"
                data.append({'id_code': id_code, 'label': label})
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 2. Group by class for stratified split
    by_class = {"normal": [], "retinopatia": []}
    for item in data:
        by_class[item['label']].append(item)

    train_data = []
    val_data = []

    # 3. Perform stratified split
    for class_name, samples in by_class.items():
        random.shuffle(samples)
        split_idx = int(len(samples) * args.train_split)
        train_data.extend(samples[:split_idx])
        val_data.extend(samples[split_idx:])

    print(f"Split completed: {len(train_data)} training, {len(val_data)} validation")
    
    # 4. Create directory structure
    splits = {"train": train_data, "val": val_data}
    classes = ["normal", "retinopatia"]
    
    for split_name in splits:
        for class_name in classes:
            path = os.path.join(args.output_dir, split_name, class_name)
            os.makedirs(path, exist_ok=True)

    # 5. Copy images
    print("Copying images...")
    
    for split_name, samples in splits.items():
        print(f"  Organizing {split_name} set...")
        
        if args.limit and args.limit > 0:
             samples = samples[:args.limit]

        extensions = ['.png', '.jpg', '.jpeg']

        for i, item in enumerate(samples):
            if (i + 1) % 100 == 0:
                print(f"    - Processed {i+1}/{len(samples)} images...")
            
            id_code = item['id_code']
            label = item['label']
            
            found = False
            for ext in extensions:
                src_path = os.path.join(args.image_dir, id_code + ext)
                if os.path.exists(src_path):
                    dst_path = os.path.join(args.output_dir, split_name, label, id_code + ext)
                    shutil.copy2(src_path, dst_path)
                    found = True
                    break
            
            if not found:
                # Optional: warn if image not found
                pass

    print(f"\nDone! Dataset organized in: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Organizer - Retinopathy")
    parser.add_argument("--csv_path", type=str, default="data/raw/train.csv", help="Path to the labels CSV")
    parser.add_argument("--image_dir", type=str, default="data/raw/train_images", help="Directory with raw images")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--train_split", type=float, default=0.8, help="Training split ratio (0.0 to 1.0)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of processed images (for testing)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")

    args = parser.parse_args()
    
    if args.seed:
        random.seed(args.seed)
        
    prepare_data(args)
