import os
import shutil
import random

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def split_data(clear_dir, hazy_dir, output_root, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)

    # Get list of filenames
    filenames = [f for f in os.listdir(clear_dir) if f.endswith('.png')]
    filenames.sort()
    random.shuffle(filenames)

    # Split
    total = len(filenames)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_files = filenames[:n_train]
    val_files = filenames[n_train:n_train + n_val]
    test_files = filenames[n_train + n_val:]

    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    # Copy files
    for split, files in splits.items():
        hazy_out = os.path.join(output_root, split, 'hazy')
        clear_out = os.path.join(output_root, split, 'clear')
        create_dir(hazy_out)
        create_dir(clear_out)

        for fname in files:
            shutil.copy(os.path.join(hazy_dir, fname), os.path.join(hazy_out, fname))
            shutil.copy(os.path.join(clear_dir, fname), os.path.join(clear_out, fname))

    print(f"Split complete: {n_train} train, {n_val} val, {len(test_files)} test")

if __name__ == "__main__":
    clear_dir = './datasets/dota_train_part1'
    hazy_dir = './datasets/dota_train_part1_hazed'
    output_root = './datasets/dotah'

    split_data(clear_dir, hazy_dir, output_root)
