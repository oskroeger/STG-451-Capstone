import os
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np

def preprocess_images(input_dir, output_dir, img_size=(224, 224), train_split=0.7, val_split=0.2):
    """
    Preprocesses images into train, validation, and test sets.

    Args:
        input_dir (str): Path to the dataset with subfolders for each country.
        output_dir (str): Path to save the preprocessed dataset.
        img_size (tuple): Target size for resizing images.
        train_split (float): Proportion of training data.
        val_split (float): Proportion of validation data.

    Returns:
        None
    """
    random.seed(42)
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Clear output directory
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for train, val, test
    splits = ['train', 'val', 'test']
    for split in splits:
        (output_path / split).mkdir(exist_ok=True)

    # Iterate through each country folder
    for country in os.listdir(input_path):
        country_path = input_path / country
        if not country_path.is_dir():
            continue

        # Gather image paths
        images = list(country_path.glob("*.jpg"))
        random.shuffle(images)

        # Split into train, val, test
        train_cutoff = int(len(images) * train_split)
        val_cutoff = train_cutoff + int(len(images) * val_split)

        data_splits = {
            'train': images[:train_cutoff],
            'val': images[train_cutoff:val_cutoff],
            'test': images[val_cutoff:]
        }

        # Process images
        for split, img_list in data_splits.items():
            split_dir = output_path / split / country
            split_dir.mkdir(parents=True, exist_ok=True)

            for img_path in img_list:
                try:
                    img = Image.open(img_path)
                    img = img.resize(img_size)
                    img.save(split_dir / img_path.name)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")

    print("Preprocessing complete! Dataset saved at:", output_path)

# Example usage
preprocess_images(
    input_dir="dataset/compressed_dataset",
    output_dir="dataset/processed_dataset"
)
