"""
Data Preprocessing Script for Waste Classification
Handles: Image resizing, normalization, label encoding, and train/val/test splitting
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import shutil
from tqdm import tqdm

# Configuration
IMG_SIZE = (256, 256)  # Chosen for balance between detail and computational efficiency
# 256x256 provides:
# - Sufficient detail to distinguish waste types
# - Reasonable training time (vs 512x512)
# - Good memory efficiency
# - Suitable for transfer learning models

CATEGORIES = ['glass', 'metal', 'paper', 'plastic', 'biological']
TRAIN_RATIO = 0.70  # 70% for training
VAL_RATIO = 0.15    # 15% for validation
TEST_RATIO = 0.15   # 15% for testing

DATA_DIR = r"d:\GARBAGE"
OUTPUT_DIR = os.path.join(DATA_DIR, "preprocessed")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_preprocess_image(image_path):
    """
    Load and preprocess a single image
    - Resize to IMG_SIZE
    - Convert to RGB (handle grayscale/RGBA)
    - Normalize pixel values to [0, 1]
    """
    try:
        img = Image.open(image_path)
        
        # Convert to RGB (handles grayscale and RGBA)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to consistent shape
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        return img_array
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def collect_dataset_info():
    """
    Collect all image paths and their labels
    Returns DataFrame with columns: filepath, category
    """
    data = []
    
    print("Collecting dataset information...")
    for category in CATEGORIES:
        category_path = os.path.join(DATA_DIR, category)
        if not os.path.exists(category_path):
            print(f"Warning: {category_path} not found!")
            continue
        
        # Get all jpg and png images
        image_files = [f for f in os.listdir(category_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            filepath = os.path.join(category_path, img_file)
            data.append({'filepath': filepath, 'category': category})
        
        print(f"  {category}: {len(image_files)} images")
    
    df = pd.DataFrame(data)
    print(f"\nTotal images: {len(df)}")
    return df


def create_label_encoder(categories):
    """
    Create and fit label encoder
    Labels: glass=0, metal=1, paper=2, plastic=3, biological=4
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(categories)
    
    # Save label mapping
    label_mapping = {label: int(idx) for idx, label in enumerate(label_encoder.classes_)}
    with open(os.path.join(OUTPUT_DIR, 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    print("\nLabel Encoding:")
    for label, idx in label_mapping.items():
        print(f"  {label} = {idx}")
    
    return label_encoder


def split_dataset(df, random_state=42):
    """
    Split dataset into train/validation/test with stratification
    Maintains class balance across all splits
    """
    print("\n--- Dataset Splitting with Stratification ---")
    
    # First split: separate test set (15%)
    train_val_df, test_df = train_test_split(
        df, 
        test_size=TEST_RATIO, 
        random_state=random_state,
        stratify=df['category']  # Preserve class distribution
    )
    
    # Second split: separate validation from train
    # 0.15 / 0.85 ≈ 0.176 to get 15% of original
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=train_val_df['category']
    )
    
    print(f"\nDataset Split:")
    print(f"  Training:   {len(train_df):5d} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df):5d} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:       {len(test_df):5d} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify stratification
    print("\nClass distribution per split:")
    for category in CATEGORIES:
        train_count = (train_df['category'] == category).sum()
        val_count = (val_df['category'] == category).sum()
        test_count = (test_df['category'] == category).sum()
        total = train_count + val_count + test_count
        
        print(f"  {category:12s}: Train={train_count:4d} ({train_count/total*100:.1f}%), "
              f"Val={val_count:4d} ({val_count/total*100:.1f}%), "
              f"Test={test_count:4d} ({test_count/total*100:.1f}%)")
    
    return train_df, val_df, test_df


def save_preprocessed_data(df, split_name, label_encoder):
    """
    Save preprocessed images and labels for a split
    Uses numpy's compressed format for efficient storage
    """
    print(f"\nProcessing {split_name} set...")
    
    images = []
    labels = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_array = load_and_preprocess_image(row['filepath'])
        if img_array is not None:
            images.append(img_array)
            labels.append(label_encoder.transform([row['category']])[0])
    
    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    # Save as compressed numpy files
    output_path_images = os.path.join(OUTPUT_DIR, f'{split_name}_images.npz')
    output_path_labels = os.path.join(OUTPUT_DIR, f'{split_name}_labels.npy')
    
    np.savez_compressed(output_path_images, images=images)
    np.save(output_path_labels, labels)
    
    print(f"  Saved {len(images)} images to {output_path_images}")
    print(f"  Shape: {images.shape}, Memory: {images.nbytes / 1024**2:.2f} MB")
    
    return images, labels


def generate_statistics(train_images, val_images, test_images):
    """
    Generate and save dataset statistics for normalization
    Calculates mean and std for standardization
    """
    print("\n--- Computing Dataset Statistics ---")
    
    # Compute mean and std across training set (used for normalization)
    train_mean = np.mean(train_images, axis=(0, 1, 2))
    train_std = np.std(train_images, axis=(0, 1, 2))
    
    stats = {
        'mean': train_mean.tolist(),
        'std': train_std.tolist(),
        'normalization_type': '[0, 1] scaling',
        'image_size': IMG_SIZE,
        'note': 'Images are already normalized to [0,1]. For mean-std normalization, use: (image - mean) / std'
    }
    
    with open(os.path.join(OUTPUT_DIR, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Train set mean (RGB): [{train_mean[0]:.4f}, {train_mean[1]:.4f}, {train_mean[2]:.4f}]")
    print(f"  Train set std  (RGB): [{train_std[0]:.4f}, {train_std[1]:.4f}, {train_std[2]:.4f}]")
    print(f"  Normalization: Pixel values scaled to [0, 1]")


def create_metadata():
    """
    Create metadata file with preprocessing details
    """
    metadata = {
        'image_size': IMG_SIZE,
        'categories': CATEGORIES,
        'num_classes': len(CATEGORIES),
        'normalization': {
            'method': 'min-max',
            'range': [0, 1],
            'formula': 'pixel_value / 255.0'
        },
        'split_ratios': {
            'train': TRAIN_RATIO,
            'validation': VAL_RATIO,
            'test': TEST_RATIO
        },
        'stratification': True,
        'preprocessing_steps': [
            '1. Convert to RGB',
            '2. Resize to 256x256',
            '3. Normalize to [0, 1]',
            '4. Train/Val/Test split with stratification'
        ],
        'choice_justification': {
            'image_size': '256x256 chosen for balance between detail preservation and computational efficiency',
            'normalization': '[0,1] normalization suitable for neural networks with ReLU activation',
            'split_strategy': 'Stratified split ensures balanced class distribution across all sets'
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✓ Metadata saved")


def main():
    """
    Main preprocessing pipeline
    """
    print("=" * 70)
    print("WASTE CLASSIFICATION - DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # Step 1: Collect dataset info
    df = collect_dataset_info()
    
    if len(df) == 0:
        print("Error: No images found!")
        return
    
    # Step 2: Create label encoder
    label_encoder = create_label_encoder(CATEGORIES)
    
    # Step 3: Split dataset with stratification
    train_df, val_df, test_df = split_dataset(df)
    
    # Step 4: Preprocess and save each split
    train_images, train_labels = save_preprocessed_data(train_df, 'train', label_encoder)
    val_images, val_labels = save_preprocessed_data(val_df, 'val', label_encoder)
    test_images, test_labels = save_preprocessed_data(test_df, 'test', label_encoder)
    
    # Step 5: Generate statistics
    generate_statistics(train_images, val_images, test_images)
    
    # Step 6: Create metadata
    create_metadata()
    
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - train_images.npz, train_labels.npy")
    print("  - val_images.npz, val_labels.npy")
    print("  - test_images.npz, test_labels.npy")
    print("  - label_mapping.json")
    print("  - dataset_stats.json")
    print("  - metadata.json")
    print("\n✓ Ready for model training!")


if __name__ == "__main__":
    main()
