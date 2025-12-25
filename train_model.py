"""
Waste Classification CNN Training Script
Trains a deep learning model for waste classification
"""

import os
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
DATA_DIR = r"d:\GARBAGE\preprocessed"
MODEL_DIR = r"d:\GARBAGE\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
IMG_SIZE = (256, 256)
NUM_CLASSES = 5


def load_data():
    """
    Load preprocessed data
    """
    print("Loading preprocessed data...")
    
    # Load training data
    train_images = np.load(os.path.join(DATA_DIR, 'train_images.npz'))['images']
    train_labels = np.load(os.path.join(DATA_DIR, 'train_labels.npy'))
    
    # Load validation data
    val_images = np.load(os.path.join(DATA_DIR, 'val_images.npz'))['images']
    val_labels = np.load(os.path.join(DATA_DIR, 'val_labels.npy'))
    
    # Load test data
    test_images = np.load(os.path.join(DATA_DIR, 'test_images.npz'))['images']
    test_labels = np.load(os.path.join(DATA_DIR, 'test_labels.npy'))
    
    print(f"  Training set:   {train_images.shape[0]} images")
    print(f"  Validation set: {val_images.shape[0]} images")
    print(f"  Test set:       {test_images.shape[0]} images")
    
    # Load label mapping
    with open(os.path.join(DATA_DIR, 'label_mapping.json'), 'r') as f:
        label_mapping = json.load(f)
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels), label_mapping


def create_data_augmentation():
    """
    Create data augmentation layer
    Improves model generalization
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ], name='data_augmentation')


def build_cnn_model(img_size=(256, 256, 3), num_classes=5):
    """
    Build a CNN model from scratch
    Architecture: Conv blocks + Dense layers
    """
    model = models.Sequential([
        # Data augmentation
        create_data_augmentation(),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=img_size),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def build_transfer_learning_model(img_size=(256, 256, 3), num_classes=5, base_model_name='MobileNetV2'):
    """
    Build model using transfer learning
    Options: MobileNetV2 or EfficientNetB0
    """
    # Load pre-trained base model
    if base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(
            input_shape=img_size,
            include_top=False,
            weights='imagenet'
        )
    else:  # EfficientNetB0
        base_model = EfficientNetB0(
            input_shape=img_size,
            include_top=False,
            weights='imagenet'
        )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build complete model
    inputs = keras.Input(shape=img_size)
    
    # Data augmentation
    x = create_data_augmentation()(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model


def plot_training_history(history, save_path):
   
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Training history plot saved to {save_path}")
    plt.close()


def evaluate_model(model, test_images, test_labels, label_mapping):
    """
    Evaluate model on test set
    """
    print("\n--- Model Evaluation on Test Set ---")
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Generate predictions
    predictions = model.predict(test_images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    for class_id in range(NUM_CLASSES):
        mask = test_labels == class_id
        class_acc = np.mean(predicted_classes[mask] == test_labels[mask])
        print(f"  {reverse_mapping[class_id]:12s}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    return test_accuracy


def save_model_info(model, test_accuracy, training_time, model_name):
    """
    Save model information
    """
    model_info = {
        'model_name': model_name,
        'architecture': model_name.split('_')[0],
        'test_accuracy': float(test_accuracy),
        'training_time_seconds': training_time,
        'image_size': IMG_SIZE,
        'num_classes': NUM_CLASSES,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(MODEL_DIR, f'{model_name}_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)


def main():
    """
    Main training pipeline
    """
    print("=" * 70)
    print("WASTE CLASSIFICATION - MODEL TRAINING")
    print("=" * 70)
    
    # Load data
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels), label_mapping = load_data()
    
    # Ask user for model choice
    print("\nSelect model architecture:")
    print("  1. Custom CNN (train from scratch)")
    print("  2. Transfer Learning - MobileNetV2 (recommended)")
    print("  3. Transfer Learning - EfficientNetB0")
    
    choice = input("\nEnter choice (1/2/3) [default: 2]: ").strip() or "2"
    
    if choice == "1":
        print("\nBuilding Custom CNN model...")
        model = build_cnn_model(img_size=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)
        model_name = "custom_cnn_model"
    elif choice == "3":
        print("\nBuilding EfficientNetB0 transfer learning model...")
        model = build_transfer_learning_model(img_size=(*IMG_SIZE, 3), num_classes=NUM_CLASSES, base_model_name='EfficientNetB0')
        model_name = "efficientnet_transfer_model"
    else:
        print("\nBuilding MobileNetV2 transfer learning model...")
        model = build_transfer_learning_model(img_size=(*IMG_SIZE, 3), num_classes=NUM_CLASSES, base_model_name='MobileNetV2')
        model_name = "mobilenet_transfer_model"
    
    # Model summary
    print("\n--- Model Architecture ---")
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_DIR, f'{model_name}_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\n--- Training {model_name} ---")
    start_time = datetime.now()
    
    history = model.fit(
        train_images, train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_images, val_labels),
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Plot training history
    plot_path = os.path.join(MODEL_DIR, f'{model_name}_history.png')
    plot_training_history(history, plot_path)
    
    # Evaluate on test set
    test_accuracy = evaluate_model(model, test_images, test_labels, label_mapping)
    
    # Save final model
    final_model_path = os.path.join(MODEL_DIR, f'{model_name}_final.h5')
    model.save(final_model_path)
    print(f"\n✓ Final model saved to {final_model_path}")
    
    # Save model info
    save_model_info(model, test_accuracy, training_time, model_name)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nBest model: {os.path.join(MODEL_DIR, f'{model_name}_best.h5')}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
