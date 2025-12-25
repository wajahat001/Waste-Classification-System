import os
import numpy as np
import json
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Configuration
MODEL_DIR = r"d:\GARBAGE\models"
DATA_DIR = r"d:\GARBAGE\preprocessed"
IMG_SIZE = (256, 256)


class WasteClassifier:
    """
    Waste classifier for inference
    """
    
    def __init__(self, model_path=None):
        """
        Initialize classifier
        
        Args:
            model_path: Path to trained model. If None, looks for best model in MODEL_DIR
        """
        # Load label mapping
        label_mapping_path = os.path.join(DATA_DIR, 'label_mapping.json')
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.load(f)
        
        # Create reverse mapping (index -> category name)
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        
        # Load model
        if model_path is None:
            # Find best model
            model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_best.h5')]
            if not model_files:
                raise ValueError("No trained model found! Please train a model first.")
            model_path = os.path.join(MODEL_DIR, model_files[0])
        
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully!")
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for inference
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_path, top_k=3):
        """
        Predict waste category for an image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top k predictions
        top_indices = np.argsort(predictions)[::-1][:top_k]
        
        results = {
            'image_path': image_path,
            'predictions': []
        }
        
        for idx in top_indices:
            category = self.reverse_mapping[idx]
            confidence = float(predictions[idx])
            results['predictions'].append({
                'category': category,
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.2f}%"
            })
        
        return results
    
    def predict_batch(self, image_paths):
        """
        Predict waste categories for multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def print_prediction(self, image_path):
        """
        Predict and print results in a formatted way
        """
        result = self.predict(image_path)
        
        print(f"\n{'='*60}")
        print(f"Image: {os.path.basename(result['image_path'])}")
        print(f"{'='*60}")
        print("\nPredictions:")
        
        for i, pred in enumerate(result['predictions'], 1):
            print(f"  {i}. {pred['category'].upper():12s} - {pred['confidence_percent']:>7s} confidence")
        
        # Highlight top prediction
        top_pred = result['predictions'][0]
        print(f"\n✓ Classification: {top_pred['category'].upper()}")
        print(f"  Confidence: {top_pred['confidence_percent']}")


def main():
    """
    Main inference demonstration
    """
    print("=" * 70)
    print("WASTE CLASSIFICATION - INFERENCE")
    print("=" * 70)
    
    # Initialize classifier
    try:
        classifier = WasteClassifier()
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Interactive mode
    print("\nEnter image path to classify (or 'quit' to exit):")
    print("Example: d:\\GARBAGE\\plastic\\plastic_1.jpg")
    
    while True:
        image_path = input("\nImage path: ").strip().strip('"')
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            break
        
        if not os.path.exists(image_path):
            print(f"Error: File not found - {image_path}")
            continue
        
        try:
            classifier.print_prediction(image_path)
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThank you for using the Waste Classifier!")


if __name__ == "__main__":
    main()
