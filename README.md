# 🌍 Waste Classification System

A complete deep learning solution for classifying waste into 5 categories: **Plastic**, **Paper**, **Metal**, **Glass**, and **Biological** (Organic).

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Web Interface](#web-interface)
- [Technical Details](#technical-details)

## ✨ Features

- ✅ **Complete Data Preprocessing Pipeline**
  - Image resizing to 256×256
  - Pixel normalization to [0, 1]
  - Label encoding with integer mapping
  - 70/15/15 train/validation/test split with stratification

- ✅ **Multiple Model Architectures**
  - Custom CNN built from scratch
  - Transfer learning with MobileNetV2 (recommended)
  - Transfer learning with EfficientNetB0

- ✅ **Advanced Training Features**
  - Data augmentation (rotation, flip, zoom, contrast)
  - Early stopping
  - Learning rate reduction
  - Model checkpointing

- ✅ **Beautiful Web Interface**
  - Drag-and-drop image upload
  - Real-time classification
  - Confidence scores for all categories
  - Responsive design

## 📁 Project Structure

```
d:\GARBAGE\
├── plastic/               # Plastic waste images (~1,500 images)
├── paper/                 # Paper waste images (~2,200 images)
├── metal/                 # Metal waste images (~1,000 images)
├── glass/                 # Glass waste images (~2,300 images)
├── biological/            # Organic waste images (~985 images)
├── preprocessed/          # Preprocessed data (generated)
│   ├── train_images.npz
│   ├── train_labels.npy
│   ├── val_images.npz
│   ├── val_labels.npy
│   ├── test_images.npz
│   ├── test_labels.npy
│   ├── label_mapping.json
│   ├── dataset_stats.json
│   └── metadata.json
├── models/                # Trained models (generated)
│   ├── *_best.h5
│   ├── *_final.h5
│   └── *_history.png
├── templates/             # Web interface templates
│   └── index.html
├── preprocess.py          # Data preprocessing script
├── train_model.py         # Model training script
├── inference.py           # Inference script
├── app.py                 # Flask web application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Data Preprocessing

### a) Image Resizing

All images are resized to **256×256** pixels.

**Justification:**
- ✅ **Balance between detail and compute**: Preserves enough detail to distinguish waste types while maintaining reasonable training time
- ✅ **Memory efficiency**: Lower than 512×512, allowing larger batch sizes
- ✅ **Transfer learning compatibility**: Standard input size for many pre-trained models
- ✅ **Training speed**: Faster than higher resolutions without sacrificing accuracy

### b) Normalization

Pixel values are scaled to **[0, 1]** using min-max normalization.

**Formula:** `normalized_pixel = pixel_value / 255.0`

**Justification:**
- ✅ Suitable for neural networks with ReLU activation
- ✅ Prevents vanishing/exploding gradients
- ✅ Faster convergence during training
- ✅ Standard practice in computer vision

### c) Label Encoding

Categories are mapped to integers:
```
glass = 0
metal = 1
paper = 2
plastic = 3
biological = 4
```

### d) Train/Validation/Test Split

**Split Ratios:**
- Training: 70% (for learning patterns)
- Validation: 15% (for hyperparameter tuning)
- Test: 15% (for final evaluation)

**Stratification:** ✅ Enabled
- Maintains class distribution across all splits
- Prevents class imbalance in any split
- Ensures reliable model evaluation

## 📦 Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

## 🚀 Usage

### Step 1: Preprocess Data

```bash
python preprocess.py
```

**Output:**
- Preprocessed images and labels in `preprocessed/` directory
- Dataset statistics and metadata
- Train/validation/test splits with stratification

**Expected time:** 5-10 minutes depending on dataset size

### Step 2: Train Model

```bash
python train_model.py
```

**Interactive Options:**
1. **Custom CNN** - Train from scratch
2. **MobileNetV2** - Transfer learning (recommended)
3. **EfficientNetB0** - Transfer learning

**Output:**
- Best model saved as `*_best.h5`
- Final model saved as `*_final.h5`
- Training history plot
- Model performance metrics

**Expected time:** 
- Custom CNN: 30-60 minutes
- Transfer learning: 15-30 minutes

### Step 3: Run Inference (Command Line)

```bash
python inference.py
```

Then enter image path when prompted:
```
Image path: d:\GARBAGE\plastic\plastic_1.jpg
```

### Step 4: Launch Web Interface

```bash
python app.py
```

Open browser and navigate to: **http://localhost:5000**

## 🎯 Model Training

### Custom CNN Architecture

```
Input (256×256×3)
    ↓
Data Augmentation
    ↓
Conv2D (32) → BatchNorm → Conv2D (32) → BatchNorm → MaxPool → Dropout
    ↓
Conv2D (64) → BatchNorm → Conv2D (64) → BatchNorm → MaxPool → Dropout
    ↓
Conv2D (128) → BatchNorm → Conv2D (128) → BatchNorm → MaxPool → Dropout
    ↓
Conv2D (256) → BatchNorm → Conv2D (256) → BatchNorm → MaxPool → Dropout
    ↓
Flatten → Dense (512) → BatchNorm → Dropout
    ↓
Dense (256) → BatchNorm → Dropout
    ↓
Dense (5, softmax)
```

### Transfer Learning

Uses pre-trained models (MobileNetV2 or EfficientNetB0) with:
- Frozen base layers
- Custom classification head
- Fine-tuning capability

### Training Configuration

- **Optimizer:** Adam (lr=0.001)
- **Loss:** Sparse Categorical Crossentropy
- **Batch Size:** 32
- **Epochs:** 50 (with early stopping)
- **Callbacks:**
  - ModelCheckpoint (save best model)
  - EarlyStopping (patience=10)
  - ReduceLROnPlateau (patience=5)

### Data Augmentation

Applied during training:
- Random horizontal flip
- Random rotation (±20%)
- Random zoom (±20%)
- Random contrast adjustment (±20%)

## 🌐 Web Interface

### Features

- 📸 **Drag-and-drop** image upload
- ⚡ **Real-time** classification
- 📊 **Confidence scores** for all categories
- 🎨 **Beautiful UI** with gradients and animations
- 📱 **Responsive design** for mobile devices

### Usage

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open browser: `http://localhost:5000`

3. Upload an image:
   - Click upload area to select file
   - Or drag and drop an image

4. View results:
   - Top prediction with highest confidence
   - All 5 category predictions with confidence scores

## 🔬 Technical Details

### Image Preprocessing Justification

**Why 256×256?**
- **Detail preservation:** Large enough to capture distinguishing features of waste types
- **Computational efficiency:** 4× faster than 512×512, 16× faster than 1024×1024
- **Memory optimization:** Allows batch size of 32-64 on GPUs with 8-16GB VRAM
- **Transfer learning compatibility:** Compatible with ImageNet pre-trained models

**Why [0, 1] normalization?**
- **Gradient stability:** Prevents exploding/vanishing gradients
- **Faster convergence:** Neural networks train better with normalized inputs
- **ReLU compatibility:** Works well with ReLU activation (outputs ≥ 0)
- **Standard practice:** Used in most computer vision models

### Dataset Statistics

| Category   | Images | Train (70%) | Val (15%) | Test (15%) |
|-----------|--------|-------------|-----------|------------|
| Plastic   | ~1,500 | ~1,050     | ~225      | ~225       |
| Paper     | ~2,200 | ~1,540     | ~330      | ~330       |
| Metal     | ~1,000 | ~700       | ~150      | ~150       |
| Glass     | ~2,300 | ~1,610     | ~345      | ~345       |
| Biological| ~985   | ~690       | ~148      | ~147       |
| **Total** | **~7,985** | **~5,590** | **~1,198** | **~1,197** |

### Model Performance Expectations

**Expected Accuracy:**
- Custom CNN: 85-90% test accuracy
- Transfer Learning (MobileNetV2): 90-95% test accuracy
- Transfer Learning (EfficientNetB0): 92-96% test accuracy

### File Formats

**Preprocessed Data:**
- Images: `.npz` (compressed numpy arrays)
- Labels: `.npy` (numpy arrays)
- Metadata: `.json` (JSON format)

**Models:**
- Keras HDF5 format (`.h5`)
- Can be converted to TensorFlow SavedModel or TFLite

## 🎓 Usage Examples

### Command Line Inference

```python
from inference import WasteClassifier

# Initialize classifier
classifier = WasteClassifier()

# Classify single image
result = classifier.predict('path/to/image.jpg')
print(result)

# Classify multiple images
results = classifier.predict_batch(['image1.jpg', 'image2.jpg'])
```

### Programmatic Usage

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('models/mobilenet_transfer_model_best.h5')

# Preprocess image
img = Image.open('test_image.jpg').convert('RGB')
img = img.resize((256, 256))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
class_id = np.argmax(predictions[0])

categories = ['glass', 'metal', 'paper', 'plastic', 'biological']
print(f"Predicted: {categories[class_id]}")
print(f"Confidence: {predictions[0][class_id] * 100:.2f}%")
```

## 🐛 Troubleshooting

### Issue: Out of memory during training

**Solution:** Reduce batch size in `train_model.py`:
```python
BATCH_SIZE = 16  # or 8
```

### Issue: Model not found

**Solution:** Train a model first:
```bash
python train_model.py
```

### Issue: Flask app won't start

**Solution:** Check if port 5000 is available:
```bash
netstat -an | findstr 5000
```

Or change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=8080)
```

## 📊 Performance Monitoring

Training history plots are saved to `models/*_history.png` showing:
- Training vs validation accuracy
- Training vs validation loss
- Helps identify overfitting/underfitting

## 🔄 Model Retraining

To retrain with new data:

1. Add images to category folders
2. Run preprocessing:
   ```bash
   python preprocess.py
   ```
3. Train new model:
   ```bash
   python train_model.py
   ```

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Suggestions and improvements are welcome!

## 📧 Support

For issues or questions, please check the troubleshooting section first.

---

**Built with ❤️ for a cleaner planet 🌍**
