"""
Flask Web Application for Waste Classification
Upload images and get real-time classification results
"""

from flask import Flask, render_template, request, jsonify
import os
import base64
from io import BytesIO
from PIL import Image
import sys

# Add parent directory to path to import inference module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import WasteClassifier

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize classifier
print("Initializing waste classifier...")
classifier = WasteClassifier()
print("✓ Classifier ready!")


@app.route('/')
def index():
    """
    Main page
    """
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    """
    Classify uploaded image
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        try:
            # Save uploaded file
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            result = classifier.predict(filepath, top_k=5)
            
            # Convert image to base64 for display
            img = Image.open(filepath)
            img.thumbnail((400, 400))
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'image': f"data:image/jpeg;base64,{img_str}",
                'predictions': result['predictions']
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/info')
def info():
    """
    Information about the model and categories
    """
    return jsonify({
        'categories': list(classifier.label_mapping.keys()),
        'model_info': 'Deep Learning CNN for waste classification',
        'categories_info': {
            'plastic': 'Plastic bottles, containers, packaging',
            'paper': 'Newspapers, cardboard, paper products',
            'metal': 'Aluminum cans, metal containers',
            'glass': 'Glass bottles, jars',
            'biological': 'Organic waste, food scraps, compostable materials'
        }
    })


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("WASTE CLASSIFICATION WEB APPLICATION")
    print("=" * 70)
    print("\nStarting Flask server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
