"""
Flask web application for CIFAR-10 CNN image classification
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from PIL import Image
import traceback

from config import Config
from inference import ModelInference

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)

# Enable CORS
CORS(app, origins=app.config['CORS_ORIGINS'])

# Initialize inference engine (lazy loading)
inference_engine = None


def get_inference_engine():
    """
    Get or initialize the inference engine
    
    Returns:
        ModelInference instance
    """
    global inference_engine
    
    if inference_engine is None:
        checkpoint_path = app.config['MODEL_CHECKPOINT']
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {checkpoint_path}. "
                "Please train the model first using: python main.py --mode train --epochs 50"
            )
        
        inference_engine = ModelInference(checkpoint_path)
        inference_engine.load_model()
    
    return inference_engine


def allowed_file(filename):
    """
    Check if file extension is allowed
    
    Args:
        filename: Name of the file
        
    Returns:
        True if allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """
    Render the main page
    """
    return render_template('index.html')


@app.route('/health')
def health():
    """
    Health check endpoint
    """
    try:
        checkpoint_exists = os.path.exists(app.config['MODEL_CHECKPOINT'])
        model_loaded = inference_engine is not None
        
        return jsonify({
            'status': 'healthy',
            'checkpoint_exists': checkpoint_exists,
            'model_loaded': model_loaded
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Accepts image file and returns predictions
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
            }), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and validate image
            image = Image.open(filepath).convert('RGB')
            
            # Get inference engine
            engine = get_inference_engine()
            
            # Make prediction
            result = engine.predict(image, top_k=5)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'prediction': result['top_prediction'],
                'top_predictions': result['top_k_predictions'],
                'all_predictions': result['all_predictions']
            })
            
        except Exception as e:
            # Clean up uploaded file if it exists
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e
            
    except FileNotFoundError as e:
        return jsonify({
            'error': 'Model not found',
            'message': str(e)
        }), 503
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/classes')
def get_classes():
    """
    Get list of CIFAR-10 classes
    """
    from inference import CIFAR10_CLASSES
    
    return jsonify({
        'classes': CIFAR10_CLASSES,
        'num_classes': len(CIFAR10_CLASSES)
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """
    Handle file too large error
    """
    return jsonify({
        'error': 'File too large',
        'message': f'Maximum file size is {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)} MB'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 errors
    """
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """
    Handle 500 errors
    """
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("CIFAR-10 CNN Flask Application")
    print("=" * 70)
    print(f"Server starting on http://{app.config['HOST']}:{app.config['PORT']}")
    print(f"Model checkpoint: {app.config['MODEL_CHECKPOINT']}")
    
    # Check if model checkpoint exists
    if os.path.exists(app.config['MODEL_CHECKPOINT']):
        print("✓ Model checkpoint found")
    else:
        print("✗ Model checkpoint not found")
        print("  Train the model first: python main.py --mode train --epochs 50")
    
    print("=" * 70)
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )
