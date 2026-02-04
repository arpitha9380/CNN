# Flask Deployment Guide

## 🚀 Quick Start

### 1. Install Flask Dependencies

```bash
pip install flask flask-cors werkzeug
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train the Model (if not already done)

```bash
python main.py --mode train --epochs 50 --save-plots
```

This will create `checkpoints/best_model.pth` which the Flask app needs.

### 3. Start the Flask Server

```bash
python app.py
```

The server will start at `http://localhost:5000`

### 4. Open in Browser

Navigate to `http://localhost:5000` in your web browser.

---

## 📋 Features

### Web Interface
- 🖼️ **Drag & Drop Upload** - Drag images or click to browse
- 👁️ **Image Preview** - See your image before prediction
- 🎯 **Real-time Predictions** - Instant classification results
- 📊 **Confidence Scores** - See prediction confidence for all classes
- 📱 **Responsive Design** - Works on desktop and mobile

### API Endpoints

#### `GET /`
Main web interface

#### `POST /predict`
Make predictions on uploaded images

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "success": true,
  "prediction": {
    "class": "cat",
    "class_id": 3,
    "confidence": 87.45
  },
  "top_predictions": [
    {"class": "cat", "class_id": 3, "confidence": 87.45},
    {"class": "dog", "class_id": 5, "confidence": 8.32},
    ...
  ],
  "all_predictions": {
    "airplane": 0.12,
    "automobile": 0.45,
    ...
  }
}
```

#### `GET /health`
Check server and model status

**Response:**
```json
{
  "status": "healthy",
  "checkpoint_exists": true,
  "model_loaded": true
}
```

#### `GET /classes`
Get list of CIFAR-10 classes

**Response:**
```json
{
  "classes": ["airplane", "automobile", "bird", ...],
  "num_classes": 10
}
```

---

## 🧪 Testing the API

### Using cURL

```bash
# Make a prediction
curl -X POST -F "file=@path/to/image.jpg" http://localhost:5000/predict

# Check health
curl http://localhost:5000/health

# Get classes
curl http://localhost:5000/classes
```

### Using Python

```python
import requests

# Upload and predict
url = 'http://localhost:5000/predict'
files = {'file': open('image.jpg', 'rb')}
response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['prediction']['class']}")
print(f"Confidence: {result['prediction']['confidence']:.2f}%")
```

### Using JavaScript (Fetch API)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:5000/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Prediction:', data.prediction.class);
    console.log('Confidence:', data.prediction.confidence);
});
```

---

## ⚙️ Configuration

Edit `config.py` to customize settings:

```python
class Config:
    # Upload settings
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    
    # Model settings
    MODEL_CHECKPOINT = 'checkpoints/best_model.pth'
    
    # Server settings
    HOST = '0.0.0.0'  # Listen on all interfaces
    PORT = 5000
    DEBUG = True  # Set to False in production
```

---

## 🐛 Troubleshooting

### Model Not Found Error

**Error:** `Model checkpoint not found at checkpoints/best_model.pth`

**Solution:** Train the model first:
```bash
python main.py --mode train --epochs 50
```

### Port Already in Use

**Error:** `Address already in use`

**Solution:** Change the port in `config.py` or kill the process:
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

### CORS Issues

If accessing from a different domain, CORS is already enabled. To restrict origins, edit `config.py`:

```python
CORS_ORIGINS = 'https://yourdomain.com'
```

### File Upload Issues

- **File too large**: Increase `MAX_CONTENT_LENGTH` in `config.py`
- **Invalid file type**: Check `ALLOWED_EXTENSIONS` in `config.py`
- **Upload fails**: Ensure `uploads/` directory has write permissions

---

## 🌐 Production Deployment

### Using Gunicorn (Linux/Mac)

```bash
# Install gunicorn
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Waitress (Windows)

```bash
# Install waitress
pip install waitress

# Run server
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### Production Checklist

- [ ] Set `DEBUG = False` in `config.py`
- [ ] Use environment variables for `SECRET_KEY`
- [ ] Configure proper CORS origins
- [ ] Use HTTPS in production
- [ ] Set up proper logging
- [ ] Configure firewall rules
- [ ] Use a reverse proxy (nginx/Apache)
- [ ] Set up monitoring

---

## 📁 Project Structure

```
CNN/
├── app.py                  # Flask application
├── inference.py            # Model inference utilities
├── config.py              # Configuration
├── templates/
│   └── index.html         # Web interface
├── static/
│   ├── css/
│   │   └── style.css      # Styles
│   └── js/
│       └── main.js        # Client-side logic
├── checkpoints/
│   └── best_model.pth     # Trained model
└── uploads/               # Temporary upload folder
```

---

## 🔒 Security Notes

1. **File Validation**: Only allowed image types are accepted
2. **File Size Limit**: Maximum 16MB per upload
3. **Secure Filenames**: Uses `secure_filename()` to prevent path traversal
4. **Temporary Storage**: Uploaded files are deleted after prediction
5. **Error Handling**: Sensitive information is not exposed in errors

---

## 📊 Performance Tips

1. **Model Loading**: Model is loaded once and cached in memory
2. **Batch Predictions**: Use `inference.predict_batch()` for multiple images
3. **GPU Acceleration**: Model automatically uses GPU if available
4. **Concurrent Requests**: Use production WSGI server (Gunicorn/Waitress)

---

## 🎨 Customization

### Change Color Scheme

Edit `static/css/style.css`:

```css
:root {
    --primary: #6366f1;      /* Primary color */
    --secondary: #8b5cf6;    /* Secondary color */
    --bg-dark: #0f172a;      /* Background */
}
```

### Add More Classes

If you train on a different dataset, update `inference.py`:

```python
CIFAR10_CLASSES = ['class1', 'class2', ...]
```

### Custom Model

To use a different model architecture, update `inference.py`:

```python
from your_model import YourModel

def load_model(self):
    self.model = YourModel()
    # ... rest of loading code
```

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review [TROUBLESHOOTING.md](file:///c:/CNN/CNN/TROUBLESHOOTING.md)
3. Check Flask logs for error details
4. Verify model checkpoint exists and is valid

---

## ✅ Quick Test

After starting the server, test with:

```bash
# 1. Check health
curl http://localhost:5000/health

# 2. Get classes
curl http://localhost:5000/classes

# 3. Make a prediction (replace with your image)
curl -X POST -F "file=@test_image.jpg" http://localhost:5000/predict
```

---

**🎉 Your CIFAR-10 CNN is now deployed as a web application!**
