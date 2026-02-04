# VS Code Setup Guide

Complete guide for running the CIFAR-10 CNN project on a new laptop using Visual Studio Code.

## Quick Start (5 Steps)

### 1. Install Prerequisites
- Python 3.8+ (https://www.python.org/downloads/)
- Git (https://git-scm.com/downloads)
- VS Code (https://code.visualstudio.com/)

### 2. Clone Repository
```bash
git clone https://github.com/arpitha9380/CNN.git
cd CNN
code .
```

### 3. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Project

**Train Model:**
```bash
python main.py --mode train --epochs 50 --save-plots
```

**Run Web App:**
```bash
python app.py
```
Then open: http://localhost:5000

---

## VS Code Extensions (Required)

Install these in VS Code (Ctrl+Shift+X):
1. **Python** (by Microsoft)
2. **Pylance** (by Microsoft)
3. **Jupyter** (by Microsoft)

---

## Common Commands

### Training
```bash
# Full training (50 epochs, ~2-4 hours)
python main.py --mode train --epochs 50 --save-plots

# Quick test (1 epoch, ~5 minutes)
python main.py --mode train --epochs 1
```

### Web Application
```bash
# Start Flask server
python app.py

# Access at: http://localhost:5000
```

### Testing Components
```bash
python model.py        # Test model architecture
python data_loader.py  # Test data loading
python inference.py    # Test inference
```

---

## Troubleshooting

### Virtual Environment Not Activating (Windows)
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

### Python Command Not Found
```bash
# Use python3 instead
python3 main.py
```

### Out of Memory
```bash
# Reduce batch size
python main.py --mode train --batch-size 32
```

### Port Already in Use
Edit `config.py` and change:
```python
PORT = 8000  # Change to any available port
```

---

## VS Code Tips

- **Open Terminal:** Ctrl+` (backtick)
- **Run Python File:** Ctrl+F5
- **Debug:** F5
- **Search Files:** Ctrl+P
- **Search in Files:** Ctrl+Shift+F
- **Git Panel:** Ctrl+Shift+G

---

## Expected Results

**After Training:**
- Model saved to: `checkpoints/best_model.pth`
- Plots saved to: `plots/`
- Test accuracy: 83-85%

**Web Application:**
- Upload images via drag-and-drop
- Get instant predictions
- View confidence scores

---

For detailed instructions, see the complete setup guide.
