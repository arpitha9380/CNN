# CIFAR-10 CNN Troubleshooting Guide

This document provides solutions to common issues you might encounter while setting up and running the CIFAR-10 CNN classifier.

## Installation Issues

### PyTorch DLL Error on Windows

**Error Message:**
```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed. 
Error loading "...\torch\lib\c10.dll" or one of its dependencies.
```

**Solutions:**

1. **Install Visual C++ Redistributables** (Most Common Fix)
   - Download and install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
   - Restart your computer after installation
   - Try running the code again

2. **Use CPU-only PyTorch** (If you don't have NVIDIA GPU)
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Update Windows**
   - Make sure Windows is fully updated
   - Some DLLs require recent Windows updates

4. **Check Python Version**
   - PyTorch 2.0+ requires Python 3.8 or higher
   - Verify: `python --version`

### CUDA Not Available

**Issue:** Training is slow or CUDA is not detected

**Check CUDA availability:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

**Solutions:**

1. **Install CUDA-enabled PyTorch:**
   ```bash
   pip uninstall torch torchvision
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Verify NVIDIA GPU:**
   - Open Device Manager → Display adapters
   - Check if NVIDIA GPU is present and drivers are updated

3. **Use CPU if no GPU:**
   - Add `--no-cuda` flag when running:
   ```bash
   python main.py --no-cuda
   ```

## Runtime Issues

### Out of Memory Error

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
   ```bash
   python main.py --batch-size 64
   # or even smaller
   python main.py --batch-size 32
   ```

2. **Use CPU instead:**
   ```bash
   python main.py --no-cuda
   ```

3. **Close other applications** using GPU memory

4. **Enable gradient checkpointing** (for advanced users - requires code modification)

### Slow Data Loading

**Issue:** Training is slow due to data loading bottleneck

**Solutions:**

1. **Reduce number of workers:**
   ```bash
   python main.py --num-workers 0
   ```

2. **Increase number of workers** (if you have multiple CPU cores):
   ```bash
   python main.py --num-workers 4
   ```

3. **Use SSD** for data storage instead of HDD

### Download Issues (CIFAR-10 Dataset)

**Issue:** Dataset download fails or is very slow

**Solutions:**

1. **Manual download:**
   - Download from: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
   - Extract to `./data/cifar-10-batches-py/`

2. **Use different network:**
   - Try using a VPN if download is blocked
   - Use mobile hotspot if university/corporate network blocks downloads

3. **Check disk space:**
   - CIFAR-10 requires ~170 MB
   - Ensure you have at least 500 MB free space

## Training Issues

### Poor Model Performance

**Issue:** Model accuracy is lower than expected

**Solutions:**

1. **Train for more epochs:**
   ```bash
   python main.py --epochs 100
   ```

2. **Adjust learning rate:**
   ```bash
   # Try lower learning rate
   python main.py --lr 0.0005
   # Or higher
   python main.py --lr 0.002
   ```

3. **Enable data augmentation** (enabled by default):
   ```bash
   # Make sure you're NOT using --no-augment flag
   python main.py
   ```

4. **Reduce dropout:**
   ```bash
   python main.py --dropout 0.3
   ```

### Training Crashes or Freezes

**Solutions:**

1. **Check system resources:**
   - Monitor RAM usage (Task Manager)
   - Close unnecessary applications

2. **Reduce batch size:**
   ```bash
   python main.py --batch-size 64
   ```

3. **Use CPU mode:**
   ```bash
   python main.py --no-cuda
   ```

4. **Update GPU drivers:**
   - Visit NVIDIA website for latest drivers

### Model Not Saving

**Issue:** Checkpoints are not being saved

**Solutions:**

1. **Check disk space:**
   - Model checkpoints require ~50 MB each
   - Ensure sufficient free space

2. **Check permissions:**
   - Make sure you have write permissions in the project directory

3. **Manually create checkpoint directory:**
   ```bash
   mkdir checkpoints
   ```

## Visualization Issues

### Plots Not Displaying

**Issue:** Matplotlib plots don't show up

**Solutions:**

1. **Use `--save-plots` flag:**
   ```bash
   python main.py --save-plots
   ```
   - Plots will be saved to `plots/` directory

2. **Check backend:**
   ```python
   import matplotlib
   print(matplotlib.get_backend())
   ```

3. **Install tkinter** (for interactive plots):
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install python3-tk
   # On Windows, tkinter should be included with Python
   ```

### Image Display Issues

**Issue:** Images appear distorted or incorrect colors

**Solution:**
- This is usually due to normalization
- The code handles this automatically in `visualize_predictions()`
- If you're writing custom visualization, denormalize first

## Common Command Line Errors

### "python: command not found"

**Solutions:**

1. **Use `python3` instead:**
   ```bash
   python3 main.py
   ```

2. **Use full path:**
   ```bash
   C:\Users\student\AppData\Local\Programs\Python\Python313\python.exe main.py
   ```

3. **Add Python to PATH** (Windows):
   - Search "Environment Variables"
   - Add Python installation directory to PATH

### "No module named 'torch'"

**Solution:**

1. **Reinstall requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Check if you're in the right environment:**
   ```bash
   pip list | grep torch
   ```

3. **Use the correct pip:**
   ```bash
   python -m pip install -r requirements.txt
   ```

## Performance Optimization

### Speed Up Training

1. **Use GPU** (if available):
   - Don't use `--no-cuda` flag
   - Verify CUDA is working

2. **Increase batch size** (if you have enough memory):
   ```bash
   python main.py --batch-size 256
   ```

3. **Reduce validation frequency:**
   - Modify code to validate every N epochs instead of every epoch

4. **Use mixed precision training** (advanced):
   - Requires code modification to use `torch.cuda.amp`

### Reduce Memory Usage

1. **Smaller batch size:**
   ```bash
   python main.py --batch-size 32
   ```

2. **Reduce model size:**
   - Modify `model.py` to use fewer filters
   - Example: Change 64→32, 128→64, 256→128

3. **Use CPU:**
   ```bash
   python main.py --no-cuda
   ```

## Getting Help

If you encounter an issue not covered here:

1. **Check error messages carefully:**
   - Read the full error traceback
   - Google the specific error message

2. **Verify your setup:**
   ```bash
   python --version
   pip list
   ```

3. **Test individual components:**
   ```bash
   python model.py
   python data_loader.py
   ```

4. **Create a minimal example:**
   - Try running with minimal settings
   ```bash
   python main.py --epochs 1 --batch-size 32 --no-cuda
   ```

## Quick Fixes Checklist

- [ ] Installed Visual C++ Redistributables (Windows)
- [ ] Updated GPU drivers
- [ ] Verified Python version (3.8+)
- [ ] Installed all requirements: `pip install -r requirements.txt`
- [ ] Have at least 2 GB free disk space
- [ ] Have at least 4 GB free RAM
- [ ] Closed other GPU-intensive applications
- [ ] Tried reducing batch size
- [ ] Tried CPU mode (`--no-cuda`)

## Still Having Issues?

1. **Check system requirements:**
   - Python 3.8+
   - 4 GB RAM minimum (8 GB recommended)
   - 2 GB free disk space
   - Windows 10/11, Linux, or macOS

2. **Try a fresh installation:**
   ```bash
   # Create new virtual environment
   python -m venv venv
   # Activate it
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   # Install requirements
   pip install -r requirements.txt
   ```

3. **Run in safe mode:**
   ```bash
   python main.py --epochs 1 --batch-size 16 --num-workers 0 --no-cuda
   ```
