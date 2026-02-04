# CIFAR-10 CNN Classifier

An end-to-end Convolutional Neural Network (CNN) implementation for classifying images from the CIFAR-10 dataset using PyTorch.

## 📋 Overview

This project implements a deep CNN architecture to classify images from the CIFAR-10 dataset into 10 different categories:
- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐕 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck

## 🏗️ Architecture

The CNN architecture consists of:
- **3 Convolutional Blocks**: Each with 2 convolutional layers, batch normalization, ReLU activation, and max pooling
- **3 Fully Connected Layers**: With batch normalization and dropout for regularization
- **Total Parameters**: ~2.8 million trainable parameters

### Network Details
```
Conv Block 1: 3 → 64 → 64 channels
Conv Block 2: 64 → 128 → 128 channels
Conv Block 3: 128 → 256 → 256 channels
FC Layers: 4096 → 512 → 256 → 10
```

## 📁 Project Structure

```
CNN/
├── main.py              # Main training and evaluation script
├── model.py             # CNN model architecture
├── data_loader.py       # Data loading and preprocessing
├── train.py             # Training loop and utilities
├── evaluate.py          # Evaluation and visualization
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── data/               # CIFAR-10 dataset (auto-downloaded)
├── checkpoints/        # Saved model checkpoints
└── plots/              # Training plots and visualizations
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CNN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training the Model

Train the model with default parameters:
```bash
python main.py --mode train
```

Train with custom parameters:
```bash
python main.py --mode train --epochs 100 --batch-size 256 --lr 0.001
```

### Evaluating the Model

Evaluate a trained model:
```bash
python main.py --mode evaluate --load-checkpoint checkpoints/best_model.pth
```

### Training and Evaluation

Run both training and evaluation:
```bash
python main.py --mode both --epochs 50 --save-plots
```

## ⚙️ Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | 50 | Number of training epochs |
| `--batch-size` | int | 128 | Batch size for training |
| `--lr` | float | 0.001 | Initial learning rate |
| `--dropout` | float | 0.5 | Dropout rate |
| `--weight-decay` | float | 1e-4 | Weight decay (L2 regularization) |
| `--data-dir` | str | ./data | Directory for CIFAR-10 data |
| `--val-split` | float | 0.1 | Validation split ratio |
| `--num-workers` | int | 2 | Number of data loading workers |
| `--no-augment` | flag | False | Disable data augmentation |
| `--checkpoint-dir` | str | checkpoints | Directory to save checkpoints |
| `--load-checkpoint` | str | None | Path to checkpoint to load |
| `--mode` | str | both | Execution mode (train/evaluate/both) |
| `--no-cuda` | flag | False | Disable CUDA training |
| `--seed` | int | 42 | Random seed |
| `--save-plots` | flag | False | Save plots to files |
| `--plot-dir` | str | plots | Directory to save plots |

## 📊 Features

### Data Augmentation
- Random cropping with padding
- Random horizontal flipping
- Random rotation (±15°)
- Color jittering (brightness, contrast, saturation)

### Training Features
- Automatic train/validation split
- Model checkpointing (best model + periodic saves)
- Learning rate scheduling (Cosine Annealing)
- Progress bars with real-time metrics
- GPU acceleration support

### Evaluation Metrics
- Overall accuracy
- Per-class accuracy
- Confusion matrix
- Precision, recall, and F1-score
- Sample predictions visualization

## 📈 Expected Performance

With default hyperparameters, the model typically achieves:
- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~85%
- **Test Accuracy**: ~83-85%

Performance can be improved by:
- Training for more epochs
- Using learning rate warmup
- Implementing additional regularization techniques
- Using ensemble methods

## 🔍 Example Usage

### Quick Start
```bash
# Train for 50 epochs and save all plots
python main.py --epochs 50 --save-plots

# Train with larger batch size and custom learning rate
python main.py --batch-size 256 --lr 0.0005 --epochs 100

# Evaluate a saved model
python main.py --mode evaluate --load-checkpoint checkpoints/best_model.pth --save-plots
```

### Testing Individual Components

Test the model architecture:
```bash
python model.py
```

Test data loading:
```bash
python data_loader.py
```

## 📝 Output Files

After training, you'll find:

**Checkpoints** (in `checkpoints/`):
- `best_model.pth` - Model with best validation accuracy
- `checkpoint_epoch_X.pth` - Periodic checkpoints every 10 epochs

**Plots** (in `plots/` if `--save-plots` is used):
- `training_history.png` - Loss and accuracy curves
- `confusion_matrix.png` - Normalized confusion matrix
- `predictions.png` - Sample predictions visualization

## 🛠️ Customization

### Modifying the Model

Edit `model.py` to change the architecture:
- Adjust number of convolutional layers
- Change filter sizes and numbers
- Modify fully connected layer dimensions
- Add residual connections

### Custom Data Augmentation

Edit `data_loader.py` to modify augmentation:
```python
train_transform = transforms.Compose([
    # Add your custom transformations here
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # ...
])
```

## 🐛 Troubleshooting

**Out of Memory Error**:
- Reduce batch size: `--batch-size 64`
- Use CPU: `--no-cuda`

**Slow Training**:
- Increase number of workers: `--num-workers 4`
- Use GPU if available
- Reduce validation split: `--val-split 0.05`

**Poor Performance**:
- Train for more epochs: `--epochs 100`
- Adjust learning rate: `--lr 0.0005`
- Enable data augmentation (default)

## 📚 Requirements

See `requirements.txt` for full list of dependencies:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- scikit-learn (for metrics)
- seaborn (for visualizations)
- tqdm (for progress bars)

## 📄 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Happy Training! 🎉**
