"""
Main script for training and evaluating CIFAR-10 CNN classifier
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os

from model import create_model
from data_loader import get_data_loaders, get_dataset_info, CIFAR10_CLASSES
from train import train_model, load_checkpoint
from evaluate import (
    evaluate_model, 
    plot_confusion_matrix, 
    plot_training_history,
    visualize_predictions
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CIFAR-10 CNN Classifier')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization) (default: 1e-4)')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for CIFAR-10 data (default: ./data)')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loading workers (default: 2)')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    
    # Model parameters
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Path to checkpoint to load')
    
    # Execution mode
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'both'],
                        help='Execution mode (default: both)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # Visualization
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to files')
    parser.add_argument('--plot-dir', type=str, default='plots',
                        help='Directory to save plots (default: plots)')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print("\n" + "=" * 70)
    print("CIFAR-10 CNN Classifier")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70 + "\n")
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_split=args.val_split,
        augment=not args.no_augment
    )
    get_dataset_info(train_loader, val_loader, test_loader)
    
    # Create model
    print("\nCreating model...")
    model = create_model(num_classes=10, dropout_rate=args.dropout)
    model = model.to(device)
    print(f"Total trainable parameters: {model.get_num_params():,}")
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"\nLoading checkpoint from {args.load_checkpoint}")
        model = load_checkpoint(model, args.load_checkpoint, device)
    
    # Training mode
    if args.mode in ['train', 'both']:
        print("\n" + "=" * 70)
        print("Training Configuration")
        print("=" * 70)
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Weight decay: {args.weight_decay}")
        print(f"Dropout: {args.dropout}")
        print(f"Data augmentation: {not args.no_augment}")
        print("=" * 70)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=args.epochs,
            save_dir=args.checkpoint_dir
        )
        
        # Plot training history
        if args.save_plots:
            os.makedirs(args.plot_dir, exist_ok=True)
            save_path = os.path.join(args.plot_dir, 'training_history.png')
        else:
            save_path = None
        
        plot_training_history(history, save_path=save_path)
        
        # Load best model for evaluation
        if val_loader:
            best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                print(f"\nLoading best model for evaluation...")
                model = load_checkpoint(model, best_model_path, device)
    
    # Evaluation mode
    if args.mode in ['evaluate', 'both']:
        print("\n" + "=" * 70)
        print("Evaluation")
        print("=" * 70)
        
        # Evaluate on test set
        results = evaluate_model(model, test_loader, device, CIFAR10_CLASSES)
        
        # Plot confusion matrix
        if args.save_plots:
            os.makedirs(args.plot_dir, exist_ok=True)
            cm_save_path = os.path.join(args.plot_dir, 'confusion_matrix.png')
        else:
            cm_save_path = None
        
        plot_confusion_matrix(
            results['confusion_matrix'],
            CIFAR10_CLASSES,
            save_path=cm_save_path
        )
        
        # Visualize predictions
        if args.save_plots:
            pred_save_path = os.path.join(args.plot_dir, 'predictions.png')
        else:
            pred_save_path = None
        
        visualize_predictions(
            model,
            test_loader,
            device,
            CIFAR10_CLASSES,
            num_images=16,
            save_path=pred_save_path
        )
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
