"""
Evaluation and testing utilities for CIFAR-10 CNN classifier
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluate model on test set and compute detailed metrics
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
        class_names: List of class names
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    correct = 0
    total = 0
    
    print("\nEvaluating model on test set...")
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Testing'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get predictions
            _, predicted = outputs.max(1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Calculate accuracy
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Calculate overall accuracy
    accuracy = 100. * correct / total
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Compute per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    
    # Generate classification report
    report = classification_report(
        all_targets, 
        all_predictions, 
        target_names=class_names,
        digits=4
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("Test Set Evaluation Results")
    print("=" * 70)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:12s}: {per_class_acc[i]:.2f}%")
    print("\n" + "=" * 70)
    print("Classification Report:")
    print("=" * 70)
    print(report)
    print("=" * 70 + "\n")
    
    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'classification_report': report
    }


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix as a heatmap
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy curves)
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot (optional)
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Plot learning rate
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def visualize_predictions(model, test_loader, device, class_names, 
                         num_images=16, save_path=None):
    """
    Visualize model predictions on sample images
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run inference on
        class_names: List of class names
        num_images: Number of images to visualize
        save_path: Path to save the plot (optional)
    """
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(test_loader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = outputs.max(1)
    
    # Move to CPU for visualization
    images = images.cpu()
    predictions = predictions.cpu()
    probabilities = probabilities.cpu()
    
    # Denormalize images for visualization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    # Create plot
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for idx in range(num_images):
        ax = axes[idx]
        
        # Display image
        img = images[idx].permute(1, 2, 0).numpy()
        ax.imshow(img)
        
        # Get prediction info
        pred_class = class_names[predictions[idx]]
        true_class = class_names[labels[idx]]
        confidence = probabilities[idx][predictions[idx]] * 100
        
        # Set title with color based on correctness
        is_correct = predictions[idx] == labels[idx]
        color = 'green' if is_correct else 'red'
        
        title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.1f}%"
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
    
    # Hide extra subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Model Predictions on Test Images', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to {save_path}")
    
    plt.show()
