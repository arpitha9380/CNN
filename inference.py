"""
Inference utilities for CIFAR-10 CNN model
Handles model loading, image preprocessing, and predictions
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from model import create_model

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


class ModelInference:
    """
    Handles model loading and inference for CIFAR-10 classification
    """
    
    def __init__(self, checkpoint_path='checkpoints/best_model.pth', device=None):
        """
        Initialize the inference engine
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on (cuda/cpu)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = self._get_transform()
        
    def _get_transform(self):
        """
        Get image preprocessing transform
        
        Returns:
            Transform pipeline for input images
        """
        # CIFAR-10 normalization values
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to CIFAR-10 size
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        return transform
    
    def load_model(self):
        """
        Load the trained model from checkpoint
        
        Returns:
            Loaded model
        """
        if self.model is not None:
            return self.model
        
        # Create model
        self.model = create_model(num_classes=10, dropout_rate=0.5)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {self.checkpoint_path}")
        print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        
        return self.model
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image or file path
            
        Returns:
            Preprocessed tensor
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            # Convert numpy array or other formats to PIL
            image = Image.fromarray(image).convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict(self, image, top_k=5):
        """
        Make prediction on a single image
        
        Args:
            image: PIL Image, file path, or numpy array
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Convert to numpy
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Get all class probabilities
        all_probs = probabilities.cpu().numpy()[0]
        
        # Format results
        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            predictions.append({
                'class': CIFAR10_CLASSES[idx],
                'class_id': int(idx),
                'confidence': float(prob * 100)
            })
        
        # All class probabilities
        all_predictions = {
            CIFAR10_CLASSES[i]: float(all_probs[i] * 100)
            for i in range(len(CIFAR10_CLASSES))
        }
        
        result = {
            'top_prediction': predictions[0],
            'top_k_predictions': predictions,
            'all_predictions': all_predictions
        }
        
        return result
    
    def predict_batch(self, images, top_k=5):
        """
        Make predictions on a batch of images
        
        Args:
            images: List of PIL Images, file paths, or numpy arrays
            top_k: Number of top predictions to return per image
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result = self.predict(image, top_k)
            results.append(result)
        
        return results


def test_inference():
    """
    Test the inference pipeline
    """
    import os
    
    # Check if checkpoint exists
    checkpoint_path = 'checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python main.py --mode train --epochs 50")
        return
    
    # Initialize inference
    print("Initializing inference engine...")
    inference = ModelInference(checkpoint_path)
    
    # Load model
    print("\nLoading model...")
    inference.load_model()
    
    # Create a dummy image for testing
    print("\nTesting with dummy image...")
    dummy_image = Image.new('RGB', (32, 32), color='red')
    
    # Make prediction
    result = inference.predict(dummy_image, top_k=5)
    
    print("\n" + "="*60)
    print("Prediction Results")
    print("="*60)
    print(f"\nTop Prediction: {result['top_prediction']['class']}")
    print(f"Confidence: {result['top_prediction']['confidence']:.2f}%")
    
    print("\nTop 5 Predictions:")
    for i, pred in enumerate(result['top_k_predictions'], 1):
        print(f"  {i}. {pred['class']}: {pred['confidence']:.2f}%")
    
    print("\n" + "="*60)
    print("Inference test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    test_inference()
