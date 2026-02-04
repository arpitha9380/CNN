"""
Quick start script for CIFAR-10 CNN training
This script provides a simple way to start training with recommended settings
"""

import subprocess
import sys

def main():
    print("=" * 70)
    print("CIFAR-10 CNN Quick Start")
    print("=" * 70)
    print("\nThis script will train the CNN on CIFAR-10 with the following settings:")
    print("  - Epochs: 50")
    print("  - Batch size: 128")
    print("  - Learning rate: 0.001")
    print("  - Data augmentation: Enabled")
    print("  - Validation split: 10%")
    print("  - Save plots: Yes")
    print("\n" + "=" * 70)
    
    response = input("\nDo you want to start training? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        print("\nStarting training...\n")
        
        # Run the main training script
        cmd = [
            sys.executable, 
            "main.py",
            "--mode", "both",
            "--epochs", "50",
            "--batch-size", "128",
            "--lr", "0.001",
            "--save-plots"
        ]
        
        subprocess.run(cmd)
    else:
        print("\nTraining cancelled.")
        print("\nTo train manually, run:")
        print("  python main.py --mode both --epochs 50 --save-plots")
        print("\nFor more options, run:")
        print("  python main.py --help")

if __name__ == "__main__":
    main()
