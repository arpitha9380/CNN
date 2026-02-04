"""
PyTorch Installation Fix for Windows
This script helps you fix the PyTorch DLL error on Windows
"""

import sys
import subprocess

def main():
    print("=" * 70)
    print("PyTorch DLL Error Fix for Windows")
    print("=" * 70)
    print("\nThe PyTorch DLL error occurs because of missing Visual C++ libraries.")
    print("\nTo fix this issue, you need to:")
    print("\n1. Download Microsoft Visual C++ Redistributable:")
    print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("\n2. Run the installer")
    print("\n3. Restart your computer")
    print("\n4. Try running your CNN project again")
    print("\n" + "=" * 70)
    print("\nAlternatively, you can:")
    print("- Use Google Colab (free GPU): https://colab.research.google.com/")
    print("- Use a Linux system or WSL (Windows Subsystem for Linux)")
    print("=" * 70)
    
    print("\n\nWould you like to:")
    print("1. Open the download link in your browser")
    print("2. Get instructions for Google Colab")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1/2/3): ")
    
    if choice == "1":
        import webbrowser
        webbrowser.open("https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("\n✅ Opening download link in your browser...")
        print("After installing, restart your computer and run: python main.py")
    elif choice == "2":
        print("\n" + "=" * 70)
        print("Google Colab Instructions:")
        print("=" * 70)
        print("\n1. Go to: https://colab.research.google.com/")
        print("2. Create a new notebook")
        print("3. Upload your CNN project files")
        print("4. Run: !pip install -r requirements.txt")
        print("5. Run: !python main.py --epochs 50 --save-plots")
        print("\nGoogle Colab provides free GPU access!")
        print("=" * 70)
    else:
        print("\nExiting...")

if __name__ == "__main__":
    main()
