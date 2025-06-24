#!/usr/bin/env python3
"""
Setup script for Image to Calendar Invite application
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_requirements():
    """Install Python requirements"""
    try:
        print("Installing Python dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Python dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def check_tesseract():
    """Check if Tesseract is installed"""
    try:
        result = subprocess.run(["tesseract", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Tesseract OCR is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("⚠ Tesseract OCR not found")
    print("\nPlease install Tesseract OCR:")
    print("\nWindows:")
    print("1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    print("2. Install and add to PATH")
    print("3. Restart your terminal")
    
    print("\nmacOS:")
    print("brew install tesseract")
    
    print("\nLinux (Ubuntu/Debian):")
    print("sudo apt-get install tesseract-ocr")
    
    return False

def main():
    """Main setup function"""
    print("Image to Calendar Invite - Setup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Check Tesseract
    tesseract_installed = check_tesseract()
    
    print("\n" + "=" * 40)
    if tesseract_installed:
        print("✓ Setup completed successfully!")
        print("\nYou can now run the application with:")
        print("python main.py")
    else:
        print("⚠ Setup completed with warnings")
        print("Please install Tesseract OCR to use the application")

if __name__ == "__main__":
    main() 