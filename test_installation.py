#!/usr/bin/env python3
"""
Test script to verify Image to Calendar Invite installation
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name} import failed: {e}")
        return False

def test_tesseract():
    """Test Tesseract OCR"""
    try:
        import pytesseract
        # Try to get version
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract OCR version: {version}")
        return True
    except Exception as e:
        print(f"✗ Tesseract OCR test failed: {e}")
        return False

def test_opencv():
    """Test OpenCV"""
    try:
        import cv2
        version = cv2.__version__
        print(f"✓ OpenCV version: {version}")
        return True
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False

def test_calendar():
    """Test calendar functionality"""
    try:
        from icalendar import Calendar, Event
        from datetime import datetime
        
        # Create a test calendar
        cal = Calendar()
        event = Event()
        event.add('summary', 'Test Event')
        event.add('dtstart', datetime.now())
        cal.add_component(event)
        
        print("✓ Calendar functionality working")
        return True
    except Exception as e:
        print(f"✗ Calendar test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Image to Calendar Invite - Installation Test")
    print("=" * 50)
    
    tests = [
        ("tkinter", "Tkinter"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("pytesseract", "Pytesseract"),
        ("dateutil.parser", "python-dateutil"),
        ("icalendar", "iCalendar"),
        ("customtkinter", "CustomTkinter"),
    ]
    
    all_passed = True
    
    # Test imports
    for module, name in tests:
        if not test_import(module, name):
            all_passed = False
    
    print("\n" + "-" * 50)
    
    # Test specific functionality
    if test_tesseract():
        print("✓ Tesseract OCR is working")
    else:
        all_passed = False
        print("⚠ Tesseract OCR may not be properly installed")
    
    if test_opencv():
        print("✓ OpenCV is working")
    else:
        all_passed = False
    
    if test_calendar():
        print("✓ Calendar functionality is working")
    else:
        all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 All tests passed! Your installation is ready.")
        print("\nYou can now run the application with:")
        print("python main.py")
    else:
        print("⚠ Some tests failed. Please check the installation.")
        print("\nTry running the setup script:")
        print("python setup.py")

if __name__ == "__main__":
    main() 