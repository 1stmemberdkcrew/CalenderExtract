# Image to Calendar Invite

A Python application that extracts dates from images using OCR (Optical Character Recognition) and creates calendar invites (.ics files).

## Features

- **Image Processing**: Upload and process images to extract text
- **Date Detection**: Automatically detect dates from various formats in the extracted text
- **Calendar Creation**: Generate .ics calendar files that can be imported into any calendar application
- **Modern UI**: Clean, dark-themed interface built with CustomTkinter
- **Multiple Date Formats**: Supports various date formats including:
  - DD/MM/YYYY or MM/DD/YYYY
  - YYYY-MM-DD
  - DD Month YYYY
  - Month DD, YYYY
  - Time + Date combinations

## Requirements

- Python 3.7 or higher
- Tesseract OCR engine

## Installation

### 1. Install Tesseract OCR

**Windows:**
1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install it and add it to your system PATH
3. Default installation path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

1. **Run the application:**
   ```bash
   python main.py
   ```

2. **Select an image:**
   - Click "Select Image" to choose an image file
   - Supported formats: JPG, JPEG, PNG, BMP, TIFF

3. **Extract date:**
   - Click "Extract Date" to process the image
   - The application will display the extracted text and detected date

4. **Create calendar invite:**
   - Fill in the event details (title, description, duration)
   - Click "Create Calendar Invite" to generate a .ics file
   - The file will be saved in the current directory

## How it Works

1. **Image Preprocessing**: The image is converted to grayscale and thresholded to improve OCR accuracy
2. **Text Extraction**: Tesseract OCR extracts text from the processed image
3. **Date Parsing**: Regular expressions and dateutil parser identify dates in various formats
4. **Calendar Generation**: Creates a standard .ics file with the detected date and user-provided details

## File Structure

```
Picturecalender/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── signal-2025-06-21-201834_002.jpeg  # Example image
```

## Troubleshooting

### Common Issues

1. **Tesseract not found:**
   - Ensure Tesseract is installed and in your system PATH
   - On Windows, you may need to restart your terminal after installation

2. **No date detected:**
   - Ensure the image contains clear, readable text
   - Try different image formats or improve image quality
   - Check that the date format is supported

3. **Import errors:**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

### Improving OCR Accuracy

- Use high-resolution images
- Ensure good contrast between text and background
- Avoid blurry or distorted images
- Use images with clear, standard fonts

## Supported Date Formats

The application can detect dates in these formats:
- `21/06/2025`
- `06/21/2025`
- `2025-06-21`
- `21 June 2025`
- `June 21, 2025`
- `2:30 PM 21/06/2025`

## License

This project is open source and available under the MIT License. 