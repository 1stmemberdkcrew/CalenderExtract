import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import pytesseract
import re
from dateutil import parser
from datetime import timedelta
from icalendar import Calendar, Event
import io
import subprocess
import sys
import os
import cv2
import numpy as np
from skimage import measure, filters, morphology
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt

# Enhanced Tesseract diagnostics
def get_tesseract_diagnostics():
    """Get comprehensive Tesseract diagnostic information"""
    diagnostics = {}
    
    # Check if pytesseract can import
    try:
        diagnostics['pytesseract_import'] = "✅ Success"
    except ImportError as e:
        diagnostics['pytesseract_import'] = f"❌ Failed: {str(e)}"
        return diagnostics
    
    # Get Tesseract version
    try:
        version = pytesseract.get_tesseract_version()
        diagnostics['tesseract_version'] = f"✅ {version}"
    except Exception as e:
        diagnostics['tesseract_version'] = f"❌ Failed: {str(e)}"
    
    # Get Tesseract path
    try:
        tesseract_path = pytesseract.get_tesseract_path()
        diagnostics['tesseract_path'] = f"✅ {tesseract_path}"
    except Exception as e:
        diagnostics['tesseract_path'] = f"❌ Failed: {str(e)}"
    
    # Check if tesseract command is available in PATH
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            diagnostics['system_tesseract'] = f"✅ {version_line}"
        else:
            diagnostics['system_tesseract'] = f"❌ Command failed: {result.stderr}"
    except FileNotFoundError:
        diagnostics['system_tesseract'] = "❌ Tesseract not found in PATH"
    except Exception as e:
        diagnostics['system_tesseract'] = f"❌ Error: {str(e)}"
    
    # Get available languages
    try:
        languages = pytesseract.get_languages()
        diagnostics['available_languages'] = f"✅ {', '.join(languages)}"
    except Exception as e:
        diagnostics['available_languages'] = f"❌ Failed: {str(e)}"
    
    return diagnostics

# Get diagnostics
tesseract_diagnostics = get_tesseract_diagnostics()

# Check if Tesseract is available
try:
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

st.set_page_config(
    page_title="Image to Calendar Event Extractor",
    page_icon="📅",
    layout="wide"
)

st.title("📅 Image to Calendar Event Extractor")
st.markdown("Upload an event flyer image to extract event details and create a calendar invite.")

# Sidebar: Always show Tesseract diagnostics at the top
with st.sidebar:
    st.header("🔧 Tesseract OCR Diagnostics (Always Visible)")
    for key, value in tesseract_diagnostics.items():
        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    st.divider()
    st.header("Settings")
    if TESSERACT_AVAILABLE:
        confidence_threshold = st.slider("OCR Confidence Threshold", 0, 100, 50, help="Adjust OCR sensitivity")
        
        # Preprocessing options
        st.subheader("🖼️ Image Preprocessing")
        preprocessing_method = st.selectbox(
            "Preprocessing Method",
            options=[
                "none",
                "basic_threshold",
                "adaptive_threshold", 
                "otsu_threshold",
                "contrast_enhancement",
                "noise_reduction",
                "morphological_ops",
                "multi_scale"
            ],
            help="Choose preprocessing method for better OCR"
        )
        
        threshold_value = st.slider("Binarization Threshold", 0, 255, 180, help="Adjust threshold for image preprocessing")
        contrast_factor = st.slider("Contrast Enhancement", 0.5, 3.0, 1.5, help="Enhance image contrast")
        
        psm_mode = st.selectbox(
            "Tesseract Page Segmentation Mode (PSM)",
            options=[
                (3, "Default (Fully automatic page segmentation)"),
                (6, "Assume a single uniform block of text"),
                (11, "Sparse text. Find as much text as possible."),
                (4, "Assume a single column of text"),
                (7, "Treat the image as a single text line"),
                (8, "Single word"),
                (9, "Single word in a circle"),
                (10, "Single character")
            ],
            format_func=lambda x: f"{x[0]}: {x[1]}",
            index=1
        )
        
        use_original = st.checkbox("Use original image (no preprocessing) for OCR", value=False)
        show_raw_ocr = st.checkbox("Always show raw OCR output", value=True, help="Display detailed OCR results")
        
        # AI Analysis options
        st.subheader("🤖 AI Analysis")
        enable_quality_assessment = st.checkbox("Enable Image Quality Assessment", value=True)
        enable_layout_analysis = st.checkbox("Enable Layout Analysis", value=True)
        enable_font_analysis = st.checkbox("Enable Font Analysis", value=True)
    else:
        st.error("⚠️ Tesseract OCR not available")
        st.info("OCR functionality is currently unavailable. Please contact support.")
    
    st.header("About")
    st.markdown("""
    This app extracts event information from images using OCR technology.
    
    **Features:**
    - Automatic date/time detection
    - Event title extraction
    - Calendar invite generation
    - AI-powered image analysis
    """)

uploaded_file = st.file_uploader("Upload an event flyer image", type=["jpg", "jpeg", "png", "bmp", "tiff"])

def preprocess_image_advanced(image, method, threshold=180, contrast_factor=1.5):
    """Advanced preprocessing with multiple methods"""
    if method == "none":
        return image
    
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    if method == "basic_threshold":
        _, processed = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    elif method == "adaptive_threshold":
        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    elif method == "otsu_threshold":
        _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    elif method == "contrast_enhancement":
        # Enhance contrast using PIL
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(contrast_factor)
        # Convert to grayscale and threshold
        gray_pil = ImageOps.grayscale(enhanced)
        processed = np.array(gray_pil)
        _, processed = cv2.threshold(processed, threshold, 255, cv2.THRESH_BINARY)
    
    elif method == "noise_reduction":
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, processed = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    
    elif method == "morphological_ops":
        # Apply morphological operations
        kernel = np.ones((2, 2), np.uint8)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    
    elif method == "multi_scale":
        # Try multiple preprocessing methods and return the best one
        methods = ["basic_threshold", "adaptive_threshold", "otsu_threshold"]
        best_result = None
        best_score = 0
        
        for m in methods:
            try:
                result = preprocess_image_advanced(image, m, threshold, contrast_factor)
                # Simple heuristic: count non-zero pixels (more text = better)
                score = np.count_nonzero(result)
                if score > best_score:
                    best_score = score
                    best_result = result
            except:
                continue
        
        if best_result is not None:
            processed = best_result
        else:
            processed = gray
    
    else:
        processed = gray
    
    # Convert back to PIL Image
    return Image.fromarray(processed)

def preprocess_image(image, threshold):
    gray = ImageOps.grayscale(image)
    bw = gray.point(lambda x: 255 if x > threshold else 0, mode='1')
    return bw

def extract_title_from_text(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    # Improved Heuristic: Group consecutive short, title-like lines at the top
    title_lines = []
    for line in lines[:6]:
        if len(line.split()) <= 3 and (line.isupper() or line.istitle()):
            title_lines.append(line)
        else:
            break
    if len(title_lines) >= 2:
        return ' '.join(title_lines)
    for line in lines[:3]:
        if line.isupper() and len(line.split()) > 1:
            return line
    if lines and len(lines[0].split()) <= 8:
        return lines[0]
    date_pattern = r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+\w+\s+\d{1,2},?\s+\d{4}'
    for i, line in enumerate(lines):
        if re.search(date_pattern, line):
            if i > 0:
                return lines[i-1]
    return "Event"

def extract_event_datetime_range(text):
    date_match = re.search(r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+\w+\s+\d{1,2},?\s+\d{4}', text)
    date_str = date_match.group(0) if date_match else None
    time_matches = re.findall(r'(\d{1,2}(?::\d{2})?\s*[ap]m)', text, re.IGNORECASE)
    if date_str and time_matches:
        try:
            times = [parser.parse(f"{date_str} {t}", fuzzy=True) for t in time_matches]
            times.sort()
            return times[0], times[-1]
        except Exception:
            pass
    if date_str:
        try:
            base_date = parser.parse(date_str, fuzzy=True)
            return base_date, base_date + timedelta(hours=1)
        except Exception:
            return None, None
    return None, None

def create_calendar_invite(title, description, start_time, end_time):
    """Create an iCalendar file"""
    cal = Calendar()
    event = Event()
    event.add('summary', title)
    event.add('description', description)
    event.add('dtstart', start_time)
    event.add('dtend', end_time)
    event.add('dtstamp', start_time)
    cal.add_component(event)
    return cal.to_ical()

def assess_image_quality(image):
    """Assess image quality for OCR"""
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Calculate various quality metrics
    quality_metrics = {}
    
    # 1. Resolution assessment
    height, width = gray.shape
    quality_metrics['resolution'] = {
        'width': width,
        'height': height,
        'pixels': width * height,
        'score': min(100, (width * height) / 10000)  # Score based on pixel count
    }
    
    # 2. Contrast assessment
    contrast = np.std(gray)
    quality_metrics['contrast'] = {
        'value': contrast,
        'score': min(100, contrast / 2)  # Higher contrast = better
    }
    
    # 3. Sharpness assessment (using Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    quality_metrics['sharpness'] = {
        'value': laplacian_var,
        'score': min(100, laplacian_var / 10)  # Higher variance = sharper
    }
    
    # 4. Noise assessment
    # Apply Gaussian blur and compare with original
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))
    quality_metrics['noise'] = {
        'value': noise,
        'score': max(0, 100 - noise * 2)  # Lower noise = better
    }
    
    # 5. Overall quality score
    overall_score = (
        quality_metrics['resolution']['score'] * 0.2 +
        quality_metrics['contrast']['score'] * 0.3 +
        quality_metrics['sharpness']['score'] * 0.3 +
        quality_metrics['noise']['score'] * 0.2
    )
    
    quality_metrics['overall_score'] = min(100, max(0, overall_score))
    
    # 6. Quality recommendations
    recommendations = []
    if quality_metrics['resolution']['score'] < 50:
        recommendations.append("📏 Low resolution - consider using a higher resolution image")
    if quality_metrics['contrast']['score'] < 50:
        recommendations.append("🌓 Low contrast - try enhancing contrast or using different lighting")
    if quality_metrics['sharpness']['score'] < 50:
        recommendations.append("📸 Image is blurry - ensure camera is steady and focused")
    if quality_metrics['noise']['score'] < 50:
        recommendations.append("🔇 High noise - try reducing camera ISO or improving lighting")
    
    quality_metrics['recommendations'] = recommendations
    
    return quality_metrics

def analyze_layout(image):
    """Analyze image layout and text regions"""
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    layout_analysis = {}
    
    # 1. Text region detection using MSER
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    
    # Filter regions by size
    min_area = 100
    max_area = gray.shape[0] * gray.shape[1] // 4
    valid_regions = [region for region in regions if min_area < len(region) < max_area]
    
    layout_analysis['text_regions'] = {
        'count': len(valid_regions),
        'total_area': sum(len(region) for region in valid_regions),
        'coverage_percent': (sum(len(region) for region in valid_regions) / (gray.shape[0] * gray.shape[1])) * 100
    }
    
    # 2. Line detection using Hough transform
    edges = canny(gray, sigma=2)
    lines = hough_line(edges)
    peaks = hough_line_peaks(*lines)
    
    # Analyze line orientations
    angles = []
    for _, angle, dist in zip(*peaks):
        angles.append(np.degrees(angle))
    
    layout_analysis['line_analysis'] = {
        'horizontal_lines': len([a for a in angles if abs(a) < 10 or abs(a - 180) < 10]),
        'vertical_lines': len([a for a in angles if abs(a - 90) < 10 or abs(a + 90) < 10]),
        'total_lines': len(angles)
    }
    
    # 3. Layout type classification
    if layout_analysis['text_regions']['coverage_percent'] > 30:
        layout_type = "Text-heavy"
    elif layout_analysis['line_analysis']['horizontal_lines'] > layout_analysis['line_analysis']['vertical_lines']:
        layout_type = "Horizontal layout"
    elif layout_analysis['line_analysis']['vertical_lines'] > layout_analysis['line_analysis']['horizontal_lines']:
        layout_type = "Vertical layout"
    else:
        layout_type = "Mixed layout"
    
    layout_analysis['layout_type'] = layout_type
    
    return layout_analysis

def analyze_font_characteristics(image, ocr_data):
    """Analyze font characteristics from OCR data"""
    if not ocr_data or 'conf' not in ocr_data:
        return {}
    
    font_analysis = {}
    
    # Extract text blocks with confidence > 0
    valid_blocks = []
    for i, conf in enumerate(ocr_data['conf']):
        if conf > 0 and ocr_data['text'][i].strip():
            valid_blocks.append({
                'text': ocr_data['text'][i],
                'conf': conf,
                'height': ocr_data['height'][i],
                'width': ocr_data['width'][i]
            })
    
    if not valid_blocks:
        return {'error': 'No valid text blocks found'}
    
    # 1. Font size analysis
    heights = [block['height'] for block in valid_blocks]
    font_analysis['font_sizes'] = {
        'min': min(heights),
        'max': max(heights),
        'avg': np.mean(heights),
        'std': np.std(heights)
    }
    
    # 2. Font size categories
    avg_height = font_analysis['font_sizes']['avg']
    if avg_height < 20:
        size_category = "Small"
    elif avg_height < 40:
        size_category = "Medium"
    elif avg_height < 60:
        size_category = "Large"
    else:
        size_category = "Very Large"
    
    font_analysis['size_category'] = size_category
    
    # 3. Text density analysis
    total_text_length = sum(len(block['text']) for block in valid_blocks)
    total_area = sum(block['height'] * block['width'] for block in valid_blocks)
    
    font_analysis['text_density'] = {
        'characters_per_pixel': total_text_length / total_area if total_area > 0 else 0,
        'total_characters': total_text_length,
        'total_area': total_area
    }
    
    # 4. Font style hints (based on text characteristics)
    style_hints = []
    
    # Check for all caps
    all_caps_count = sum(1 for block in valid_blocks if block['text'].isupper())
    if all_caps_count > len(valid_blocks) * 0.7:
        style_hints.append("All caps text detected")
    
    # Check for mixed case
    mixed_case_count = sum(1 for block in valid_blocks if not block['text'].isupper() and not block['text'].islower())
    if mixed_case_count > len(valid_blocks) * 0.5:
        style_hints.append("Mixed case text detected")
    
    # Check for numbers
    number_count = sum(1 for block in valid_blocks if any(c.isdigit() for c in block['text']))
    if number_count > len(valid_blocks) * 0.3:
        style_hints.append("Numerical content detected")
    
    font_analysis['style_hints'] = style_hints
    
    return font_analysis

def get_ocr_recommendations(quality_metrics, layout_analysis, font_analysis):
    """Generate OCR recommendations based on analysis"""
    recommendations = []
    
    # Quality-based recommendations
    if quality_metrics['overall_score'] < 50:
        recommendations.append("⚠️ Poor image quality detected. Consider:")
        recommendations.append("   • Using a higher resolution image")
        recommendations.append("   • Improving lighting conditions")
        recommendations.append("   • Ensuring camera is steady and focused")
    
    # Layout-based recommendations
    if layout_analysis['text_regions']['coverage_percent'] < 10:
        recommendations.append("📝 Low text coverage detected. Try:")
        recommendations.append("   • Ensuring text is clearly visible")
        recommendations.append("   • Avoiding heavy backgrounds")
    
    if layout_analysis['layout_type'] == "Vertical layout":
        recommendations.append("📐 Vertical layout detected. Consider using PSM mode 4 or 6")
    
    # Font-based recommendations
    if font_analysis.get('size_category') == "Small":
        recommendations.append("🔍 Small font detected. Try:")
        recommendations.append("   • Using higher resolution image")
        recommendations.append("   • Zooming in on text areas")
    
    if "All caps text detected" in font_analysis.get('style_hints', []):
        recommendations.append("🔤 All caps text detected. This may affect OCR accuracy")
    
    return recommendations

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # AI Analysis Section
        if enable_quality_assessment or enable_layout_analysis:
            st.subheader("🤖 AI Analysis")
            
            # Image Quality Assessment
            if enable_quality_assessment:
                with st.expander("📊 Image Quality Assessment", expanded=True):
                    quality_metrics = assess_image_quality(image)
                    
                    # Display quality score
                    col_q1, col_q2 = st.columns(2)
                    with col_q1:
                        st.metric("Overall Quality Score", f"{quality_metrics['overall_score']:.1f}/100")
                    with col_q2:
                        if quality_metrics['overall_score'] >= 80:
                            st.success("Excellent quality")
                        elif quality_metrics['overall_score'] >= 60:
                            st.info("Good quality")
                        elif quality_metrics['overall_score'] >= 40:
                            st.warning("Fair quality")
                        else:
                            st.error("Poor quality")
                    
                    # Detailed metrics
                    st.write("**Detailed Metrics:**")
                    st.write(f"• Resolution: {quality_metrics['resolution']['width']}x{quality_metrics['resolution']['height']} ({quality_metrics['resolution']['score']:.1f}/100)")
                    st.write(f"• Contrast: {quality_metrics['contrast']['value']:.1f} ({quality_metrics['contrast']['score']:.1f}/100)")
                    st.write(f"• Sharpness: {quality_metrics['sharpness']['value']:.1f} ({quality_metrics['sharpness']['score']:.1f}/100)")
                    st.write(f"• Noise: {quality_metrics['noise']['value']:.1f} ({quality_metrics['noise']['score']:.1f}/100)")
                    
                    # Recommendations
                    if quality_metrics['recommendations']:
                        st.write("**Recommendations:**")
                        for rec in quality_metrics['recommendations']:
                            st.write(rec)
            
            # Layout Analysis
            if enable_layout_analysis:
                with st.expander("📐 Layout Analysis", expanded=True):
                    layout_analysis = analyze_layout(image)
                    
                    st.write(f"**Layout Type:** {layout_analysis['layout_type']}")
                    st.write(f"**Text Regions:** {layout_analysis['text_regions']['count']} detected")
                    st.write(f"**Text Coverage:** {layout_analysis['text_regions']['coverage_percent']:.1f}%")
                    st.write(f"**Lines Detected:** {layout_analysis['line_analysis']['total_lines']} total")
                    st.write(f"  - Horizontal: {layout_analysis['line_analysis']['horizontal_lines']}")
                    st.write(f"  - Vertical: {layout_analysis['line_analysis']['vertical_lines']}")
        
        # Preprocess image for better OCR
        if preprocessing_method != "none":
            preprocessed_image = preprocess_image_advanced(image, preprocessing_method, threshold_value, contrast_factor)
            st.image(preprocessed_image, caption=f"Preprocessed: {preprocessing_method} (Threshold: {threshold_value})", use_column_width=True, channels="GRAY")
        else:
            preprocessed_image = image
            st.info("No preprocessing applied.")
        
        # Decide which image to use for OCR
        if use_original:
            ocr_image = image
            st.info("Using original image for OCR.")
        else:
            ocr_image = preprocessed_image
            st.info(f"Using {preprocessing_method} preprocessed image for OCR.")
        
        # Process image with OCR
        if TESSERACT_AVAILABLE:
            try:
                st.subheader("🔍 OCR Processing")
                custom_config = f'--psm {psm_mode[0]}'
                text = pytesseract.image_to_string(ocr_image, config=custom_config)
                
                try:
                    ocr_data = pytesseract.image_to_data(ocr_image, output_type=pytesseract.Output.DICT, config=custom_config)
                    confidence_scores = [score for score in ocr_data['conf'] if score > 0]
                    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                    st.metric("Average OCR Confidence", f"{avg_confidence:.1f}%")
                    st.metric("Words Detected", len([word for word in ocr_data['text'] if word.strip()]))
                    
                    # Font Analysis
                    if enable_font_analysis and ocr_data:
                        with st.expander("🔤 Font Analysis", expanded=True):
                            font_analysis = analyze_font_characteristics(image, ocr_data)
                            
                            if 'error' not in font_analysis:
                                st.write(f"**Font Size Category:** {font_analysis['size_category']}")
                                st.write(f"**Font Size Range:** {font_analysis['font_sizes']['min']:.1f} - {font_analysis['font_sizes']['max']:.1f} pixels")
                                st.write(f"**Average Font Size:** {font_analysis['font_sizes']['avg']:.1f} pixels")
                                st.write(f"**Text Density:** {font_analysis['text_density']['characters_per_pixel']:.6f} chars/pixel")
                                
                                if font_analysis['style_hints']:
                                    st.write("**Style Hints:**")
                                    for hint in font_analysis['style_hints']:
                                        st.write(f"• {hint}")
                            else:
                                st.warning("Font analysis not available")
                    
                    # Generate recommendations
                    if enable_quality_assessment and enable_layout_analysis:
                        quality_metrics = assess_image_quality(image)
                        layout_analysis = analyze_layout(image)
                        font_analysis = analyze_font_characteristics(image, ocr_data) if ocr_data else {}
                        
                        recommendations = get_ocr_recommendations(quality_metrics, layout_analysis, font_analysis)
                        
                        if recommendations:
                            with st.expander("💡 OCR Recommendations", expanded=True):
                                for rec in recommendations:
                                    st.write(rec)
                    
                    # Show preprocessing effectiveness
                    if len(text.strip()) > 0:
                        st.success(f"✅ OCR Success! Extracted {len(text.split())} words")
                    else:
                        st.warning("⚠️ No text extracted. Try different preprocessing or PSM mode.")
                        
                except Exception as e:
                    st.warning(f"Could not get detailed OCR data: {str(e)}")
                
                st.subheader("📝 Raw OCR Output")
                st.text_area("Raw OCR Text:", value=text, height=200, key="raw_ocr_output")
                
                if 'ocr_data' in locals():
                    with st.expander("🔍 Detailed OCR Data"):
                        st.write("**OCR Data Structure:**")
                        st.json({k: v[:10] if isinstance(v, list) and len(v) > 10 else v for k, v in ocr_data.items()})
                        st.write("**Words with Confidence Scores:**")
                        words_with_conf = []
                        for i, (text_word, conf) in enumerate(zip(ocr_data['text'], ocr_data['conf'])):
                            if text_word.strip():
                                words_with_conf.append(f"'{text_word}': {conf}%")
                        st.text('\n'.join(words_with_conf[:50]))
                        if len(words_with_conf) > 50:
                            st.write(f"... and {len(words_with_conf) - 50} more words")
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.info("Please try uploading a clearer image or contact support if the issue persists.")
                st.stop()
        else:
            st.error("❌ OCR not available")
            st.info("Tesseract OCR is not installed. Please contact support to enable OCR functionality.")
            st.stop()
    
    with col2:
        st.subheader("🎯 Extracted Information")
        
        # Extract information
        title = extract_title_from_text(text)
        start, end = extract_event_datetime_range(text)
        
        # Display results
        st.write("**Event Title:**")
        event_title = st.text_input("Edit title if needed:", value=title, key="title_input")
        
        st.write("**Event Description:**")
        event_description = st.text_area("Edit description if needed:", value=text[:200] + "..." if len(text) > 200 else text, key="desc_input")
        
        if start and end:
            st.write("**Event Time:**")
            st.success(f"📅 {start.strftime('%Y-%m-%d %I:%M %p')} to {end.strftime('%Y-%m-%d %I:%M %p')}")
            
            # Allow manual editing of times
            col_start, col_end = st.columns(2)
            with col_start:
                new_start = st.datetime_input("Start time:", value=start, key="start_input")
            with col_end:
                new_end = st.datetime_input("End time:", value=end, key="end_input")
            
            # Create calendar invite
            if st.button("📅 Create Calendar Invite", type="primary"):
                try:
                    cal_data = create_calendar_invite(
                        event_title, 
                        event_description, 
                        new_start, 
                        new_end
                    )
                    
                    # Create download button
                    st.download_button(
                        label="📥 Download .ics file",
                        data=cal_data,
                        file_name=f"{event_title.replace(' ', '_')}.ics",
                        mime="text/calendar"
                    )
                    
                    st.success("✅ Calendar invite created successfully!")
                    
                except Exception as e:
                    st.error(f"Error creating calendar invite: {str(e)}")
        else:
            st.warning("⚠️ Could not extract event time from the image.")
            st.info("Please check if the image contains clear date and time information.")
    
    # Additional debug information
    with st.expander("🔧 Debug Information"):
        st.write("**Image Information:**")
        st.write(f"- Format: {image.format}")
        st.write(f"- Mode: {image.mode}")
        st.write(f"- Size: {image.size}")
        
        st.write("**Processing Information:**")
        st.write(f"- Tesseract Available: {TESSERACT_AVAILABLE}")
        if TESSERACT_AVAILABLE:
            st.write(f"- Text Length: {len(text)} characters")
            st.write(f"- Lines: {len(text.split(chr(10)))}")
            st.write(f"- Words: {len(text.split())}") 