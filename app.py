import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
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

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
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