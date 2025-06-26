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
from pyzbar.pyzbar import decode as decode_qr

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
        
        # Advanced OCR settings
        st.subheader("🔧 Advanced OCR Settings")
        
        # Language selection
        try:
            available_languages = pytesseract.get_languages()
            selected_language = st.selectbox(
                "OCR Language",
                options=['eng'] + [lang for lang in available_languages if lang != 'eng'],
                index=0,
                help="Select language for better OCR accuracy"
            )
        except:
            selected_language = 'eng'
        
        # OCR Engine Mode
        oem_mode = st.selectbox(
            "OCR Engine Mode (OEM)",
            options=[
                (0, "Legacy engine only"),
                (1, "Neural nets LSTM engine only"),
                (2, "Legacy + LSTM engines"),
                (3, "Default, based on what is available")
            ],
            format_func=lambda x: f"{x[0]}: {x[1]}",
            index=3
        )
        
        # Additional OCR parameters
        st.write("**OCR Parameters:**")
        tessedit_char_whitelist = st.text_input(
            "Character Whitelist",
            value="",
            help="Only recognize these characters (leave empty for all)"
        )
        
        tessedit_char_blacklist = st.text_input(
            "Character Blacklist", 
            value="",
            help="Exclude these characters from recognition"
        )
        
        # Confidence filtering
        min_confidence = st.slider(
            "Minimum Confidence (%)",
            min_value=0,
            max_value=100,
            value=0,
            help="Filter out low-confidence text"
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
    - Enhanced image preprocessing
    """)

# Add a user note above the uploader
st.markdown("""
**Tip:** You can drag and drop an image file here from your computer. 
If you want to use an image from a web page, right-click and save it first, then drag it in. 

*Supported formats: JPG, JPEG, PNG, BMP, TIFF.*
""")

# Make the uploader area visually larger
uploaded_file = st.file_uploader(
    "Upload an event flyer image (drag & drop supported)",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    label_visibility="visible",
    key="main_image_uploader",
    help="Drag and drop or click to select an image file."
)

# Add extra vertical space below the uploader for visual emphasis
st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

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
    """Enhanced date/time extraction with multiple patterns"""
    # More comprehensive date patterns
    date_patterns = [
        # Full date with day name
        r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+\w+\s+\d{1,2},?\s+\d{4}',
        # Date without day name
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        # Numeric date formats
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
        # Relative dates
        r'\b(today|tomorrow|next\s+\w+)\b',
        # Date ranges
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\s*[-–—]\s*\d{1,2}[/-]\d{1,2}[/-]\d{4}\b'
    ]
    
    # Enhanced time patterns
    time_patterns = [
        # Standard time formats
        r'\b\d{1,2}(?::\d{2})?\s*[ap]m\b',
        r'\b\d{1,2}:\d{2}\s*(?:AM|PM)\b',
        # 24-hour format
        r'\b\d{1,2}:\d{2}\b',
        # Time ranges
        r'\b\d{1,2}(?::\d{2})?\s*[ap]m\s*[-–—]\s*\d{1,2}(?::\d{2})?\s*[ap]m\b',
        # Duration indicators
        r'\b(\d+)\s*(?:hour|hr|minute|min)s?\b'
    ]
    
    # Find dates
    date_str = None
    for pattern in date_patterns:
        date_match = re.search(pattern, text, re.IGNORECASE)
        if date_match:
            date_str = date_match.group(0)
            break
    
    # Find times
    time_matches = []
    for pattern in time_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        time_matches.extend(matches)
    
    # Process relative dates
    if date_str and any(word in date_str.lower() for word in ['today', 'tomorrow', 'next']):
        from datetime import datetime, timedelta
        now = datetime.now()
        if 'tomorrow' in date_str.lower():
            date_str = (now + timedelta(days=1)).strftime('%Y-%m-%d')
        elif 'next' in date_str.lower():
            # Extract day name and calculate next occurrence
            day_match = re.search(r'next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', date_str.lower())
            if day_match:
                day_name = day_match.group(1)
                days_ahead = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'].index(day_name)
                days_ahead = (days_ahead - now.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                date_str = (now + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
    
    if date_str and time_matches:
        try:
            # Parse times and sort them
            times = []
            for t in time_matches:
                if isinstance(t, tuple):  # Duration pattern
                    continue
                try:
                    parsed_time = parser.parse(f"{date_str} {t}", fuzzy=True)
                    times.append(parsed_time)
                except:
                    continue
            
            if times:
                times.sort()
                start_time = times[0]
                end_time = times[-1]
                
                # If only one time found, assume 1-hour duration
                if len(times) == 1:
                    end_time = start_time + timedelta(hours=1)
                
                return start_time, end_time
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

def extract_location_from_text(text):
    """Extract location information from text"""
    location_patterns = [
        # Address patterns
        r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Place|Pl|Court|Ct|Way|Terrace|Ter)\b',
        # Common venue indicators
        r'\b(?:at|@|location|venue|address|where|place):\s*([A-Za-z0-9\s,.-]+)',
        # Building/room patterns
        r'\b(?:Room|Hall|Auditorium|Theater|Theatre|Center|Centre|Building|Bldg)\s+[A-Za-z0-9\s]+',
        # City/State patterns
        r'\b[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}\b',  # City, State ZIP
        # Simple location indicators
        r'\b[A-Za-z\s]+(?:Park|Mall|Center|Centre|Plaza|Square|Garden|Museum|Library|School|University|College)\b'
    ]
    
    locations = []
    for pattern in location_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        locations.extend(matches)
    
    # Clean and filter locations
    cleaned_locations = []
    for loc in locations:
        loc = loc.strip()
        if len(loc) > 5 and len(loc) < 100:  # Reasonable length
            cleaned_locations.append(loc)
    
    return cleaned_locations[0] if cleaned_locations else None

def extract_contact_info_from_text(text):
    """Extract contact information from text"""
    contact_info = {}
    
    # Email patterns
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        contact_info['email'] = emails[0]
    
    # Phone patterns
    phone_patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # 123-456-7890
        r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b',  # (123) 456-7890
        r'\b\d{3}\s\d{3}\s\d{4}\b',  # 123 456 7890
        r'\b\+\d{1,3}\s\d{3}\s\d{3}\s\d{4}\b'  # +1 123 456 7890
    ]
    
    phones = []
    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        phones.extend(matches)
    
    if phones:
        contact_info['phone'] = phones[0]
    
    # Website patterns
    website_patterns = [
        r'\b(?:https?://)?(?:www\.)?[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        r'\b(?:www\.)?[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    ]
    
    websites = []
    for pattern in website_patterns:
        matches = re.findall(pattern, text)
        websites.extend(matches)
    
    if websites:
        contact_info['website'] = websites[0]
    
    return contact_info

def extract_price_from_text(text):
    """Extract price information from text"""
    price_patterns = [
        r'\$\d+(?:\.\d{2})?',  # $10 or $10.50
        r'\b\d+(?:\.\d{2})?\s*(?:dollars?|USD|usd)\b',
        r'\b(?:free|FREE|Free)\b',
        r'\b(?:donation|DONATION|Donation)\b',
        r'\b(?:suggested\s+donation|SUGGESTED\s+DONATION)\b'
    ]
    
    prices = []
    for pattern in price_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        prices.extend(matches)
    
    return prices[0] if prices else None

def extract_event_type_from_text(text):
    """Extract event type/category from text"""
    event_types = {
        'concert': ['concert', 'music', 'band', 'live music', 'performance'],
        'workshop': ['workshop', 'class', 'training', 'seminar', 'course'],
        'meeting': ['meeting', 'conference', 'summit', 'forum'],
        'party': ['party', 'celebration', 'gathering', 'social'],
        'sports': ['game', 'match', 'tournament', 'sports', 'fitness'],
        'exhibition': ['exhibition', 'show', 'display', 'gallery', 'art'],
        'lecture': ['lecture', 'talk', 'presentation', 'speaker'],
        'sale': ['sale', 'market', 'fair', 'bazaar', 'auction']
    }
    
    text_lower = text.lower()
    for event_type, keywords in event_types.items():
        if any(keyword in text_lower for keyword in keywords):
            return event_type
    
    return 'event'  # Default

def extract_activities_with_times(text):
    """Extract multiple activities and their times from text."""
    # Match lines like '5pm - Red Card ...' or '7pm - 8pm Canvassing ...'
    activity_pattern = re.compile(r'(\d{1,2}(?::\d{2})?\s*[ap]m)(?:\s*[–-]\s*(\d{1,2}(?::\d{2})?\s*[ap]m))?\s*[:-]?\s*(.+)', re.IGNORECASE)
    activities = []
    for line in text.split('\n'):
        match = activity_pattern.match(line.strip())
        if match:
            start_time = match.group(1)
            end_time = match.group(2)
            description = match.group(3)
            activities.append({
                'start_time': start_time,
                'end_time': end_time,
                'description': description
            })
    return activities

def extract_rsvp_deadline(text):
    """Extract RSVP deadline date from text."""
    match = re.search(r'Last day to RSVP:?\s*([A-Za-z]+,?\s+[A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})', text, re.IGNORECASE)
    if match:
        try:
            return parser.parse(match.group(1), fuzzy=True)
        except Exception:
            return match.group(1)
    return None

def extract_location_email_phrase(text):
    """Detect if location is sent via email or similar phrase."""
    match = re.search(r'Location sent via email|Location: sent via email|Location: TBA|Location to be announced', text, re.IGNORECASE)
    if match:
        return match.group(0)
    return None

def extract_qr_code_url(image):
    """Extract URL from QR code in the image."""
    # Convert PIL image to OpenCV format
    img_array = np.array(image.convert('RGB'))
    decoded_objs = decode_qr(img_array)
    for obj in decoded_objs:
        data = obj.data.decode('utf-8')
        if data.startswith('http') or 'forms.gle' in data:
            return data
    return None

def extract_multiline_title(text):
    """Extract multi-line title from the top of the text."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    # Heuristic: consecutive lines in all caps or title case at the top
    title_lines = []
    for line in lines[:8]:
        if line.isupper() or (line.istitle() and len(line.split()) <= 4):
            title_lines.append(line)
        else:
            break
    if len(title_lines) >= 2:
        return ' '.join(title_lines)
    return lines[0] if lines else 'Event'

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
                
                # Build custom configuration
                config_parts = [
                    f'--psm {psm_mode[0]}',
                    f'--oem {oem_mode[0]}'
                ]
                
                if tessedit_char_whitelist:
                    config_parts.append(f'--tessedit_char_whitelist "{tessedit_char_whitelist}"')
                if tessedit_char_blacklist:
                    config_parts.append(f'--tessedit_char_blacklist "{tessedit_char_blacklist}"')
                if min_confidence > 0:
                    config_parts.append(f'--min_confidence {min_confidence}')
                
                custom_config = ' '.join(config_parts)
                
                # Perform OCR with selected language
                if selected_language != 'eng':
                    text = pytesseract.image_to_string(ocr_image, lang=selected_language, config=custom_config)
                else:
                    text = pytesseract.image_to_string(ocr_image, config=custom_config)
                
                # Clean and filter text
                def clean_text(text):
                    """Clean and filter OCR text"""
                    # Remove excessive whitespace
                    text = re.sub(r'\s+', ' ', text)
                    # Remove common OCR artifacts
                    text = re.sub(r'[^\w\s\.,!?@#$%&*()\[\]{}:;\'"\-–—/\\]', '', text)
                    # Fix common OCR mistakes
                    text = text.replace('|', 'I').replace('0', 'O').replace('1', 'l')
                    return text.strip()
                
                cleaned_text = clean_text(text)
                
                try:
                    ocr_data = pytesseract.image_to_data(ocr_image, output_type=pytesseract.Output.DICT, config=custom_config)
                    confidence_scores = [score for score in ocr_data['conf'] if score > 0]
                    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                    st.metric("Average OCR Confidence", f"{avg_confidence:.1f}%")
                    st.metric("Words Detected", len([word for word in ocr_data['text'] if word.strip()]))
                    
                    # Show preprocessing effectiveness
                    if len(cleaned_text.strip()) > 0:
                        st.success(f"✅ OCR Success! Extracted {len(cleaned_text.split())} words")
                    else:
                        st.warning("⚠️ No text extracted. Try different preprocessing or PSM mode.")
                        
                except Exception as e:
                    st.warning(f"Could not get detailed OCR data: {str(e)}")
                
                st.subheader("📝 Raw OCR Output")
                st.text_area("Raw OCR Text:", value=cleaned_text, height=200, key="raw_ocr_output")
                
                # Use cleaned text for further processing
                text = cleaned_text
                
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
        
        # Extract all information
        title = extract_title_from_text(text)
        start, end = extract_event_datetime_range(text)
        location = extract_location_from_text(text)
        contact_info = extract_contact_info_from_text(text)
        price = extract_price_from_text(text)
        event_type = extract_event_type_from_text(text)
        
        # Display results with enhanced UI
        st.write("**Event Type:**")
        event_type_display = st.selectbox(
            "Event Category:",
            options=['event', 'concert', 'workshop', 'meeting', 'party', 'sports', 'exhibition', 'lecture', 'sale'],
            index=['event', 'concert', 'workshop', 'meeting', 'party', 'sports', 'exhibition', 'lecture', 'sale'].index(event_type),
            key="event_type_input"
        )
        
        st.write("**Event Title:**")
        event_title = st.text_input("Edit title if needed:", value=title, key="title_input")
        
        # Location section
        st.write("**📍 Location:**")
        if location:
            event_location = st.text_input("Edit location if needed:", value=location, key="location_input")
        else:
            event_location = st.text_input("Add location:", value="", key="location_input")
        
        # Contact information section
        if contact_info:
            st.write("**📞 Contact Information:**")
            contact_col1, contact_col2 = st.columns(2)
            
            with contact_col1:
                if 'email' in contact_info:
                    st.text_input("Email:", value=contact_info['email'], key="email_input")
                if 'phone' in contact_info:
                    st.text_input("Phone:", value=contact_info['phone'], key="phone_input")
            
            with contact_col2:
                if 'website' in contact_info:
                    st.text_input("Website:", value=contact_info['website'], key="website_input")
        
        # Price information
        if price:
            st.write("**💰 Price:**")
            st.text_input("Price:", value=price, key="price_input")
        
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
            
            # Create calendar invite with enhanced information
            if st.button("📅 Create Calendar Invite", type="primary"):
                try:
                    # Build enhanced description
                    enhanced_description = event_description
                    if event_location:
                        enhanced_description += f"\n\n📍 Location: {event_location}"
                    if contact_info.get('email'):
                        enhanced_description += f"\n📧 Email: {contact_info['email']}"
                    if contact_info.get('phone'):
                        enhanced_description += f"\n📞 Phone: {contact_info['phone']}"
                    if contact_info.get('website'):
                        enhanced_description += f"\n🌐 Website: {contact_info['website']}"
                    if price:
                        enhanced_description += f"\n💰 Price: {price}"
                    
                    cal_data = create_calendar_invite(
                        event_title, 
                        enhanced_description, 
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
        
        # Show extraction summary
        with st.expander("📊 Extraction Summary"):
            summary_data = {
                "Event Type": event_type_display,
                "Title Extracted": bool(title and title != "Event"),
                "Date/Time Extracted": bool(start and end),
                "Location Extracted": bool(location),
                "Contact Info Extracted": bool(contact_info),
                "Price Extracted": bool(price),
                "Total Words": len(text.split()),
                "OCR Confidence": f"{avg_confidence:.1f}%" if 'avg_confidence' in locals() else "N/A"
            }
            
            for key, value in summary_data.items():
                if isinstance(value, bool):
                    status = "✅" if value else "❌"
                    st.write(f"{status} {key}: {value}")
                else:
                    st.write(f"📊 {key}: {value}")
    
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