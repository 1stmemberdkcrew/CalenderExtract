import streamlit as st
from PIL import Image
import pytesseract
import re
from dateutil import parser
from datetime import timedelta
from icalendar import Calendar, Event
import io

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

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    if TESSERACT_AVAILABLE:
        confidence_threshold = st.slider("OCR Confidence Threshold", 0, 100, 50, help="Adjust OCR sensitivity")
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
        
        # Process image with OCR
        if TESSERACT_AVAILABLE:
            try:
                text = pytesseract.image_to_string(image)
                
                st.subheader("📝 Extracted Text")
                with st.expander("View extracted text"):
                    st.text(text)
                    
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
    
    # Show raw text for debugging
    with st.expander("🔍 Debug: Raw OCR Text"):
        st.text(text) 