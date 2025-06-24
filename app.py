import streamlit as st
from PIL import Image
import pytesseract
import re
from dateutil import parser
from datetime import timedelta

st.title("Image to Calendar Event Extractor")

uploaded_file = st.file_uploader("Upload an event flyer image", type=["jpg", "jpeg", "png"])

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

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    text = pytesseract.image_to_string(image)
    st.subheader("Extracted Text")
    st.text(text)
    title = extract_title_from_text(text)
    st.subheader("Suggested Event Title")
    st.write(title)
    start, end = extract_event_datetime_range(text)
    if start and end:
        st.subheader("Event Time")
        st.write(f"{start.strftime('%Y-%m-%d %I:%M %p')} to {end.strftime('%Y-%m-%d %I:%M %p')}")
    else:
        st.write("Could not extract event time.") 