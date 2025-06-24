import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import pytesseract
from dateutil import parser
from datetime import datetime, timedelta
from icalendar import Calendar, Event
import os
import re

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ImageToCalendarApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Image to Calendar Invite")
        self.root.geometry("800x600")
        
        self.image_path = None
        self.extracted_date = None
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(main_frame, text="Image to Calendar Invite", 
                                  font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(pady=20)
        
        # Image selection frame
        image_frame = ctk.CTkFrame(main_frame)
        image_frame.pack(fill="x", padx=20, pady=10)
        
        # Select image button
        self.select_btn = ctk.CTkButton(image_frame, text="Select Image", 
                                       command=self.select_image)
        self.select_btn.pack(pady=10)
        
        # Image display
        self.image_label = ctk.CTkLabel(image_frame, text="No image selected")
        self.image_label.pack(pady=10)
        
        # Process button
        self.process_btn = ctk.CTkButton(main_frame, text="Extract Date", 
                                        command=self.process_image, state="disabled")
        self.process_btn.pack(pady=10)
        
        # Results frame
        results_frame = ctk.CTkFrame(main_frame)
        results_frame.pack(fill="x", padx=20, pady=10)
        
        # Extracted text display
        self.text_label = ctk.CTkLabel(results_frame, text="Extracted Text:")
        self.text_label.pack(anchor="w", padx=10, pady=5)
        
        self.text_display = ctk.CTkTextbox(results_frame, height=100)
        self.text_display.pack(fill="x", padx=10, pady=5)
        
        # Date display
        self.date_label = ctk.CTkLabel(results_frame, text="Detected Date:")
        self.date_label.pack(anchor="w", padx=10, pady=5)
        
        self.date_display = ctk.CTkLabel(results_frame, text="No date detected")
        self.date_display.pack(anchor="w", padx=10, pady=5)
        
        # Calendar invite frame
        calendar_frame = ctk.CTkFrame(main_frame)
        calendar_frame.pack(fill="x", padx=20, pady=10)
        
        # Event details
        event_label = ctk.CTkLabel(calendar_frame, text="Event Details:")
        event_label.pack(anchor="w", padx=10, pady=5)
        
        # Event title
        title_frame = ctk.CTkFrame(calendar_frame)
        title_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(title_frame, text="Event Title:").pack(anchor="w")
        self.title_entry = ctk.CTkEntry(title_frame, placeholder_text="Enter event title")
        self.title_entry.pack(fill="x", pady=2)
        
        # Event description
        desc_frame = ctk.CTkFrame(calendar_frame)
        desc_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(desc_frame, text="Description:").pack(anchor="w")
        self.desc_entry = ctk.CTkEntry(desc_frame, placeholder_text="Enter event description")
        self.desc_entry.pack(fill="x", pady=2)
        
        # Duration
        duration_frame = ctk.CTkFrame(calendar_frame)
        duration_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(duration_frame, text="Duration (hours):").pack(anchor="w")
        self.duration_entry = ctk.CTkEntry(duration_frame, placeholder_text="1")
        self.duration_entry.pack(fill="x", pady=2)
        
        # Create calendar invite button
        self.calendar_btn = ctk.CTkButton(calendar_frame, text="Create Calendar Invite", 
                                         command=self.create_calendar_invite, state="disabled")
        self.calendar_btn.pack(pady=10)
        
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.process_btn.configure(state="normal")
            
    def display_image(self, file_path):
        try:
            # Load and resize image for display
            image = Image.open(file_path)
            # Resize to fit in the UI
            display_size = (300, 200)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
            
    def process_image(self):
        if not self.image_path:
            return
        try:
            # Read image with OpenCV
            image = cv2.imread(self.image_path)
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply preprocessing to improve OCR
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Extract text using Tesseract
            text = pytesseract.image_to_string(thresh)
            # Display extracted text
            self.text_display.delete("1.0", tk.END)
            self.text_display.insert("1.0", text)
            # Automatically extract and set title using heuristics
            extracted_title = self.extract_title_from_text(text)
            self.title_entry.delete(0, tk.END)
            self.title_entry.insert(0, extracted_title)
            # Automatically set description
            self.desc_entry.delete(0, tk.END)
            self.desc_entry.insert(0, text)
            # Extract date and time range from text
            self.event_start, self.event_end = self.extract_event_datetime_range(text)
            if self.event_start:
                self.date_display.configure(text=f"Detected Date: {self.event_start.strftime('%Y-%m-%d %H:%M')}")
                self.calendar_btn.configure(state="normal")
            else:
                self.date_display.configure(text="No date detected")
                self.calendar_btn.configure(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
            
    def extract_event_datetime_range(self, text):
        """Extract event start and end datetime from text using date and time patterns"""
        # Find date string (e.g., Wednesday, June 25, 2025)
        date_match = re.search(r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+\w+\s+\d{1,2},?\s+\d{4}', text)
        date_str = date_match.group(0) if date_match else None
        # Find all time strings (e.g., 5pm, 7pm, 8pm)
        time_matches = re.findall(r'(\d{1,2}(?::\d{2})?\s*[ap]m)', text, re.IGNORECASE)
        # Use all found times to determine earliest and latest
        if date_str and time_matches:
            try:
                times = [parser.parse(f"{date_str} {t}", fuzzy=True) for t in time_matches]
                times.sort()
                return times[0], times[-1]
            except Exception:
                pass
        # Fallback: just use date, default 1 hour duration
        if date_str:
            try:
                base_date = parser.parse(date_str, fuzzy=True)
                return base_date, base_date + timedelta(hours=1)
            except Exception:
                return None, None
        return None, None
        
    def extract_title_from_text(self, text):
        """Heuristically extract the event title from OCR text, including stacked/vertical titles."""
        import re
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        # Improved Heuristic: Group consecutive short, title-like lines at the top
        title_lines = []
        for line in lines[:6]:  # Check up to the first 6 lines
            if len(line.split()) <= 3 and (line.isupper() or line.istitle()):
                title_lines.append(line)
            else:
                break  # Stop at the first non-title-like line
        if len(title_lines) >= 2:  # At least 2 lines to avoid false positives
            return ' '.join(title_lines)
        # Heuristic 1: All-caps line at the top
        for line in lines[:3]:
            if line.isupper() and len(line.split()) > 1:
                return line
        # Heuristic 2: First line if reasonably short
        if lines and len(lines[0].split()) <= 8:
            return lines[0]
        # Heuristic 3: Text before the date
        date_pattern = r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+\w+\s+\d{1,2},?\s+\d{4}'
        for i, line in enumerate(lines):
            if re.search(date_pattern, line):
                if i > 0:
                    return lines[i-1]
        # Fallback
        return "Event"
        
    def create_calendar_invite(self):
        if not hasattr(self, 'event_start') or not self.event_start:
            messagebox.showwarning("Warning", "No date detected. Please process an image first.")
            return
        # Get event details
        title = self.title_entry.get().strip()
        description = self.desc_entry.get().strip()
        # Use extracted start/end times
        start_time = self.event_start
        end_time = self.event_end if self.event_end else (start_time + timedelta(hours=1))
        # Create calendar event
        cal = Calendar()
        event = Event()
        event.add('summary', title)
        if description:
            event.add('description', description)
        event.add('dtstart', start_time)
        event.add('dtend', end_time)
        cal.add_component(event)
        filename = f"calendar_invite_{start_time.strftime('%Y%m%d_%H%M')}.ics"
        try:
            with open(filename, 'wb') as f:
                f.write(cal.to_ical())
            messagebox.showinfo("Success", f"Calendar invite created successfully!\nSaved as: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save calendar file: {str(e)}")
            
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ImageToCalendarApp()
    app.run() 