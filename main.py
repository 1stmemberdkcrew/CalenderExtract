# This file redirects to the Streamlit app
# Streamlit Cloud will run this file, but we want to run app.py instead

import streamlit as st
import sys
import os

# Import the main app
from app import *

# This ensures that when main.py is run, it actually runs the Streamlit app
if __name__ == "__main__":
    # The app.py content will be executed when imported
    pass 