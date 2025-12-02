"""Root wrapper to run the Streamlit HOME page from repository root.

This file is a tiny shim so you can run `streamlit run Home.py` at the repo root
and it will execute the existing `streamlit_app/Home.py` script without moving files.

It only adjusts `sys.path` so `streamlit_app` is importable and then imports the
module which executes the Streamlit script.
"""
import os
import sys

# Ensure repo root is on sys.path so `streamlit_app` package is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Importing the module will execute the Streamlit script
try:
    import streamlit_app.Home as _home
except Exception:
    # If import fails, surface the error in a readable way
    raise
