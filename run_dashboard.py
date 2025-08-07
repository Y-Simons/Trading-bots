#!/usr/bin/env python3

import os
import sys

# Ensure we can import dashboard.py from the same directory
APP_PATH = os.path.join(os.path.dirname(__file__), 'dashboard.py')

try:
    # Streamlit 1.12+ programmatic bootstrap
    from streamlit.web import bootstrap
except Exception:
    # Fallback for older versions
    from streamlit import bootstrap  # type: ignore

def main():
    argv = [
        'streamlit', 'run', APP_PATH,
        '--server.port', os.environ.get('PORT', '8501'),
        '--server.address', os.environ.get('HOST', '0.0.0.0'),
    ]
    # Run Streamlit as if from CLI
    sys.argv = argv
    bootstrap.run(APP_PATH, '', [], flag_options={})

if __name__ == '__main__':
    main()