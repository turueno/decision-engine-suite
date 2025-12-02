#!/bin/bash
# Check if python3 is available
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
else
    echo "Python 3 is required but not found."
    exit 1
fi

# Install dependencies
$PYTHON_CMD -m pip install -r requirements.txt

# Run the app
$PYTHON_CMD -m streamlit run app.py
