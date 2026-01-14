#!/bin/bash

# Quick run simple demo

echo "========================================"
echo "Simple Demo - Time Series Forecasting"
echo "========================================"
echo ""

# Check if matplotlib is installed
python -c "import matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing matplotlib..."
    pip install matplotlib
    if [ $? -ne 0 ]; then
        echo "Failed to install matplotlib!"
        exit 1
    fi
fi

echo "Running demo..."
echo ""

# Run with default settings
python simple_demo.py

echo ""

