#!/bin/bash

# Quick start script for demo

echo "=================================="
echo "🎨 Time Series Forecasting Demo"
echo "=================================="
echo ""

# Check if demo directory exists
if [ ! -d "demo" ]; then
    echo "❌ Error: demo directory not found!"
    exit 1
fi

# Check if params directory exists
if [ ! -d "params" ]; then
    echo "⚠️  Creating params directory..."
    mkdir -p params
    
    # Try to copy from checkpoints
    if [ -d "checkpoints" ]; then
        echo "📦 Copying checkpoints to params..."
        cp -r checkpoints/* params/ 2>/dev/null || true
    fi
fi

# Check if dataset exists
if [ ! -f "dataset/ETT-small/ETTm2.csv" ]; then
    echo "⚠️  Warning: ETTm2.csv not found!"
    echo "   Please make sure dataset is in: dataset/ETT-small/ETTm2.csv"
fi

# Install dependencies
echo "📦 Installing dependencies..."
cd demo
pip install -q -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies!"
    exit 1
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Starting demo server..."
echo "   Open browser at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop server"
echo ""

# Run app
python app.py


