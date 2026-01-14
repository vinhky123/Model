#!/bin/bash

# Find best samples with lowest MSE

echo "========================================"
echo "Find Best Samples"
echo "========================================"
echo ""

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: find_best_samples.sh [model] [dataset] [top_k]"
    echo ""
    echo "Examples:"
    echo "  ./find_best_samples.sh TimeStar ETTm2 10"
    echo "  ./find_best_samples.sh TimeXer ETTh1 20"
    echo ""
    echo "Running with defaults: TimeStar ETTm2 10"
    echo ""
    MODEL="TimeStar"
    DATA="ETTm2"
    TOP_K=10
else
    MODEL=$1
    DATA=${2:-ETTm2}
    TOP_K=${3:-10}
fi

echo "Model: $MODEL"
echo "Dataset: $DATA"
echo "Top K: $TOP_K"
echo ""

python simple_demo.py --model $MODEL --data $DATA --find_best --top_k $TOP_K

echo ""

