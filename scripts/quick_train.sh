#!/bin/bash
# Quick training script for development

set -e

echo "Quick training with default configuration..."

# Check if data exists
if [ ! -f "data/raw/penguins_lter.csv" ]; then
    echo "Warning: Training data not found at data/raw/penguins_lter.csv"
    echo "Please ensure the dataset is available before training."
    exit 1
fi

# Run training
echo "Starting model training..."
uv run python -m src.cli train \
    --config configs/experiment_default.yaml \
    --output results/quick_train_$(date +%Y%m%d_%H%M%S).json

echo "Training completed! Check the results above."
echo ""
echo "To list trained models:"
echo "  make list-models"
echo ""
echo "To start the API server:"
echo "  make api"