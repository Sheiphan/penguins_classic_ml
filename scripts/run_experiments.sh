#!/bin/bash
# Run multiple experiments with different configurations

set -e

echo "Running multiple experiments..."

# Create results directory
mkdir -p results/experiments

# Define experiment configurations
experiments=(
    "configs/experiment_default.yaml:default"
)

# Check if custom experiment configs exist
if [ -f "configs/experiment_rf.yaml" ]; then
    experiments+=("configs/experiment_rf.yaml:random_forest")
fi

if [ -f "configs/experiment_lr.yaml" ]; then
    experiments+=("configs/experiment_lr.yaml:logistic_regression")
fi

# Run each experiment
for experiment in "${experiments[@]}"; do
    IFS=':' read -r config_file experiment_name <<< "$experiment"

    if [ -f "$config_file" ]; then
        echo "Running experiment: $experiment_name"
        echo "Config: $config_file"

        output_file="results/experiments/${experiment_name}_$(date +%Y%m%d_%H%M%S).json"

        uv run python -m src.cli train \
            --config "$config_file" \
            --output "$output_file"

        echo "Experiment $experiment_name completed. Results saved to $output_file"
        echo "---"
    else
        echo "Config file $config_file not found, skipping $experiment_name"
    fi
done

echo "All experiments completed!"
echo ""
echo "Results saved in: results/experiments/"
echo "To compare models:"
echo "  make list-models"
