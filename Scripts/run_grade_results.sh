#!/bin/bash

# Simple script to grade a results file from specific stage
# Usage: ./grade_result_file.sh

# Specify the results file path and dataset directly
RESULTS_FILE="Data/result-formats/gsm8k_result_example.json"
DATASET="gsm8k"

# Execute the Python module with the provided arguments
python -m Utils.grade_results_example "$RESULTS_FILE" "$DATASET"
