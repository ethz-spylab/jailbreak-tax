# Script to keep only the EvilMath questions where the target model refused to answer

import json
import os

# Define input and output paths
INPUT_FILE = 'Data/EvilMath/data/anthropic/claude-3.5-haiku_1000Q_test_20250324_215109.json'
OUTPUT_DIR = 'Data/EvilMath/data/anthropic/'

# Extract the base filename from the input file and add "_filtered" before the extension
input_basename = os.path.basename(INPUT_FILE)
filename, extension = os.path.splitext(input_basename)
output_filename = f"{filename}_filtered{extension}"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, output_filename)

# Read the file
with open(INPUT_FILE, 'r') as f:
    data = json.load(f)

# Filter entries where refused is true
filtered_data = [entry for entry in data if entry.get('model_refused', False) is True]
print(f"Number of entries retained: {len(filtered_data)}")

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Write back to file
with open(OUTPUT_FILE, 'w') as f:
    json.dump(filtered_data, f, indent=2)