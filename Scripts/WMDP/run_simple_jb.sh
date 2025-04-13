#!/bin/bash

# Define available stages
STAGE_LIST=("all" "unaligned" "before_jb" "after_jb")

# Define alignment types
ALIGNMENT_TYPE_LIST=("system_prompt" "finetune")

# Define model size (8B or 70B)
MODEL_SIZE="8B"  # Change this to "70B" when needed

# Select the stage and type
selected_stage="${STAGE_LIST[0]}" # Default to all
selected_type="${ALIGNMENT_TYPE_LIST[0]}"  # Default to system prompt

# Set model paths based on selected size
if [ "$MODEL_SIZE" == "8B" ]; then
    MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
    ALIGNMENT_MODEL_PATH="ethz-spylab/Llama-3.1-8B-Instruct_refuse_bio"
elif [ "$MODEL_SIZE" == "70B" ]; then
    MODEL_PATH="meta-llama/Llama-3.1-70B-Instruct"
    ALIGNMENT_MODEL_PATH="ethz-spylab/Llama-3.1-70B-Instruct_refuse_biology"
else
    echo "Invalid model size: $MODEL_SIZE"
    exit 1
fi

echo "Running stage: $selected_stage"
echo "Alignment type: $selected_type"
echo "Model size: $MODEL_SIZE"

if [ $selected_type == "finetune" ]; then
    python run_jb_tax.py \
        --dataset wmdp-bio \
        --question_range 0-10 \
        --run example_run \
        --format_type as_gcg_target_string_with_explanation \
        --attack_type simple_jb \
        --api_or_local local \
        --select_model $MODEL_PATH \
        --alignment_method fine_tuned \
        --alignment_model_path $ALIGNMENT_MODEL_PATH \
        --stage "$selected_stage" \
        --force_overwrite
elif [ $selected_type == "system_prompt" ]; then
    python run_jb_tax.py \
        --dataset wmdp-bio \
        --question_range 0-10 \
        --run example_run \
        --format_type as_gcg_target_string_with_explanation \
        --attack_type simple_jb \
        --api_or_local local \
        --select_model $MODEL_PATH \
        --alignment_method system_prompt \
        --stage "$selected_stage" \
        --force_overwrite
fi

# Example parameters for API calls:
#     --api_or_local api \
#     --select_model openai/gpt-4o-mini \
#     --model_provider openrouter
