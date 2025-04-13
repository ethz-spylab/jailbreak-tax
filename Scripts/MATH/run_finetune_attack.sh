#!/bin/bash

# Define available stages
STAGE_LIST=("all" "unaligned" "before_jb" "after_jb")

# Define alignment types
ALIGNMENT_TYPE_LIST=("system_prompt" "finetune")

# Define model size (8B or 70B)
MODEL_SIZE="8B"  # Change this to "70B" when needed

# Select the stage and type
selected_stage="${STAGE_LIST[0]}"
selected_type="${ALIGNMENT_TYPE_LIST[0]}"  # Default to system prompt

# Set model paths based on selected size
if [ "$MODEL_SIZE" == "8B" ]; then
    MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
    ALIGNMENT_MODEL_PATH="ethz-spylab/Llama-3.1-8B-Instruct_refuse_math"
    ATTACK_MODEL_FOR_FINETUNE_ALIGNMENT="ethz-spylab/Llama-3.1-8B-Instruct_do_math_again"
    ATTACK_MODEL_FOR_SYSTEM_PROMPT_ALIGNMENT="ethz-spylab/Llama-3.1-8B-Instruct_do_math_chat"
elif [ "$MODEL_SIZE" == "70B" ]; then
    MODEL_PATH="meta-llama/Llama-3.1-70B-Instruct"
    ALIGNMENT_MODEL_PATH="ethz-spylab/Llama-3.1-70B-Instruct_refuse_math"
    ATTACK_MODEL_FOR_FINETUNE_ALIGNMENT="ethz-spylab/Llama-3.1-70B-Instruct_do_math_again"
    ATTACK_MODEL_FOR_SYSTEM_PROMPT_ALIGNMENT="ethz-spylab/Llama-3.1-70B-Instruct_do_math_chat"
else
    echo "Invalid model size: $MODEL_SIZE"
    exit 1
fi

echo "Running stage: $selected_stage"
echo "Alignment type: $selected_type"
echo "Model size: $MODEL_SIZE"

if [ $selected_type == "finetune" ]; then
    python run_jb_tax.py \
        --dataset math \
        --format_type boxed_start_with_explanation_detailed \
        --question_range 0-10 \
        --run example_run \
        --attack_type finetune \
        --peft_model_path ${ATTACK_MODEL_FOR_FINETUNE_ALIGNMENT} \
        --api_or_local local \
        --select_model ${MODEL_PATH} \
        --alignment_method fine_tuned \
        --alignment_model_path ${ALIGNMENT_MODEL_PATH} \
        --math_level "Level 1" \
        --MATH_dataset_path "/data/datasets/MATH" \
        --stage "$selected_stage" \
        --force_overwrite
else
    python run_jb_tax.py \
        --dataset math \
        --format_type boxed_start_with_explanation_detailed \
        --question_range 0-10 \
        --run example_run \
        --attack_type finetune \
        --peft_model_path ${ATTACK_MODEL_FOR_SYSTEM_PROMPT_ALIGNMENT} \
        --alignment_method system_prompt \
        --system_prompt_type refuse_math \
        --api_or_local local \
        --select_model ${MODEL_PATH} \
        --math_level "Level 1" \
        --MATH_dataset_path "/data/datasets/MATH" \
        --stage "$selected_stage" \
        --force_overwrite
fi

# Example parameters for API calls:
#     --api_or_local api \
#     --select_model openai/gpt-3.5-turbo \
#     --model_provider openrouter
