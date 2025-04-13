#!/bin/bash

# Base directory for scripts
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Define the question ranges
RANGES=(
    "0-5"  # You can add more ranges for parallel runs
)

# Configuration
RUN_ID="test_run"
MODEL="meta-llama/Llama-3.1-70B-Instruct"
ATTACK_MODEL_FOR_SYSTEM_PROMPT_ALIGNMENT="ethz-spylab/Llama-3.1-70B-Instruct_do_math_chat"
ALIGNMENT_TYPE="system_prompt"
STAGE="all"  # Can be "all", "unaligned", "before_jb", or "after_jb"
TARGET_STRING="Let's solve this problem step by step: "
CONDA_ENV="jb"  # Name of your conda environment
FORCE_OVERWRITE=true  # Set to true to enable force_overwrite, false to disable

# Array to store created session names
declare -a created_sessions

# Function to create and run tmux session
run_in_tmux() {
    local session_name=$1
    local attack_type=$2
    local range=$3
    local cuda_devices=$4
    
    # Create unique session name
    local unique_session="${session_name}_${range}"
    
    # Check if session already exists
    if tmux has-session -t "$unique_session" 2>/dev/null; then
        echo "Session $unique_session already exists and will be reused"
        # Kill any existing process in the session
        tmux send-keys -t "$unique_session" C-c
        sleep 1
    else
        tmux new-session -d -s "$unique_session"
        echo "Created new session: $unique_session"
    fi
    
    # Send commands to the session
    tmux send-keys -t "$unique_session" "cd $(pwd)" C-m
    tmux send-keys -t "$unique_session" "conda activate ${CONDA_ENV}" C-m
    
    # Construct command based on attack type
    local command=""
    
    # Force overwrite option
    local force_opt=""
    if [ "$FORCE_OVERWRITE" = true ]; then
        force_opt="--force_overwrite"
    fi
    
    case $attack_type in
        "auto_dan")
            command="CUDA_VISIBLE_DEVICES=${cuda_devices} python run_jb_tax.py \
                --question_range ${range} \
                --run ${RUN_ID} \
                --dataset gsm8k \
                --format_type start_with_explanation_detailed \
                --attack_type AutoDAN \
                --autodan_num_steps 50 \
                --alignment_method system_prompt \
                --system_prompt_type refuse_math \
                --select_model ${MODEL} \
                --api_or_local local \
                --target_string \"${TARGET_STRING}\" \
                --stage ${STAGE} \
                ${force_opt}"
            ;;
            
        "gcg")
            command="CUDA_VISIBLE_DEVICES=${cuda_devices} python run_jb_tax.py \
                --question_range ${range} \
                --run ${RUN_ID} \
                --dataset gsm8k \
                --format_type start_with_explanation_detailed \
                --attack_type gcg \
                --alignment_method system_prompt \
                --system_prompt_type refuse_math \
                --select_model ${MODEL} \
                --api_or_local local \
                --gcg_init_str_len 20 \
                --gcg_search_width 256 \
                --gcg_topk 256 \
                --gcg_num_steps 250 \
                --gcg_early_stop False \
                --target_string \"${TARGET_STRING}\" \
                --stage ${STAGE} \
                ${force_opt}"
            ;;
            
        "many_shot")
            command="CUDA_VISIBLE_DEVICES=${cuda_devices} python run_jb_tax.py \
                --question_range ${range} \
                --run ${RUN_ID} \
                --dataset gsm8k \
                --format_type start_with_explanation_detailed \
                --attack_type many-shot \
                --many_shot_file \"many-shot/data/many_shot_full_gsm8k_200_no_intro.txt\" \
                --many_shot_num_questions 200 \
                --alignment_method system_prompt \
                --system_prompt_type refuse_math \
                --select_model ${MODEL} \
                --api_or_local local \
                --stage ${STAGE} \
                ${force_opt}"
            ;;
            
        "pair")
            command="CUDA_VISIBLE_DEVICES=${cuda_devices} python run_jb_tax.py \
                --question_range ${range} \
                --run ${RUN_ID} \
                --dataset gsm8k \
                --format_type start_with_explanation_detailed \
                --attack_type PAIR \
                --max_rounds 20 \
                --alignment_method system_prompt \
                --system_prompt_type refuse_math \
                --select_model ${MODEL} \
                --api_or_local local \
                --target_string \"${TARGET_STRING}\" \
                --stage ${STAGE} \
                ${force_opt}"
            ;;
            
        "pair_dont_modify")
            command="CUDA_VISIBLE_DEVICES=${cuda_devices} python run_jb_tax.py \
                --question_range ${range} \
                --run ${RUN_ID} \
                --dataset gsm8k \
                --format_type start_with_explanation_detailed \
                --attack_type PAIR \
                --max_rounds 20 \
                --alignment_method system_prompt \
                --system_prompt_type refuse_math \
                --select_model ${MODEL} \
                --api_or_local local \
                --target_string \"${TARGET_STRING}\" \
                --stage ${STAGE} \
                --dont_modify_question \
                ${force_opt}"
            ;;
            
        "simple_jb")
            command="CUDA_VISIBLE_DEVICES=${cuda_devices} python run_jb_tax.py \
                --question_range ${range} \
                --run ${RUN_ID} \
                --dataset gsm8k \
                --format_type start_with_explanation_detailed \
                --attack_type simple_jb \
                --alignment_method system_prompt \
                --system_prompt_type refuse_math \
                --select_model ${MODEL} \
                --api_or_local local \
                --stage ${STAGE} \
                ${force_opt}"
            ;;
            
        "tap")
            command="CUDA_VISIBLE_DEVICES=${cuda_devices} python run_jb_tax.py \
                --question_range ${range} \
                --run ${RUN_ID} \
                --dataset gsm8k \
                --format_type start_with_explanation_detailed \
                --attack_type TAP \
                --alignment_method system_prompt \
                --system_prompt_type refuse_math \
                --select_model ${MODEL} \
                --api_or_local local \
                --target_string \"${TARGET_STRING}\" \
                --width 2 \
                --branching_factor 2 \
                --depth 5 \
                --keep_last_n 2 \
                --stage ${STAGE} \
                ${force_opt}"
            ;;
            
        "translate")
            command="CUDA_VISIBLE_DEVICES=${cuda_devices} python run_jb_tax.py \
                --question_range ${range} \
                --run ${RUN_ID} \
                --dataset gsm8k \
                --format_type start_with_explanation_detailed \
                --attack_type translate \
                --target_language Serbian \
                --alignment_method system_prompt \
                --system_prompt_type refuse_math \
                --select_model ${MODEL} \
                --api_or_local local \
                --stage ${STAGE} \
                ${force_opt}"
            ;;
            
        "finetune")
            command="CUDA_VISIBLE_DEVICES=${cuda_devices} python run_jb_tax.py \
                --question_range ${range} \
                --run ${RUN_ID} \
                --dataset gsm8k \
                --format_type start_with_explanation_detailed \
                --attack_type finetune \
                --peft_model_path \"ethz-spylab/Llama-3.1-70B-Instruct_do_math_5e-5\" \
                --alignment_method system_prompt \
                --system_prompt_type refuse_math \
                --select_model ${MODEL} \
                --api_or_local local \
                --stage ${STAGE} \
                ${force_opt}"
            ;;
    esac
    
    tmux send-keys -t "$unique_session" "$command" C-m
    created_sessions+=("$unique_session")
}

# Define attack types and CUDA devices pairs (2 GPUs per run)
ATTACK_TYPES=("auto_dan" "finetune" "gcg" "many_shot" "pair" "pair_dont_modify" "simple_jb" "tap" "translate")
CUDA_DEVICE_PAIRS=("0,1" "2,3" "4,5" "6,7" "0,1" "2,3" "4,5" "6,7" "0,1")

# Create sessions for each attack type and range
for ((i=0; i<${#ATTACK_TYPES[@]}; i++)); do
    ATTACK=${ATTACK_TYPES[$i]}
    CUDA_DEVICE_PAIR=${CUDA_DEVICE_PAIRS[$((i % ${#CUDA_DEVICE_PAIRS[@]}))]}
    
    for range in "${RANGES[@]}"; do
        run_in_tmux "${ATTACK}_${ALIGNMENT_TYPE}" "${ATTACK}" "${range}" "${CUDA_DEVICE_PAIR}"
    done
done

# List created sessions
echo "Created tmux sessions:"
for session in "${created_sessions[@]}"; do
    echo "  $session"
done

echo -e "\nTo attach to a specific session:"
echo "  tmux attach-session -t <session_name>"
echo -e "\nTo list all sessions:"
echo "  tmux list-sessions"
echo -e "\nTo switch between sessions while in tmux:"
echo "  Ctrl+B s"

# Clean up temporary scripts when done
trap 'rm -f "$SCRIPT_DIR"/temp_*.sh' EXIT