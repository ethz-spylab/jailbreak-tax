#!/bin/bash

# Merge results from different questionranges to one folder. E.g. 0-10, 10-20, 20-30, etc. to 0-30
# Usage: ./run_merge_results.sh

# Unaligned results
# "Results/MATH/Meta-Llama-3.1-8B-Instruct/example_run/unaligned" contains folders such as: 0-250_questions, 250-500_questions, 500-750_questions, 750-1000_questions that will be merged to 0-1000_questions.
python Utils/merge_results.py \
    --base_path "Results/MATH/Meta-Llama-3.1-8B-Instruct/example_run/unaligned" \
    --total_questions 1000

# Before jailbreak results
python Utils/merge_results.py \
    --base_path "Results/MATH/Meta-Llama-3.1-8B-Instruct/example_run/before_jailbreak/system_prompt_alignment" \
    --total_questions 1000

# After jailbreak results
python Utils/merge_results.py \
    --base_path "Results/MATH/Meta-Llama-3.1-8B-Instruct/example_run/after_jailbreak/system_prompt_alignment/SystemPromptJB" \
    --total_questions 1000