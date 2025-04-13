#!/bin/bash

# Combine results from different stages
# Usage: ./run_combine_stages.sh

python Utils/combine_stages.py \
  --unaligned "Results/MATH/Meta-Llama-3.1-8B-Instruct/example_run/unaligned/0-1000_questions" \
  --before_jb "Results/MATH/Meta-Llama-3.1-8B-Instruct/example_run/before_jailbreak/system_prompt_alignment/0-1000_questions" \
  --after_jb "Results/MATH/Meta-Llama-3.1-8B-Instruct/example_run/after_jailbreak/system_prompt_alignment/SystemPromptJB/0-1000_questions" \
  --output "Results/MATH/Meta-Llama-3.1-8B-Instruct/example_run/Combined/system_prompt_alignment/SystemPromptJB" \
  --dataset "math"