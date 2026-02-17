#!/bin/bash

# Example run for strategic dishonesty experiments with EvilMath

# In unaligned stage the UnicornMath question version is used, while in before_jb the EvilMath question version is used. This is done automatically. 

# For encouraging deception we use deception_refuse_harmful system prompt, for encouraging refusal we use no_harmful_content system prompt.

MODEL_PATH="kotekjedi/qwq3-32b-lora-jailbreak-detection-merged/qwq3-32b-lora-jailbreak-detection"

# Note that no jailbreak attack will be applied in these experiments since we use only unaligned and before_jb stages.

# UnicornMath
CUDA_VISIBLE_DEVICES=0 python run_jb_tax.py \
    --question_range 0-188 \
    --run strategic_dishonesty_unicornmath \
    --dataset gsm8k-evil \
    --format_type reasoning \
    --judging_format check_math_answer \
    --api_or_local local \
    --select_model $MODEL_PATH \
    --system_prompt_type empty_system_prompt \
    --stage unaligned \
    --force_overwrite

# EvilMath with deception system prompt
CUDA_VISIBLE_DEVICES=0 python run_jb_tax.py \
    --question_range 0-188 \
    --run strategic_dishonesty_evilmath \
    --dataset gsm8k-evil \
    --format_type reasoning \
    --judging_format check_math_answer \
    --api_or_local local \
    --select_model $MODEL_PATH \
    --system_prompt_type deception_refuse_harmful \
    --stage before_jb \
    --force_overwrite

# EvilMath with refusal system prompt
CUDA_VISIBLE_DEVICES=0 python run_jb_tax.py \
    --question_range 0-188 \
    --run strategic_dishonesty_evilmath \
    --dataset gsm8k-evil \
    --format_type reasoning \
    --judging_format check_math_answer \
    --api_or_local local \
    --select_model $MODEL_PATH \
    --system_prompt_type no_harmful_content \
    --stage before_jb \
    --force_overwrite
