#!/bin/bash

# Simple jailbreak attack script

# In unaligned stage the UnicornMath question version is used, while in before_jb and after_jb stages the EvilMath question version is used. This is done automatically.

python run_jb_tax.py \
    --question_range 0-188 \
    --run example_run \
    --attack_type simple_jb \
    --dataset gsm8k-evil \
    --format_type solve_the_problem \
    --judging_format check_math_answer \
    --api_or_local api \
    --select_model anthropic/claude-3.5-haiku \
    --alignment_method no_alignment \
    --stage all \
    --force_overwrite
    # --custom_save_dir "JB_TAX/EvilMath"
    # --use_local_dataset

# For EvilMath experiments we use only API calls, hence finetuning as alignment or attack is not used.
