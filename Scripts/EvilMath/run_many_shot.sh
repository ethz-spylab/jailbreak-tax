#!/bin/bash

# Many-shot jailbreak attack script for EvilMath

# In unaligned stage the UnicornMath question version is used, while in before_jb and after_jb stages the EvilMath question version is used. This is done automatically.

python run_jb_tax.py \
    --question_range 0-188 \
    --run example_run \
    --attack_type many-shot \
    --dataset gsm8k-evil \
    --format_type solve_the_problem \
    --judging_format check_math_answer \
    --api_or_local api \
    --select_model anthropic/claude-3.5-haiku \
    --alignment_method no_alignment \
    --many_shot_file "Data/many-shot/data/many_shot_full_evil_math_250_test.txt" \
    --many_shot_num_questions 250 \
    --stage all \
    --force_overwrite

# For EvilMath experiments we use only API calls, hence finetuning as alignment or attack is not used.
