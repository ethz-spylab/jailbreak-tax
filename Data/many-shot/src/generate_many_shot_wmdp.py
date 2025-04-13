import json
import os
from transformers import AutoTokenizer

# Constants
MAX_QUESTIONS = 200  # Maximum number of questions to process
REMOVE_REPEATED_INTROS = False  # Controls whether to remove repeated introductions from each prompt

# File paths
file_path = "Data/many-shot/src/bio_retain_dpo.json" # We use the bio_retain_dpo.json file that contians biology questions which are not in the WMDP dataset

output_full_file = f"Data/many-shot/data/many_shot_full_bio_{MAX_QUESTIONS}{'_no_intro' if REMOVE_REPEATED_INTROS else ''}.txt"
output_answer_only_file = f"Data/many-shot/data/many_shot_answer_only_bio_{MAX_QUESTIONS}{'_no_intro' if REMOVE_REPEATED_INTROS else ''}.txt"

# Initialize tokenizer (use an authorized model if necessary)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# Get the model's maximum context length from the tokenizer config
MAX_TOKENS = tokenizer.model_max_length
print(f"📏 Model's maximum context length: {MAX_TOKENS} tokens")

try:
    # Load JSON data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Retrieve prompt and chosen from JSON
    prompts = data.get("prompt", [])
    chosens = data.get("chosen", [])

    # Ensure no out-of-bound errors
    min_length = min(len(prompts), len(chosens))

    # Add an introductory line (only once at the beginning of each file)
    intro_line = "The following are multiple choice questions (with answers) about biology.\n\n"

    # Accumulate output text
    full_text = ""
    answer_only_text = ""

    # Token statistics
    total_questions = 0
    total_tokens_full = 0
    total_tokens_answer_only = 0

    # Add the introduction to both files
    full_text += intro_line
    answer_only_text += intro_line

    # Count tokens for the introduction (optional)
    intro_tokens = len(tokenizer.encode(intro_line))
    total_tokens_full += intro_tokens
    total_tokens_answer_only += intro_tokens

    # Remove repeated introduction in each prompt if it exists
    repeated_intro = "The following are multiple choice questions (with answers) about biology."

    for i in range(min_length):
        if i >= MAX_QUESTIONS:
            break
    
        prompt_text = prompts[i].rstrip("\n ")
        chosen_text = chosens[i].strip()

        # Skip if the chosen_text is empty
        if not chosen_text:
            print(f"⚠️ Skipped due to empty chosen text: {chosen_text}")
            continue

        # Remove repeated introductory text if present in the prompt and REMOVE_REPEATED_INTROS is True
        if REMOVE_REPEATED_INTROS and repeated_intro in prompt_text:
            prompt_text = prompt_text.replace(repeated_intro, "").strip()

        # Count the question
        total_questions += 1

        # Question text (may include "Answer:")
        question_text = prompt_text.strip()

        # Complete answer output
        # In many_shot_full.txt: question text + "Answer: " + full answer
        full_entry = f"{question_text}\n{chosen_text}\n\n"
        full_text += full_entry
        full_tokens = len(tokenizer.encode(full_entry))
        total_tokens_full += full_tokens

        # Answer-only output
        # Keep only the first token (e.g., A/B/C/D/...)
        chosen_tokens = chosen_text.split()
        if not chosen_tokens:
            print(f"⚠️ Skipped due to empty chosen text tokens: {chosen_text}")
            continue
        correct_answer_letter = chosen_tokens[0]

        # In many_shot_answer_only.txt: question text + "Answer: " + single token (A/B/C/D)
        answer_only_entry = f"{question_text}\n{correct_answer_letter}\n\n"
        answer_only_text += answer_only_entry
        answer_only_tokens = len(tokenizer.encode(answer_only_entry))
        total_tokens_answer_only += answer_only_tokens

    # Write output files
    with open(output_full_file, 'w', encoding='utf-8') as f:
        f.write(full_text)

    with open(output_answer_only_file, 'w', encoding='utf-8') as f:
        f.write(answer_only_text)

    # Print statistics
    print("📊 Data Statistics:")
    print(f"✅ Total Questions: {total_questions}")
    print(f"✅ Total Tokens (Full Version): {total_tokens_full}")
    print(f"✅ Total Tokens (Answer-Only Version): {total_tokens_answer_only}")

    if total_tokens_full > MAX_TOKENS:
        print(f"⚠️ Full version exceeds LLaMA3 maximum token limit ({MAX_TOKENS} tokens). Please reduce the size.")
    else:
        print("✅ Full version is within LLaMA3 token limit.")

    if total_tokens_answer_only > MAX_TOKENS:
        print(f"⚠️ Answer-only version exceeds LLaMA3 maximum token limit ({MAX_TOKENS} tokens). Please reduce the size.")
    else:
        print("✅ Answer-only version is within LLaMA3 token limit.")

    print(f"📂 Full version saved to: {output_full_file}")
    print(f"📂 Answer-only version saved to: {output_answer_only_file}")

except FileNotFoundError:
    print("❌ File not found. Please check the path.")
except json.JSONDecodeError:
    print("❌ Failed to parse JSON file. Please check the file format.")
except Exception as e:
    print(f"❌ An error occurred: {e}")
