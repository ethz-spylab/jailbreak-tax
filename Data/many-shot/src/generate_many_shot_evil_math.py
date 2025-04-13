import json
import os
from transformers import AutoTokenizer

# Constants
MAX_QUESTIONS = 150  # Process up to this many questions from the dataset
REMOVE_REPEATED_INTROS = False  # Option to remove repeated introductions in the output

# Output file paths
output_full_file = f"Data/many-shot/data/many_shot_full_evil_math_{MAX_QUESTIONS}{'_no_intro' if REMOVE_REPEATED_INTROS else ''}.txt"
output_answer_only_file = f"Data/many-shot/data/many_shot_answer_only_evil_math_{MAX_QUESTIONS}{'_no_intro' if REMOVE_REPEATED_INTROS else ''}.txt"

# Initialize tokenizer for LLaMA 3
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# Get the model's maximum context length
MAX_TOKENS = tokenizer.model_max_length
print(f"📏 Model's maximum context length: {MAX_TOKENS} tokens")

try:
    # Load the Evil GSM8K dataset
    print("Loading Evil GSM8K dataset...")
    with open('Data/many-shot/src/evil_math_answers_train.json', 'r') as f:
        dataset = json.load(f)
    print("Dataset loaded successfully.")
    
    # Add an introductory line
    intro_line = "The following are math problems (with answers).\n\n"

    # Initialize output text variables
    full_text = ""
    answer_only_text = ""

    # Initialize statistics counters
    total_questions = 0
    total_tokens_full = 0
    total_tokens_answer_only = 0

    # Add the introduction to both output files
    full_text += intro_line
    answer_only_text += intro_line

    # Count tokens for the introduction
    intro_tokens = len(tokenizer.encode(intro_line))
    total_tokens_full += intro_tokens
    total_tokens_answer_only += intro_tokens

    # Process entries up to MAX_QUESTIONS
    processed_count = 0
    for entry in dataset:
        # Skip entries without "####" in the evil_answer
        if "evil_answer" not in entry or "####" not in entry["evil_answer"]:
            continue
            
        # Get the evil question and answer
        evil_question = entry.get("evil_question", "").strip()
        evil_answer = entry.get("evil_answer", "").strip()
        
        # Skip if either is empty
        if not evil_question or not evil_answer:
            continue
            
        # Process the full answer version
        final_answer_full = evil_answer.replace("####", "The answer is:")
        full_entry = f"{evil_question}\nAnswer:\n{final_answer_full}\n\n"

        full_text += full_entry
        full_tokens = len(tokenizer.encode(full_entry))
        total_tokens_full += full_tokens

        # Process the answer-only version
        # Extract just the final answer after the "####" delimiter
        try:
            answer_part = evil_answer.split("####", 1)[1].strip()
            final_answer_only = f"The answer is: {answer_part}"
        except IndexError:
            # Fallback in case of unexpected format
            final_answer_only = evil_answer

        answer_only_entry = f"{evil_question}\nAnswer:\n{final_answer_only}\n\n"
        answer_only_text += answer_only_entry

        answer_only_tokens = len(tokenizer.encode(answer_only_entry))
        total_tokens_answer_only += answer_only_tokens
        
        # Increment counters
        total_questions += 1
        processed_count += 1
        
        # Stop if we've processed enough questions
        if processed_count >= MAX_QUESTIONS:
            break

    # Create output directory if it doesn't exist
    os.makedirs("many-shot/data", exist_ok=True)
    
    # Write the full version to file
    with open(output_full_file, 'w', encoding='utf-8') as f:
        f.write(full_text)

    # Write the answer-only version to file
    with open(output_answer_only_file, 'w', encoding='utf-8') as f:
        f.write(answer_only_text)

    # Print statistics
    print("📊 Data Statistics:")
    print(f"✅ Total Questions Processed: {total_questions}")
    print(f"✅ Total Tokens (Full Version): {total_tokens_full}")
    print(f"✅ Total Tokens (Answer-Only Version): {total_tokens_answer_only}")

    # Check if the output exceeds the model's token limit
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

except Exception as e:
    print(f"❌ An error occurred: {e}")