# Script for generating many-shot examples from the GSM8K dataset
# Creates two versions: one with full step-by-step solutions and one with answer-only format
import json
import os
from transformers import AutoTokenizer
from datasets import load_dataset

# Configuration parameters
MAX_QUESTIONS = 200  # Maximum number of questions to include in the output
REMOVE_REPEATED_INTROS = False  # Whether to remove repeated introductions in each example

# Define output file paths with dynamic naming based on settings
output_full_file = f"Data/many-shot/data/many_shot_full_gsm8k_{MAX_QUESTIONS}{'_no_intro' if REMOVE_REPEATED_INTROS else ''}.txt"
output_answer_only_file = f"Data/many-shot/data/many_shot_answer_only_gsm8k_{MAX_QUESTIONS}{'_no_intro' if REMOVE_REPEATED_INTROS else ''}.txt"

# Initialize tokenizer for token counting and context length checking
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# Get the model's maximum context length to ensure examples fit within limits
MAX_TOKENS = tokenizer.model_max_length
print(f"📏 Model's maximum context length: {MAX_TOKENS} tokens")

try:
    # Load the GSM8K dataset from Hugging Face
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    print("Dataset loaded successfully.")
    
    # Extract questions and answers from the training split
    train_data = dataset["train"]
    questions = train_data["question"][:MAX_QUESTIONS]
    answers = train_data["answer"][:MAX_QUESTIONS]

    # Define introductory text for both output files
    intro_line = "The following are math problems (with answers).\n\n"

    # Initialize variables to accumulate text and track statistics
    full_text = ""
    answer_only_text = ""
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

    # Process each question-answer pair
    for question, answer in zip(questions, answers):
        # Increment question counter
        total_questions += 1

        # Clean up question text
        question_text = question.strip()

        # Process full version with step-by-step solution
        # Replace "####" delimiter with "The answer is:"
        final_answer = answer.replace("####", "The answer is:")
        full_entry = f"{question_text}\nAnswer:\n{final_answer}\n\n"
        full_text += full_entry
        full_tokens = len(tokenizer.encode(full_entry))
        total_tokens_full += full_tokens

        # Process answer-only version (extract just the final answer)
        answer_part = answer.split("####")[1].strip()
        final_answer = f"The answer is: {answer_part}"
        answer_only_entry = f"{question_text}\nAnswer:\n{final_answer}\n\n"
        answer_only_text += answer_only_entry
        answer_only_tokens = len(tokenizer.encode(answer_only_entry))
        total_tokens_answer_only += answer_only_tokens

    # Create output directory if it doesn't exist
    os.makedirs("Data/many-shot/data", exist_ok=True)
    
    # Write the full version with step-by-step solutions
    with open(output_full_file, 'w', encoding='utf-8') as f:
        f.write(full_text)

    # Write the answer-only version
    with open(output_answer_only_file, 'w', encoding='utf-8') as f:
        f.write(answer_only_text)

    # Print statistics about the processed data
    print("📊 Data Statistics:")
    print(f"✅ Total Questions: {total_questions}")
    print(f"✅ Total Tokens (Full Version): {total_tokens_full}")
    print(f"✅ Total Tokens (Answer-Only Version): {total_tokens_answer_only}")

    # Check if the outputs fit within the model's context window
    if total_tokens_full > MAX_TOKENS:
        print(f"⚠️ Full version exceeds LLaMA3 maximum token limit ({MAX_TOKENS} tokens). Please reduce the size.")
    else:
        print("✅ Full version is within LLaMA3 token limit.")

    if total_tokens_answer_only > MAX_TOKENS:
        print(f"⚠️ Answer-only version exceeds LLaMA3 maximum token limit ({MAX_TOKENS} tokens). Please reduce the size.")
    else:
        print("✅ Answer-only version is within LLaMA3 token limit.")

    # Print file paths for easy reference
    print(f"📂 Full version saved to: {output_full_file}")
    print(f"📂 Answer-only version saved to: {output_answer_only_file}")

except Exception as e:
    print(f"❌ An error occurred: {e}")
