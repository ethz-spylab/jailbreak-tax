import json
import os
import random
from transformers import AutoTokenizer
from collections import defaultdict

# Constants
TOTAL_QUESTIONS = 210  # Set your desired total number of questions here
NUM_LEVELS = 5  # Number of levels per subject
NUM_SUBJECTS = 7  # Total number of subjects in MATH dataset
MAX_QUESTIONS_PER_SUBJECT_PER_LEVEL = TOTAL_QUESTIONS // (NUM_SUBJECTS * NUM_LEVELS)  # Automatically calculated
REMOVE_REPEATED_INTROS = False

def load_math_problems(data_path, split="train"):
    """Load problems from the MATH dataset organized by subject and level."""
    split_path = os.path.join(data_path, split)
    problems_by_category = defaultdict(lambda: defaultdict(list))
    
    # Walk through all subject directories
    for subject in sorted(os.listdir(split_path)):
        subject_path = os.path.join(split_path, subject)
        if os.path.isdir(subject_path):
            # Sort files to ensure consistent ordering
            for problem_file in sorted(os.listdir(subject_path)):
                if problem_file.endswith('.json'):
                    with open(os.path.join(subject_path, problem_file), 'r') as f:
                        problem = json.load(f)
                        level = problem.get('level', 'Unknown')
                        problem['subject'] = subject
                        problems_by_category[subject][level].append(problem)
    
    return problems_by_category

def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    MAX_TOKENS = tokenizer.model_max_length
    print(f"📏 Model's maximum context length: {MAX_TOKENS} tokens")

    # Output file paths
    output_dir = "Data/many-shot/data"
    os.makedirs(output_dir, exist_ok=True)
    output_full_file = os.path.join(output_dir, f"many_shot_full_math_{TOTAL_QUESTIONS}{'_no_intro' if REMOVE_REPEATED_INTROS else ''}.json")
    output_txt_file = os.path.join(output_dir, f"many_shot_full_math_{TOTAL_QUESTIONS}{'_no_intro' if REMOVE_REPEATED_INTROS else ''}.txt")
    output_answer_only_file = os.path.join(output_dir, f"many_shot_answer_only_math_{TOTAL_QUESTIONS}{'_no_intro' if REMOVE_REPEATED_INTROS else ''}.txt")
    
    try:
        # Load MATH dataset
        print("Loading MATH dataset...")
        problems_by_category = load_math_problems("/data/datasets/MATH_BACKUP", "train")
        print("Dataset loaded successfully.")

        # Select equal number of problems from each subject and level
        selected_problems = []
        for subject, level_problems in problems_by_category.items():
            print(f"\nProcessing subject: {subject}")
            for level, problems in level_problems.items():
                num_available = len(problems)
                num_to_select = min(MAX_QUESTIONS_PER_SUBJECT_PER_LEVEL, num_available)
                selected = random.sample(problems, num_to_select)
                
                print(f"Selected {num_to_select} problems from {level}")
                selected_problems.extend(selected)

        # Shuffle the selected problems
        random.shuffle(selected_problems)

        # Prepare the final data structure
        final_data = {
            "metadata": {
                "total_problems": len(selected_problems),
                "questions_per_subject": MAX_QUESTIONS_PER_SUBJECT_PER_LEVEL,
                "remove_repeated_intros": REMOVE_REPEATED_INTROS
            },
            "problems": []
        }

        # Process each problem
        total_tokens_full = 0
        total_tokens_answer_only = 0
        intro_line = "The following are math problems (with answers).\n\n"
        
        # Count tokens for the introduction
        intro_tokens = len(tokenizer.encode(intro_line))
        total_tokens_full += intro_tokens
        total_tokens_answer_only += intro_tokens

        for problem in selected_problems:
            formatted_problem = {
                "subject": problem["subject"],
                "level": problem["level"],
                "question": problem["problem"],
                "solution": problem["solution"]
            }
            
            # Count tokens for full version
            problem_text = f"{problem['problem']}\nSolution:\n{problem['solution']}\n\n"
            full_tokens = len(tokenizer.encode(problem_text))
            total_tokens_full += full_tokens
            
            # Count tokens for answer-only version
            # Extract just the final answer from the solution
            solution_lines = problem["solution"].strip().split('\n')
            final_answer = solution_lines[-1] if solution_lines else problem["solution"]
            
            # If the final answer starts with "####", extract just that part
            if "####" in final_answer:
                final_answer = final_answer.split("####", 1)[1].strip()
                
            answer_only_text = f"{problem['problem']}\nAnswer: {final_answer}\n\n"
            answer_only_tokens = len(tokenizer.encode(answer_only_text))
            total_tokens_answer_only += answer_only_tokens
            
            final_data["problems"].append(formatted_problem)

        # Save the output files
        # JSON format
        with open(output_full_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2)
            
        # TXT format - full version
        with open(output_txt_file, 'w', encoding='utf-8') as f:
            f.write(intro_line)
            for problem in final_data["problems"]:
                f.write(f"{problem['question']}\n")
                f.write(f"Solution:\n{problem['solution']}\n\n")
        
        # TXT format - answer-only version
        with open(output_answer_only_file, 'w', encoding='utf-8') as f:
            f.write(intro_line)
            for problem in final_data["problems"]:
                f.write(f"{problem['question']}\n")
                solution_lines = problem["solution"].strip().split('\n')
                final_answer = solution_lines[-1] if solution_lines else problem["solution"]
                if "####" in final_answer:
                    final_answer = final_answer.split("####", 1)[1].strip()
                f.write(f"Answer: {final_answer}\n\n")

        # Print statistics
        print("\n📊 Data Statistics:")
        print(f"✅ Total Problems: {len(selected_problems)}")
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

        print(f"📂 Full version (JSON) saved to: {output_full_file}")
        print(f"📂 Full version (TXT) saved to: {output_txt_file}")
        print(f"📂 Answer-only version saved to: {output_answer_only_file}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
