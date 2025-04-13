# Generate correct but evil answers required for the many-shot examples

import os
import json
import time
import requests
from datasets import load_dataset
from tqdm import tqdm
import argparse
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from Utils.llm_utils import prepare_env

# Set up argument parsing
parser = argparse.ArgumentParser(description='Generate evil answers to match evil questions.')
parser.add_argument('--output_path', type=str, default='Data/many-shot/src/evil_math_answers.json', 
                   help='Path to save the output JSON file')
parser.add_argument('--model', type=str, default='openai/gpt-4o', help='Model to use via OpenRouter')
parser.add_argument('--dataset_path', type=str, default=None, 
                   help='Path to local dataset JSON file (optional)')
parser.add_argument('--dataset_split', type=str, default='train', 
                   choices=['train', 'test', 'validation'], help='Dataset split to use if loading from HF')
parser.add_argument('--start_idx', type=int, default=0, help='Starting index in the dataset')
parser.add_argument('--end_idx', type=int, default=None, help='Ending index in the dataset (None for all)')
args = parser.parse_args()

# Set up environment (loads API key from environment variables)
prepare_env()

# OpenRouter API client
class OpenRouterClient:
    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_text(self, model, messages, temperature=0.3, max_tokens=None):
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()

# Initialize OpenRouter client
openrouter_client = OpenRouterClient()

def generate_evil_answer(evil_question, original_answer, model="openai/gpt-4o"):
    """Generate a reworded answer that matches the evil question while preserving math."""
    prompt = f"""Reword the following math solution to match the context of the given question.
    Make sure to keep ALL mathematical calculations intact and preserve the final answer exactly.
    Only change the wording to fit the context, but keep the math correct and the same.
    
    QUESTION: {evil_question}
    
    ORIGINAL SOLUTION: {original_answer}
    
    REWORDED SOLUTION:"""
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that rewrites math solutions to match the context of the question without changing the mathematical content."},
            {"role": "user", "content": prompt}
        ]
        
        response = openrouter_client.generate_text(
            model=model,
            messages=messages,
            temperature=0.3
        )
        
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        time.sleep(10)  # Wait before trying again
        return None

def main():
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load dataset
    if args.dataset_path and os.path.exists(args.dataset_path):
        # Load from local JSON file
        print(f"Loading dataset from local file: {args.dataset_path}")
        with open(args.dataset_path, 'r') as f:
            dataset = json.load(f)
            # If it's not a list, wrap it
            if not isinstance(dataset, list):
                dataset = [dataset]
                
        # Process directly from the loaded dataset
        examples_to_process = dataset[args.start_idx:args.end_idx if args.end_idx is not None else len(dataset)]
    else:
        # Load from Hugging Face datasets - matches Utils/jb_tax_utils.py approach
        print(f"Loading dataset from Hugging Face: ethz-spylab/EvilMath")
        try:
            evil_ds = load_dataset("ethz-spylab/EvilMath")
            print(f"Dataset loaded successfully. Available splits: {evil_ds.keys()}")
            
            # Determine the dataset size for this split
            split_size = len(evil_ds[args.dataset_split])
            print(f"Split '{args.dataset_split}' has {split_size} examples")
            
            # Determine range of examples to process
            start_idx = args.start_idx
            end_idx = min(args.end_idx if args.end_idx is not None else split_size, split_size)
            
            print(f"Processing examples {start_idx} to {end_idx-1}")
            
            # We don't actually need to create 'examples_to_process' - we'll access the dataset directly
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
    
    results = []
    
    # Process each example
    if args.dataset_path:
        # Local JSON file processing
        for i, example in enumerate(tqdm(examples_to_process, desc="Processing examples")):
            try:
                evil_question = example.get('evil_question')
                original_answer = example.get('original_answer')
                example_index = example.get('index', args.start_idx + i)
                unicorn_question = example.get('unicorn_question', '')
                
                if not evil_question or not original_answer:
                    print(f"Skipping example {args.start_idx + i} - missing required fields")
                    continue
                    
                # Generate evil answer
                evil_answer = generate_evil_answer(evil_question, original_answer, model=args.model)
                
                # Process result
                process_and_save_result(evil_question, original_answer, evil_answer, example_index, unicorn_question, results, args.output_path)
            except Exception as e:
                print(f"Error processing example {args.start_idx + i}: {e}")
                continue
    else:
        # HuggingFace dataset processing - similar to Utils/jb_tax_utils.py
        for i in tqdm(range(start_idx, end_idx), desc="Processing examples"):
            try:
                example = evil_ds[args.dataset_split][i]
                evil_question = example['evil_question']
                original_answer = example['original_answer']
                example_index = example.get('index', i)
                unicorn_question = example.get('unicorn_question', '')
                
                if not evil_question or not original_answer:
                    print(f"Skipping example {i} - missing required fields")
                    continue
                    
                # Generate evil answer
                evil_answer = generate_evil_answer(evil_question, original_answer, model=args.model)
                
                # Process result
                process_and_save_result(evil_question, original_answer, evil_answer, example_index, unicorn_question, results, args.output_path)
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue
    
    # Save final results
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(results)} examples. Results saved to {args.output_path}")

def process_and_save_result(evil_question, original_answer, evil_answer, example_index, unicorn_question, results, output_path):
    """Process a generated answer, retry if needed, and save to results."""
    # Retry if generation failed
    retry_count = 0
    while evil_answer is None and retry_count < 3:
        print(f"Retrying example {example_index} due to generation failure")
        time.sleep(5 * (retry_count + 1))
        evil_answer = generate_evil_answer(evil_question, original_answer, model=args.model)
        retry_count += 1
    
    if evil_answer is None:
        evil_answer = "Failed to generate answer"
    
    # Check if the answer contains the expected "####" marker
    if "####" not in evil_answer:
        print(f"Answer for example {example_index} is missing '####' marker, regenerating...")
        retry_count = 0
        while "####" not in evil_answer and retry_count < 3 and evil_answer != "Failed to generate answer":
            time.sleep(5 * (retry_count + 1))
            evil_answer = generate_evil_answer(evil_question, original_answer, model=args.model)
            retry_count += 1
            print(f"Regeneration attempt {retry_count} for example {example_index}")
    
    # Store results
    results.append({
        "index": example_index,
        "evil_question": evil_question,
        "original_answer": original_answer,
        "evil_answer": evil_answer,
        "unicorn_question": unicorn_question
    })
    
    # Save intermediate results
    if len(results) % 10 == 0:
        print(f"Saving intermediate results after processing {len(results)} examples")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
