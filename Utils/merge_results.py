import os
import json
from pathlib import Path
from typing import Dict, List, Union

def combine_grading_results(base_path: str, output_path: str, ignore_folders: List[str] = None) -> None:
    """
    Combine results from multiple question folders including all related files
    with consistent naming patterns.
    
    Args:
        base_path: Path to the base directory containing question folders
        output_path: Path where to save combined results
        ignore_folders: List of folder names to ignore (e.g., ["450-499_questions"])
    """
    if ignore_folders is None:
        ignore_folders = []
    
    # Initialize combined totals
    combined_totals = {
        "correct": 0,
        "incorrect": 0,
        "refusal": 0
    }
    
    # Initialize combined question IDs
    combined_question_ids = {
        "correct": [],
        "incorrect": [],
        "refusal": []
    }
    
    # Initialize combined results and prompts lists
    combined_results = []
    combined_prompts = []
    
    # Initialize question counter
    total_questions_processed = 0
    
    # Find all question folders, excluding ignored ones
    question_folders = [
        d for d in Path(base_path).rglob('*_questions') 
        if d.is_dir() and d.name not in ignore_folders
    ]
    
    print(f"\nFound {len(question_folders)} folders to process")
    print("Ignored folders:", ignore_folders if ignore_folders else "None")
    
    # Get base filenames from first folder to maintain consistency
    base_filenames = {}
    if question_folders:
        print(f"\nLooking for files in first folder: {question_folders[0]}")
        for file in question_folders[0].glob('*.json'):
            print(f"Found file: {file.name}")
            if file.name == 'grading_totals.json':
                base_filenames['grading_totals'] = file.name
            elif file.name.startswith('prompts_'):
                base_filenames['prompts'] = file.name
            elif file.name.startswith('results_'):
                base_filenames['results'] = file.name
    
    # Process each folder
    for folder in question_folders:
        print(f"\nProcessing folder: {folder}")
        
        # Process grading totals file
        if 'grading_totals' in base_filenames:
            grading_file = folder / base_filenames['grading_totals']
            if grading_file.exists():
                with open(grading_file, 'r') as f:
                    try:
                        grading_data = json.load(f)
                        # Add totals
                        for category in combined_totals:
                            combined_totals[category] += grading_data['totals'][category]
                        # Add question IDs
                        for category in combined_question_ids:
                            combined_question_ids[category].extend(grading_data['question_ids'][category])
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse {grading_file}")
        
        # Process results file
        if 'results' in base_filenames:
            results_file = folder / base_filenames['results']
            if results_file.exists():
                with open(results_file, 'r') as f:
                    try:
                        results_data = json.load(f)
                        if isinstance(results_data, list):
                            combined_results.extend(results_data)
                            total_questions_processed += len(results_data)
                        else:
                            combined_results.append(results_data)
                            total_questions_processed += 1
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse {results_file}")
        
        # Process prompts file
        if 'prompts' in base_filenames:
            prompts_file = folder / base_filenames['prompts']
            if prompts_file.exists():
                with open(prompts_file, 'r') as f:
                    try:
                        prompts_data = json.load(f)
                        if isinstance(prompts_data, list):
                            combined_prompts.extend(prompts_data)
                        else:
                            combined_prompts.append(prompts_data)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse {prompts_file}")
    
    # Sort combined results and prompts by question_id
    combined_results.sort(key=lambda x: x.get('question_id', 0))
    combined_prompts.sort(key=lambda x: x.get('question_id', 0))
    
    # Sort question IDs and remove duplicates
    for category in combined_question_ids:
        combined_question_ids[category] = sorted(list(set(combined_question_ids[category])))
    
    # Calculate new percentages
    total_questions = sum(combined_totals.values())
    percentages = {
        category: round((count / total_questions * 100), 3) if total_questions > 0 else 0
        for category, count in combined_totals.items()
    }
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save combined grading totals
    grading_output = {
        "totals": combined_totals,
        "percentages": percentages,
        "question_ids": combined_question_ids
    }
    
    # Save all files with their original names
    if 'grading_totals' in base_filenames:
        with open(output_dir / base_filenames['grading_totals'], 'w') as f:
            json.dump(grading_output, f, indent=2)
    
    if 'results' in base_filenames:
        with open(output_dir / base_filenames['results'], 'w') as f:
            json.dump(combined_results, f, indent=2)
    
    if 'prompts' in base_filenames:
        with open(output_dir / base_filenames['prompts'], 'w') as f:
            json.dump(combined_prompts, f, indent=2)
    
    print(f"\nTotal questions processed: {total_questions_processed}")
    print(f"Questions by category:")
    for category, count in combined_totals.items():
        print(f"  {category}: {count} ({percentages[category]:.3f}%)")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Combine results from multiple question folders")
    parser.add_argument("--base_path", required=True, help="Base path containing the question folders")
    parser.add_argument("--total_questions", type=int, required=True, help="Total number of questions")
    parser.add_argument("--ignore", nargs="+", help="List of folder names to ignore", default=[])
    
    args = parser.parse_args()
    
    # Generate output folder name
    output_folder = f"0-{args.total_questions}_questions"
    
    # Construct output path - keep it at the same level as base_path
    output_path = os.path.join(args.base_path, output_folder)
    
    # Combine results
    combine_grading_results(args.base_path, output_path, ignore_folders=args.ignore)
    
    print(f"Results combined successfully and saved to: {output_path}")

if __name__ == "__main__":
    main()
