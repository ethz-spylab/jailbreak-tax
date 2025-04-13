import argparse
import json
import os
from pathlib import Path
import logging
import sys

# Add logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils.jb_tax_grading_utils import combine_grading_results, grade_results

def find_results_file(folder_path, file_pattern="results"):
    """Find results file in a folder that matches the pattern."""
    folder_path = Path(folder_path)
    
    # Look for files that start with the pattern and end with .json
    for file in folder_path.glob(f"*{file_pattern}*.json"):
        if file.is_file() and "grading" not in file.name and "combined" not in file.name:
            return str(file)
    
    # If no file found, check subdirectories
    for subdir in folder_path.iterdir():
        if subdir.is_dir():
            for file in subdir.glob(f"*{file_pattern}*.json"):
                if file.is_file() and "grading" not in file.name and "combined" not in file.name:
                    return str(file)
    
    raise FileNotFoundError(f"No results file found in {folder_path}")

def ensure_graded(results_path, dataset):
    """Ensure the results file has been graded, grade it if not."""
    # Check if grading_totals.json exists
    totals_path = Path(results_path).parent / "grading_totals.json"
    
    if not totals_path.exists():
        logging.info(f"Grading results file: {results_path}")
        grade_results(results_path, dataset)
    else:
        logging.info(f"Results already graded: {results_path}")

def combine_from_folders(unaligned_folder, before_jb_folder, after_jb_folder, output_folder, dataset):
    """
    Combine results from three folders representing different experiment stages.
    
    Args:
        unaligned_folder: Path to folder with unaligned results
        before_jb_folder: Path to folder with before-jailbreak results
        after_jb_folder: Path to folder with after-jailbreak results
        output_folder: Path to save combined results
        dataset: Dataset type (gsm8k, math, wmdp, etc.)
    """
    # Find results files in each folder
    unaligned_path = find_results_file(unaligned_folder)
    before_jb_path = find_results_file(before_jb_folder)
    after_jb_path = find_results_file(after_jb_folder)
    
    logging.info(f"Found result files:")
    logging.info(f"  Unaligned: {unaligned_path}")
    logging.info(f"  Before JB: {before_jb_path}")
    logging.info(f"  After JB: {after_jb_path}")
    
    # Ensure all results are graded
    ensure_graded(unaligned_path, dataset)
    ensure_graded(before_jb_path, dataset)
    ensure_graded(after_jb_path, dataset)
    
    # Create output path
    output_path = Path(output_folder)
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Combine results
    logging.info(f"Combining results to: {output_path}")
    combine_grading_results(
        unaligned_path=unaligned_path,
        before_jb_path=before_jb_path,
        after_jb_path=after_jb_path,
        output_dir=str(output_path)
    )
    
    logging.info(f"Combined results saved to: {output_path}")
    
    return str(output_path / "combined_statistics.json")

def main():
    # Define paths directly in the script
    # You can change these paths to your actual experiment folders
    UNALIGNED_FOLDER = "JB_Tax_Results/EvilMath/anthropic/claude-3.5-haiku/api/solve_the_problem/gsm8k-evil/benign_questions/No_Attack/example_run/0-188_questions"
    BEFORE_JB_FOLDER = "JB_Tax_Results/EvilMath/anthropic/claude-3.5-haiku/api/solve_the_problem/gsm8k-evil/evil_questions/No_Attack/example_run/0-188_questions"
    AFTER_JB_FOLDER = "JB_Tax_Results/EvilMath/anthropic/claude-3.5-haiku/api/solve_the_problem/gsm8k-evil/evil_questions/Pair/example_run_5/0-188_questions"
    OUTPUT_FOLDER = "JB_Tax_Results/EvilMath/anthropic/claude-3.5-haiku/api/solve_the_problem/gsm8k-evil/evil_questions/Pair/5_rounds"
    DATASET_TYPE = "gsm8k-evil"  # or "gsm8k", "math", etc.
    
    # Also support command-line arguments if provided
    parser = argparse.ArgumentParser(description="Combine results from three experiment folders")
    parser.add_argument("--unaligned", type=str, default=UNALIGNED_FOLDER, help="Path to unaligned results folder")
    parser.add_argument("--before_jb", type=str, default=BEFORE_JB_FOLDER, help="Path to before-jailbreak results folder")
    parser.add_argument("--after_jb", type=str, default=AFTER_JB_FOLDER, help="Path to after-jailbreak results folder")
    parser.add_argument("--output", type=str, default=OUTPUT_FOLDER, help="Path to output folder")
    parser.add_argument("--dataset", type=str, default=DATASET_TYPE, help="Dataset type (gsm8k, math, wmdp, etc.)")
    
    args = parser.parse_args()
    
    try:
        combined_stats_path = combine_from_folders(
            args.unaligned,
            args.before_jb, 
            args.after_jb, 
            args.output,
            args.dataset
        )
        
        # Print key statistics from combined results
        with open(combined_stats_path, 'r') as f:
            stats = json.load(f)
            
        logging.info("\nKey Statistics:")
        logging.info(f"Base Utility: {stats['base_util']}")
        logging.info(f"Jailbreak Success: {stats['jb_succ']}")
        logging.info(f"Jailbreak Utility: {stats['jb_util']}")
        logging.info(f"Jailbreak Tax: {stats['jb_tax']}")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())