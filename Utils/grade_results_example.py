import sys
from Utils.jb_tax_grading_utils import grade_results

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m Utils.grade_results_example <results_file_path> <dataset>")
        print("Example: python -m Utils.grade_results_example Data/result-formats/gsm8k_result_example.json gsm8k")
        sys.exit(1)
    
    results_file = sys.argv[1]
    dataset = sys.argv[2]
    
    grade_results(str(results_file), dataset=dataset)
    print(f"Grading completed for: {results_file}")
    print(f"Grading totals saved to: {results_file.rsplit('/', 1)[0]}/grading_totals.json")
