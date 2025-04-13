import torch
import random
from pathlib import Path
from Utils.llm_utils import prepare_env
from Utils.jb_tax_utils import (
    ModelConfig,
    DatasetConfig,
    AttackConfig,
    AlignmentConfig,
    WMDPExperiment,
    GSM8KExperiment,
    MATHExperiment
)
from Utils.jb_tax_grading_utils import grade_results, combine_grading_results
from Utils.argument_parser import parse_args

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
prepare_env()

def get_experiment_class(dataset_name):
    """Return appropriate experiment class based on dataset name."""
    experiment_classes = {
        "wmdp-bio": WMDPExperiment,
        "gsm8k": GSM8KExperiment,
        "gsm8k-evil": GSM8KExperiment,
        "math": MATHExperiment
    }
    return experiment_classes.get(dataset_name.lower())

def setup_stage_directory(experiment, stage_name):
    """Set up stage-specific directory and return the original directory path.
    
    Args:
        experiment: Experiment instance
        stage_name: Name of the current stage
    
    Returns:
        Path: Original directory path if custom_save_dir was set, None otherwise
    """
    if experiment.config.custom_save_dir:
        original_dir = Path(experiment.config.custom_save_dir)
        stage_dir = original_dir / stage_name.lower().replace(" ", "_")
        experiment.config.custom_save_dir = str(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        return original_dir
    return None

def get_output_directory(experiment, run_id):
    if experiment.config.custom_save_dir:
        return experiment.config.custom_save_dir
    else:
        model_name = experiment.config.select_model.split('/')[-1]
        alignment_method = "no_pseudo_alignment" if experiment.config.alignment_method == "no_alignment" else f"{experiment.config.alignment_method}_alignment"
        attack_type = experiment.get_standardized_attack_name()
        return f"{experiment._get_base_output_dir()}/{model_name}/{run_id}/Combined/{alignment_method}/{attack_type}/{experiment.config.question_range}_questions"

def run_experiment_stage(experiment, apply_alignment, apply_attack, stage_name, save_formatted_prompts=True):
    """Run a single stage of the experiment."""
    print(f"\nRunning {stage_name} stage...")
    
    # Set up stage directory and store original path
    original_dir = setup_stage_directory(experiment, stage_name)
    
    experiment.set_apply_alignment(apply_alignment)
    experiment.set_apply_attack(apply_attack)
    if save_formatted_prompts:
        experiment.save_formatted_prompts()
    experiment.setup_logging()
    results, results_file = experiment.run()
    
    print(f"Grading results for {stage_name}...")
    grade_results(str(results_file), dataset=experiment.config.dataset)
    
    # Restore original custom_save_dir
    if original_dir:
        experiment.config.custom_save_dir = str(original_dir)
    
    return results_file

def main():
    args = parse_args()
    
    # Get appropriate experiment class
    ExperimentClass = get_experiment_class(args.dataset)
    if not ExperimentClass:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Create experiment instance from args
    experiment = ExperimentClass(args=args)

    results_files = {}

    if args.stage == "all" or args.stage == "unaligned":
        # Stage 1: No alignment, no attack
        results_files["unaligned"] = run_experiment_stage(
            experiment=experiment, 
            apply_alignment=False, 
            apply_attack=False, 
            stage_name="UNALIGNED",
            save_formatted_prompts=True
        )

    if args.stage == "all" or args.stage == "before_jb":
        # Stage 2: With alignment, no attack
        results_files["before_jb"] = run_experiment_stage(
            experiment=experiment, 
            apply_alignment=True, 
            apply_attack=False, 
            stage_name="BEFORE JAILBREAK",
            save_formatted_prompts=True
        )

    if args.stage == "all" or args.stage == "after_jb":
        # Stage 3: With alignment and attack
        results_files["after_jb"] = run_experiment_stage(
            experiment=experiment, 
            apply_alignment=True, 
            apply_attack=True, 
            stage_name="AFTER JAILBREAK",
            save_formatted_prompts=True
        )

    # Combine all results if all stages were run
    if args.stage == "all":
        print("\nCombining results from all stages...")
        output_dir = str(get_output_directory(experiment, args.run))
        combine_grading_results(
            unaligned_path=str(results_files["unaligned"]),
            before_jb_path=str(results_files["before_jb"]),
            after_jb_path=str(results_files["after_jb"]),
            output_dir=output_dir
        )
        print(f'Combined results saved to: {output_dir}')

    # Cleanup
    experiment.cleanup()
    print(f"\n{'All stages' if args.stage == 'all' else args.stage.upper()} completed successfully!")

if __name__ == "__main__":
    main() 