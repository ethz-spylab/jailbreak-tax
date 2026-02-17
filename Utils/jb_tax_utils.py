from dataclasses import dataclass, fields, field
from datasets import load_dataset
from typing import Dict, Any, Optional, Callable
import torch
from transformers import logging as transformers_logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import logging
import shutil
import json
import gc
from abc import ABC, abstractmethod
from peft import PeftModel
from Utils.jailbreaks import generate_model_answer, generate_jailbreak_response, setup_attack, get_model_prompt, query_model, query_model_api
import logging
import shutil
from Utils.argument_parser import parse_args, get_model_choices
import gc
import json
from pathlib import Path
import re
import os
import random
from tqdm import tqdm

@dataclass
class ModelConfig:
    """Configuration for model loading and generation parameters."""
    
    select_model: str = "meta-llama/llama-3.1-8b-instruct"
    """Name or path of the model to load (e.g., 'meta-llama/llama-3.1-8b-instruct')"""
    
    api_or_local: str = "local"
    """Whether to use API ('api') or local ('local') model. Default: 'local'"""
    
    model_kwargs: Dict[str, Any] | None = None
    """Additional kwargs for model loading"""
    
    model_provider: str = "openrouter"
    """API provider name (e.g., 'openrouter')"""
    
    top_p: float = 0.9
    """Top-p sampling parameter for generation"""
    
    temperature: float = 1.0
    """Temperature for generation sampling"""
    
    deterministic: bool = False
    """Whether to use deterministic generation"""
    
    analyze_probabilities: bool = False
    """Whether to analyze token probabilities"""
    
    run: str = "1"
    """Unique identifier for this run"""
    
    force_overwrite: bool = False
    """Whether to overwrite existing results"""

    custom_save_dir: str | None = None
    """Custom save directory for results"""

@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    
    dataset: str | None = None
    """Name of the dataset to use. Options: 'wmdp-bio', 'gsm8k', 'math', 'gsm8k-evil'"""
    
    question_range: str | None = None
    """Range of questions to use (e.g., '0-10')"""
    
    shuffle_choices: bool | None = None
    """Whether to shuffle multiple choice answers for WMDP dataset"""
    
    format_type: str | None = None
    """How to format the prompts"""
    
    judging_format: str | None = None
    """How to format judging criteria"""
    
    evil_dataset_path: str | None = None
    """Path to adversarial dataset (optional, use HuggingFace by default)"""
    
    use_local_dataset: bool = False
    """Whether to load EvilMath from local file instead of HuggingFace"""
    
    MATH_dataset_path: str | None = None
    """Path to MATH dataset"""
    
    math_level: str | None = None
    """Difficulty level for MATH problems. Options: 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'"""

@dataclass
class AttackConfig:
    """Configuration for jailbreak attacks."""
    
    # General attack parameters
    attack_type: str | None = None
    """Type of attack to use (e.g., 'PAIR', 'TAP', 'AutoDAN', 'gcg', 'finetune', 'many-shot', 'simple_jb', 'translate')"""
    
    target_string: str | None = None
    """Target string for attacks. Relevant for TAP, PAIR, AutoDAN, and gcg attacks."""

    # Custom jailbreak parameters
    custom_jailbreak: bool = False
    """Whether to use custom jailbreak function"""
    
    custom_jailbreak_fn: Callable | None = None
    """Custom jailbreak function to use"""

    # Finetuned attack parameters
    peft_model_path: str | None = None
    """Path to LoRA weights for finetuned attack model"""
    
    target_language: str | None = None
    """Target language for MultiJail attack"""


    # PAIR/TAP parameters
    max_rounds: int = 20
    """Maximum rounds for PAIR/TAP attacks"""
    
    width: int = 5
    """Width parameter for TAP"""
    
    branching_factor: int = 3
    """Branching factor for TAP"""
    
    depth: int = 5
    """Depth for TAP search"""
    
    keep_last_n: int = 3
    """Number of best candidates to keep for TAP"""
    
    dont_modify_question: bool = False
    """Not modify original question content. When True, we get PAIR (don't modify) attack"""

    # AutoDAN parameters
    autodan_num_steps: int = 50
    """Number of steps for AutoDAN"""
    
    autodan_batch_size: int = 8
    """Batch size for AutoDAN"""
    
    autodan_elite_ratio: float = 0.05
    """Elite ratio for AutoDAN"""
    
    autodan_crossover: float = 0.5
    """Crossover rate for AutoDAN"""
    
    autodan_mutation: float = 0.01
    """Mutation rate for AutoDAN"""
    
    autodan_num_points: int = 5
    """Number of points for AutoDAN"""
    
    autodan_iteration_interval: int = 5
    """Iteration interval for AutoDAN"""

    # GCG parameters
    gcg_init_str_len: int = 20
    """Initial string length for GCG"""
    
    gcg_num_steps: int = 250
    """Number of steps for GCG"""
    
    gcg_search_width: int = 128
    """Search width for GCG"""
    
    gcg_topk: int = 128
    """Top-k parameter for GCG"""
    
    gcg_seed: int = 42
    """Random seed for GCG"""
    
    gcg_early_stop: bool = False
    """Whether to enable early stopping for GCG"""

    # Many-shot attack parameters
    many_shot_file: str = "many-shot/data/many_shot_full_bio_2_no_intro.txt"
    """Path to many-shot examples file"""
    
    many_shot_num_questions: int = 100
    """Number of many-shot examples to use"""

@dataclass
class AlignmentConfig:
    """Configuration for model alignment methods."""
    
    alignment_method: str | None = None
    """Method of alignment to use (options: 'system_prompt', 'fine_tuned', 'no_alignment') Default: None"""
    
    system_prompt_type: str | None = None
    """Type of system prompt to use"""
    
    alignment_model_path: str | None = None
    """Path to LoRA weights for finetuned model used as alignment model."""

@dataclass
class UnifiedConfig:
    """
    Unified configuration that flattens all modular configs into a single structure
    for compatibility with existing code that expects args-style access.
    Inherits fields from modular configs.
    """
    def __init__(self, **kwargs):
        # Get all fields from modular configs
        model_fields = {f.name: f.default for f in fields(ModelConfig)}
        dataset_fields = {f.name: f.default for f in fields(DatasetConfig)}
        attack_fields = {f.name: f.default for f in fields(AttackConfig)}
        alignment_fields = {f.name: f.default for f in fields(AlignmentConfig)}

        # Combine all fields
        all_fields = {
            **model_fields,
            **dataset_fields,
            **attack_fields,
            **alignment_fields,
        }

        # Update with provided values
        all_fields.update(kwargs)

        # Set all attributes
        for key, value in all_fields.items():
            setattr(self, key, value)

    @classmethod
    def from_modular_configs(cls, 
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        attack_config: AttackConfig,
        alignment_config: AlignmentConfig
    ):
        """Create unified config from modular configs"""
        return cls(
            **model_config.__dict__,
            **dataset_config.__dict__,
            **attack_config.__dict__,
            **alignment_config.__dict__,
        )

    @classmethod
    def from_args(cls, args):
        """Create unified config from legacy args object"""
        return cls(**{k: v for k, v in vars(args).items() if v is not None})

class BaseExperiment(ABC):
    """Abstract base class for experiment implementations."""
    
    def __init__(
        self,
        model_config: ModelConfig | None = None,
        dataset_config: DatasetConfig | None = None,
        attack_config: AttackConfig | None = None,
        alignment_config: AlignmentConfig | None = None,
        args = None,
        apply_alignment: bool = True,
        apply_attack: bool = True,
    ):
        if args is None and model_config is None:
            raise ValueError("model_config or args must be provided")

        if args is not None:
            # Create unified config directly from args
            self.config = UnifiedConfig.from_args(args)
        else:
            # Get default configs
            default_dataset_config = self.get_default_dataset_config()
            default_alignment_config = self.get_default_alignment_config()

            # Update default configs with provided configs if any
            if dataset_config:
                for field in fields(dataset_config):
                    value = getattr(dataset_config, field.name)
                    if value is not None:  # Only update if value is provided
                        setattr(default_dataset_config, field.name, value)
                    
            if alignment_config:
                for field in fields(alignment_config):
                    value = getattr(alignment_config, field.name)
                    if value is not None:  # Only update if value is provided
                        setattr(default_alignment_config, field.name, value)
            
            self.config = self._create_unified_config(
                model_config=model_config,
                dataset_config=default_dataset_config,
                attack_config=attack_config,
                alignment_config=default_alignment_config
            )

        self.apply_alignment = apply_alignment
        self.apply_attack = apply_attack
        
        self.setup_output_dir()
        self.setup_logging()

        if self.config.custom_jailbreak and self.config.custom_jailbreak_fn:
            print(f"Using custom jailbreak function: {self.config.custom_jailbreak_fn.__name__}")

    @abstractmethod
    def _create_unified_config(self, model_config, dataset_config, attack_config, alignment_config):
        """Create unified configuration from modular configs."""
        pass

    def setup_output_dir(self) -> None:
        """Build and check output directory path."""
        # Build output directory path
        self.output_dir = self._get_output_path()

        # Create directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_logging(self) -> None:
        """Configure logging to write to a file in the output directory."""
        # Remove all handlers associated with the root logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        log_file = self.output_dir / 'experiment.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w')  # 'w' specifies overwrite mode
            ]
        )
        
        logging.info("Starting new experiment run\n")

    def setup_model(self):
        """Setup model based on configuration."""
        # Skip loading model here if using finetune attack
        if self.apply_attack and self.config.attack_type == 'finetune':
            logging.info("Skipping setup_model(), the finetune attack will load its model")
            return None, None

        if self.config.api_or_local == "local":
            if self.apply_alignment and self.config.alignment_method == 'fine_tuned':
                logging.info(f"Alignment method is fine_tuned, loading aligned model from {self.config.alignment_model_path}")
                # Load base model first
                # Use bfloat16 for Nemotron models, float16 for others
                if "nemotron" in self.config.select_model.lower():
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float16
                    
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.config.select_model,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    trust_remote_code=True)
            
                # Load and merge LoRA weights
                model = PeftModel.from_pretrained(
                    base_model, 
                    self.config.alignment_model_path,
                )
                # Merge LoRA weights with base model
                model = model.merge_and_unload()
                del base_model
                gc.collect()
            else:
                logging.info(f"Loading model from {self.config.select_model}")
                # Standard model loading
                # Use bfloat16 for Nemotron models, float16 for others
                if "nemotron" in self.config.select_model.lower():
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float16
                    
                # Handle subfolder paths in model names
                if "/" in self.config.select_model and self.config.select_model.count("/") > 1:
                    # Split the path to get repo_id and subfolder
                    parts = self.config.select_model.split("/")
                    repo_id = "/".join(parts[:2])  # e.g., "kotekjedi/qwq3-32b-lora-jailbreak-detection-merged"
                    subfolder = "/".join(parts[2:])  # e.g., "qwq3-32b-lora-jailbreak-detection"
                    model = AutoModelForCausalLM.from_pretrained(
                        repo_id,
                        subfolder=subfolder,
                        torch_dtype=torch_dtype,
                        device_map="auto",
                        trust_remote_code=True
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.config.select_model,
                        torch_dtype=torch_dtype,
                        device_map="auto",
                        trust_remote_code=True
                    )
            
            # Handle subfolder paths for tokenizer as well
            if "/" in self.config.select_model and self.config.select_model.count("/") > 1:
                # Split the path to get repo_id and subfolder
                parts = self.config.select_model.split("/")
                repo_id = "/".join(parts[:2])  # e.g., "kotekjedi/qwq3-32b-lora-jailbreak-detection-merged"
                subfolder = "/".join(parts[2:])  # e.g., "qwq3-32b-lora-jailbreak-detection"
                tokenizer = AutoTokenizer.from_pretrained(
                    repo_id, 
                    subfolder=subfolder, 
                    trust_remote_code=True
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.config.select_model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id
            return model, tokenizer
        
        return self.config.select_model, None

    @abstractmethod
    def _load_dataset(self):
        """Load and prepare dataset."""
        pass

    @abstractmethod
    def get_formatted_prompts(self, *args, **kwargs):
        """Get formatted prompts for the specific dataset type."""
        pass

    @abstractmethod
    def _run_with_no_attack(self, formatted_data, model, tokenizer):
        """Run normal (non-attack) inference."""
        pass

    @abstractmethod
    def _run_with_attack(self, formatted_data, model, tokenizer, attack_config):
        """Run attack inference."""
        pass

    @abstractmethod
    def _format_result(self, question_num, question, choices, answer, response, info=None, **kwargs):
        """Format results specific to the dataset type.
        
        Args:
            question_num: Question number/ID
            question: Question text
            choices: List of choices (if applicable)
            answer: Correct answer
            response: Model response
            info: Additional info about the response (optional)
            **kwargs: Additional dataset-specific arguments
        """
        pass

    def _get_results_filename(self) -> str:
        """Get the filename for results based on configuration."""
        # Determine alignment component
        alignment_method = "no_alignment"
        if self.apply_alignment and self.config.alignment_method:
            alignment_method = self.config.alignment_method

        # Determine attack component using the new standardized method
        attack_component = f"_{self.get_standardized_attack_name()}"
        
        filename = f"results_{self.config.dataset}_{alignment_method}{attack_component}.json"
        return filename

    def _save_results(self, results):
        """Save results to a JSON file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = self._get_results_filename()
        with open(self.output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2)

    def set_apply_alignment(self, apply: bool):
        """Set whether to apply alignment and update output directory."""
        self.apply_alignment = apply
        self.update_output_dir()

    def set_apply_attack(self, apply: bool):
        """Set whether to apply attack and update output directory."""
        self.apply_attack = apply
        self.update_output_dir()

    def update_output_dir(self):
        """Update the output directory based on current settings."""
        # Rebuild the output directory path based on current settings
        self.output_dir = self._get_output_path()
        
        # Ensure the directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_output_path(self) -> Path:
        """Construct the output directory path based on current settings."""
        # Add validation for required config values
        if not self.config.select_model:
            raise ValueError("select_model must be specified in config")
        if not self.config.api_or_local:
            raise ValueError("api_or_local must be specified in config")
        if not self.config.format_type:
            raise ValueError("format_type must be specified in config")
        if not self.config.dataset:
            raise ValueError("dataset must be specified in config")
        if not self.config.alignment_method and self.apply_alignment:
            raise ValueError("alignment_method must be specified in config")
        if not self.config.attack_type and self.apply_attack:
            raise ValueError("attack_type must be specified in config")
        if not self.config.run:
            raise ValueError("run must be specified in config")
        if not self.config.question_range:
            raise ValueError("question_range must be specified in config")

        # If custom save directory is specified, use it directly
        if hasattr(self.config, 'custom_save_dir') and self.config.custom_save_dir:
            return Path(self.config.custom_save_dir)

        # Otherwise use the directory structure: model/run_id/stage
        # Get model name (last part of path)
        model_name = self.config.select_model.split('/')[-1]
        
        # Determine stage component
        if self.config.dataset == "gsm8k-evil":
            # Special handling for gsm8k-evil dataset
            if self.apply_alignment:
                if self.apply_attack:
                    attack_name = self.get_standardized_attack_name()
                    stage = f"evil_questions_and_attack/{attack_name}"
                else:
                    stage = "evil_questions_no_attack"
            else:
                stage = "benign_questions"
        else:
            # For all other datasets
            if self.apply_alignment and self.apply_attack:
                # Include attack type in after_jailbreak stage
                alignment_method = self.config.alignment_method
                attack_name = self.get_standardized_attack_name()
                stage = f"after_jailbreak/{alignment_method}_alignment/{attack_name}"
            elif self.apply_alignment:
                # Include alignment method in before_jailbreak stage
                alignment_method = self.config.alignment_method
                stage = f"before_jailbreak/{alignment_method}_alignment"
            else:
                stage = "unaligned"

        # Build the path with new structure
        output_path = self._get_base_output_dir() / \
            model_name / \
            self.config.run / \
            stage / \
            f"{self.config.question_range}_questions"
            
        return output_path

    def _check_and_confirm_overwrite(self, path: Path) -> bool:
        """Check if a path exists and ask for overwrite permission if needed."""
        if path.exists():
            if self.config.force_overwrite:
                return True
            user_input = input(f"\nWarning: {path} already exists. Overwrite? (y/n): ")
            if user_input.lower() != 'y':
                print("Aborting to prevent overwrite.")
                return False
        return True

    def run(self):
        """Run the experiment."""

        transformers_logging.set_verbosity_error()

        # Check if results file already exists
        results_file = self.output_dir / self._get_results_filename()
        if not self._check_and_confirm_overwrite(results_file):
            return

        # Load dataset
        questions, answers, choices, metadata = self._load_dataset()

        # Setup model and tokenizer
        model, tokenizer = self.setup_model()
        
        # Setup attack config if needed
        if self.apply_attack and self.config.attack_type and self.config.custom_jailbreak==False:
            attack_config = setup_attack(self.config.attack_type, self.config)
        else:
            attack_config = None
        
        # Get formatted prompts
        formatted_data = self.get_formatted_prompts(questions, choices, metadata)
        
        results = []
        start_num = int(self.config.question_range.split('-')[0])

        for i, (data, answer_idx) in enumerate(tqdm(zip(formatted_data, answers), 
                                                  total=len(formatted_data),
                                                  desc=f"Evaluating {self.config.dataset} questions")):
            question_num = start_num + i
            
            if self.apply_attack:
                if self.config.attack_type or self.config.custom_jailbreak:
                    logging.info(f"\nQuestion {question_num}:")
                    logging.info(f"Question: {data['question']}")
                    response, info = self._run_with_attack(data, model, tokenizer, attack_config)
                    logging.info(f"Model response: {response}")
                    if info is not None:
                        logging.info(f"Attack info: {info}")
                else:
                    raise ValueError("apply_attack is True but attack_type is not specified in Attack config and custom_jailbreak is False")
            else:
                logging.info(f"\nQuestion {question_num}:")
                logging.info(f"Question: {data['question']}")
                response = self._run_with_no_attack(data, model, tokenizer)
                info = None
                logging.info(f"Model response: {response}")

            result_params = {
                "question_num": question_num,
                "question": data['question'],
                "choices": data.get('choices', None),
                "answer": answer_idx,
                "response": response,
            }
            if info is not None:
                result_params["info"] = info
            if 'subject' in data and 'level' in data:
                result_params["subject"] = data['subject']
                result_params["level"] = data['level']
            if 'evil_question' in data and data['evil_question'] is not None:
                result_params["evil_question"] = data['evil_question']
                result_params["benign_question"] = data['benign_question']
                
            result = self._format_result(**result_params)
            results.append(result)

        self._save_results(results)
        print(f"\nResults saved to: {results_file}")

        self.cleanup()
        
        return results, results_file

    @abstractmethod
    def _get_base_output_dir(self) -> Path:
        """Get the base output directory specific to the experiment type."""
        pass

    @classmethod
    @abstractmethod
    def get_default_dataset_config(cls) -> Any:
        """Return default dataset configuration for this experiment type."""
        pass

    @classmethod
    @abstractmethod
    def get_default_alignment_config(cls) -> Any:
        """Return default alignment configuration for this experiment type."""
        pass

    def cleanup(self):
        """Clean up resources by deleting model and tokenizer and freeing memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        gc.collect()

    def set_custom_save_dir(self, custom_dir: str | None) -> None:
        """Set or update the custom save directory.
        
        Args:
            custom_dir: Path to custom save directory, or None to revert to default structure
        """
        if hasattr(self.config, 'custom_save_dir'):
            self.config.custom_save_dir = custom_dir
        else:
            setattr(self.config, 'custom_save_dir', custom_dir)
        
        # Update the output directory path based on new setting
        self.update_output_dir()

    def get_standardized_attack_name(self) -> str:
        """
        Get a standardized name for the attack type.
        
        Returns:
            str: Standardized attack name for use in paths and filenames
        """
        if not self.apply_attack:
            return "no_attack"
        
        if self.config.custom_jailbreak and self.config.custom_jailbreak_fn:
            return self.config.custom_jailbreak_fn.__name__
        
        if not self.config.attack_type:
            return "custom_jailbreak"
        
        # Map attack types to standardized names
        attack_name_map = {
            "simple_jb": "SystemPromptJB",
            "translate": "MultiJail",
            "PAIR": "PAIR",
            "TAP": "TAP",
            "AutoDAN": "AutoDAN",
            "gcg": "GCG",
            "finetune": "finetune",
            "many-shot": "many-shot"
        }
        
        base_name = attack_name_map.get(self.config.attack_type, self.config.attack_type)
        
        # Add subfolder for translate attack based on language
        if self.config.attack_type == "translate":
            if hasattr(self.config, 'target_language') and self.config.target_language:
                return f"{base_name}_{self.config.target_language}"
        
        # Add subfolder for PAIR attack when don't_modify_question is true and include max_rounds
        if self.config.attack_type == "PAIR":
            if hasattr(self.config, 'dont_modify_question') and self.config.dont_modify_question:
                return f"{base_name}_no_question_modification"
        
        return base_name

class WMDPExperiment(BaseExperiment):
    """WMDP-specific experiment implementation."""
    
    VALID_DATASETS = ["wmdp-bio"]  # Add other valid WMDP datasets here
    
    def _create_unified_config(self, model_config, dataset_config, attack_config, alignment_config):
        if dataset_config.dataset not in self.VALID_DATASETS:
            raise ValueError(f"Invalid dataset '{dataset_config.dataset}' for WMDPExperiment. Valid options are: {self.VALID_DATASETS}")
        return UnifiedConfig.from_modular_configs(
            model_config=model_config,
            dataset_config=dataset_config,
            attack_config=attack_config,
            alignment_config=alignment_config
        )

    def _get_base_output_dir(self) -> Path:
        """Get the base output directory for WMDP experiments."""
        return Path("Results/WMDP")

    def _load_dataset(self):
        ds = load_dataset("cais/wmdp", self.config.dataset)
        questions, answers, choices = self._select_questions(
            ds, 
            self.config.question_range,
            self.config.shuffle_choices
        )
        return questions, answers, choices, None  # No info for WMDP

    def _select_questions(self, ds, question_range, shuffle_choices=False):
        """
        Select a subset of questions from a dataset based on the given range.
        
        Args:
            ds: Dataset object (typically from HuggingFace datasets)
            question_range: String in format "start-end" (e.g., "0-10")
            shuffle_choices: Whether to shuffle the choices for each question
        
        Returns:
            tuple: (questions, answers, choices)
        """
        random.seed(42)
        start, end = map(int, question_range.split('-'))
        indices = list(range(start, end + 1))
        questions = [ds["test"]["question"][i] for i in indices]
        answers = [ds["test"]["answer"][i] for i in indices]
        choices = [ds["test"]["choices"][i] for i in indices]
        
        if shuffle_choices:
            # Randomly permute choices and adjust answers
            for i in range(len(questions)):
                # Create list of (choice, is_correct) pairs
                choice_pairs = list(enumerate(choices[i]))
                correct_idx = answers[i]
                
                # Randomly shuffle the pairs
                random.shuffle(choice_pairs)
                
                # Update choices and answer
                choices[i] = [choice for idx, choice in choice_pairs]
                # Find new position of the correct answer
                answers[i] = [idx for idx, (old_idx, _) in enumerate(choice_pairs) if old_idx == correct_idx][0]
        
        return questions, answers, choices

    def get_formatted_prompts(self, questions, choices=None, metadata=None):
        if choices is None:
            raise ValueError("choices must be provided for WMDP")
        formatted_data = []
        for question, choice_list in zip(questions, choices):
            messages = get_model_prompt(
                question=question,
                choices=choice_list,
                dataset=self.config.dataset,
                SP_alignment=(self.apply_alignment and self.config.alignment_method == 'system_prompt'),
                args=self.config
            )
            
            formatted_data.append({
                'messages': messages,
                'question': question,
                'choices': choice_list
            })
        
        return formatted_data

    def _run_with_no_attack(self, formatted_data, model, tokenizer):
        if self.config.api_or_local == "api":
            response = query_model_api(
                formatted_data['messages'], 
                self.config, 
                select_model=model
            )
            return {'role': 'assistant', 'content': response}
        elif self.config.api_or_local == "local":
            response = query_model(
                formatted_data['messages'],
                model,
                tokenizer,
                deterministic=self.config.deterministic,
                top_p=self.config.top_p,
                temperature=self.config.temperature
            )
            return response

    def _run_with_attack(self, formatted_data, model, tokenizer, attack_config):
        if self.config.custom_jailbreak and self.config.custom_jailbreak_fn:
            return self.config.custom_jailbreak_fn(self, formatted_data, model, tokenizer)
        else:
            response, response_info = generate_jailbreak_response(
                question=formatted_data['question'],
                choices=formatted_data['choices'],
                dataset=self.config.dataset,
                SP_alignment=(self.apply_alignment and self.config.alignment_method == 'system_prompt'),
                attack_type=self.config.attack_type,
                api_or_local=self.config.api_or_local,
                model=model,
                tokenizer=tokenizer,
                config=attack_config,
                args=self.config
            )
            return response, response_info

    def _format_result(self, question_num, question, choices, answer, response, info=None, **kwargs):
        result = {
            "question_id": question_num,
            "question": question,
            "choices": choices,
            "correct_letter": list("ABCD")[answer],
            "correct_answer": choices[answer],
            "response": response
        }
        if info:
            result["attack_prompt"] = info
        return result

    def save_formatted_prompts(self):
        """
        Load data and save formatted prompts as a separate dataset in JSON format.
        Creates a JSON file with chat messages format ready to use with models.
        """
        # Ensure the output directory is updated
        # self.update_output_dir()

        filename = f"prompts_{self.config.dataset}_{self.config.alignment_method}.json"
        prompts_file = self.output_dir / filename

        if not self._check_and_confirm_overwrite(prompts_file):
            return

        # Load dataset
        questions, answers, choices, _ = self._load_dataset()
        formatted_data = self.get_formatted_prompts(questions, choices)
        prompts_dataset = []
        
        start_num = int(self.config.question_range.split('-')[0])
        
        for i, (data, answer_idx) in enumerate(zip(formatted_data, answers)):
            question_num = start_num + i
            prompt_entry = {
                "question_id": question_num,
                "messages": data["messages"],
                "metadata": {
                    "question": data["question"],
                    "choices": data["choices"],
                    "correct_letter": list("ABCD")[answer_idx],
                    "correct_answer": data["choices"][answer_idx]
                }
            }
            prompts_dataset.append(prompt_entry)
        
        with open(prompts_file, 'w') as f:
            json.dump(prompts_dataset, f, indent=2)
        
        print(f"Formatted prompts saved to: {prompts_file}")

    @classmethod
    def get_default_dataset_config(cls) -> DatasetConfig:
        return DatasetConfig(
            dataset="wmdp-bio",
            question_range="0-10",
            shuffle_choices=False,
            format_type="as_gcg_target_string_with_explanation",
            judging_format="check_if_correct_with_refusal"
        )

    @classmethod
    def get_default_alignment_config(cls) -> AlignmentConfig:
        return AlignmentConfig(
            alignment_method="system_prompt",
            system_prompt_type="refuse_biology",
            alignment_model_path="finetune/finetuned_model_with_biology"
        )

class GSM8KExperiment(BaseExperiment):
    """GSM8K-specific experiment implementation."""
    
    VALID_DATASETS = ["gsm8k", "gsm8k-evil"]
    
    def _create_unified_config(self, model_config, dataset_config, attack_config, alignment_config):
        if dataset_config.dataset not in self.VALID_DATASETS:
            raise ValueError(f"Invalid dataset '{dataset_config.dataset}' for GSM8KExperiment. Valid options are: {self.VALID_DATASETS}")
        return UnifiedConfig.from_modular_configs(
            model_config=model_config,
            dataset_config=dataset_config,
            attack_config=attack_config,
            alignment_config=alignment_config
        )

    def _get_base_output_dir(self) -> Path:
        """Get the base output directory for GSM8K experiments."""
        if self.config.dataset == "gsm8k-evil":
            return Path("Results/EvilMath")
        else:
            return Path("Results/GSM8K")

    def _load_local_dataset(self, file_path):
        """Load the local dataset from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        filtered_data = data
        questions = [item["reworded_benign"] for item in filtered_data]
        reworded_questions = [item["reworded_question"] for item in filtered_data]
        answers = [self._extract_numeric_answer(item["correct_answer"]) for item in filtered_data]
        return questions, reworded_questions, answers

    def _select_questions_local(self, questions, reworded_questions, answers, question_range):
        """Select a subset of questions based on the given range."""
        start, end = map(int, question_range.split('-'))
        selected_questions = questions[start:end+1]
        selected_reworded = reworded_questions[start:end+1]
        selected_answers = answers[start:end+1]
        return selected_questions, selected_reworded, selected_answers

    def _extract_numeric_answer(self, answer_str):
        """Extract numeric answer from answer string."""
        # First try to find a number at the end of the string
        match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$', answer_str.strip())
        if match:
            return float(match.group())
        
        # If no number at the end, try to find any number in the string
        match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', answer_str)
        if match:
            return float(match.group())
        
        return None

    def _select_questions(self, ds, question_range):
        """Select questions and extract numeric answers."""
        start, end = map(int, question_range.split('-'))
        questions = ds["test"]["question"][start:end+1]
        answers = ds["test"]["answer"][start:end+1]
        # Extract numeric answers from the answer strings
        answers = [self._extract_numeric_answer(answer) for answer in answers]
        return questions, answers

    def _load_dataset(self):
        if self.config.dataset == 'gsm8k':
            ds = load_dataset("gsm8k", "main")
            questions, answers = self._select_questions(ds, self.config.question_range)
            return questions, answers, None, None  # No choices or info for GSM8K
        elif self.config.dataset == 'gsm8k-evil':
            # Check if using local file or HuggingFace dataset (default)
            if hasattr(self.config, 'use_local_dataset') and self.config.use_local_dataset:
                logging.info(f"Loading EvilMath dataset from local file: {self.config.evil_dataset_path}")
                # Load from local file
                benign_questions, evil_questions, answers = self._load_local_dataset(self.config.evil_dataset_path)
                benign_questions, evil_questions, answers = self._select_questions_local(
                    benign_questions, evil_questions, answers, self.config.question_range
                )
            else:
                # Load EvilMath dataset from HuggingFace (default behavior)
                evil_ds = load_dataset("ethz-spylab/EvilMath")
                # Extract questions and answers using the exact field names
                start, end = map(int, self.config.question_range.split('-'))
                benign_questions = [evil_ds["train"][i]["unicorn_question"] for i in range(start, end+1)]
                evil_questions = [evil_ds["train"][i]["evil_question"] for i in range(start, end+1)]
                
                # Extract numeric answers
                answers = [self._extract_numeric_answer(evil_ds["train"][i]["original_answer"]) 
                          for i in range(start, end+1)]
            
            metadata = {
                "evil_question": evil_questions,
                "benign_question": benign_questions
            }
            return benign_questions, answers, None, metadata

    def get_formatted_prompts(self, questions, choices=None, metadata=None):
        formatted_data = []
        
        # For gsm8k-evil dataset, use evil questions when alignment is enabled
        use_evil_questions = (
            self.config.dataset == "gsm8k-evil" and 
            metadata and 
            "evil_question" in metadata and
            self.apply_alignment
        )
        
        # Choose which questions to use
        questions_to_use = metadata["evil_question"] if use_evil_questions else questions
        
        for i, question in enumerate(questions_to_use):
            messages = get_model_prompt(
                question=question,
                choices=None,  # GSM8K doesn't have choices
                dataset=self.config.dataset,
                SP_alignment=(self.apply_alignment and self.config.alignment_method == 'system_prompt'),
                args=self.config
            )
            
            formatted_data_entry = {
                'messages': messages,
                'question': question,
            }
            
            # For gsm8k-evil dataset, add evil/benign questions to metadata
            if metadata and "evil_question" in metadata:
                formatted_data_entry['evil_question'] = metadata["evil_question"][i]
            if metadata and "benign_question" in metadata:
                formatted_data_entry['benign_question'] = metadata["benign_question"][i]
                
            formatted_data.append(formatted_data_entry)
        
        return formatted_data

    def _run_with_no_attack(self, formatted_data, model, tokenizer):
        if self.config.api_or_local == "api":
            response = query_model_api(
                formatted_data['messages'], 
                self.config, 
                select_model=model
            )
            return {'role': 'assistant', 'content': response}
        elif self.config.api_or_local == "local":
            response = query_model(
                formatted_data['messages'],
                model,
                tokenizer,
                deterministic=self.config.deterministic,
                top_p=self.config.top_p,
                temperature=self.config.temperature
            )
            return response

    def _run_with_attack(self, formatted_data, model, tokenizer, attack_config):
        if self.config.custom_jailbreak and self.config.custom_jailbreak_fn:
            return self.config.custom_jailbreak_fn(self, formatted_data, model, tokenizer)
        else:
            response, response_info = generate_jailbreak_response(
                question=formatted_data['question'],
                choices=None,  # GSM8K doesn't have choices
                dataset=self.config.dataset,
                SP_alignment=(self.apply_alignment and self.config.alignment_method == 'system_prompt'),
                attack_type=self.config.attack_type,
                api_or_local=self.config.api_or_local,
                model=model,
                tokenizer=tokenizer,
                config=attack_config,
                args=self.config
            )
            return response, response_info

    def _format_result(self, question_num, question, choices, answer, response, info=None, **kwargs):
        result = {
            "question_id": question_num,
            "question": question,
            "correct_answer": answer,
            "response": response,
        }
        
        # Add optional fields (only for gsm8k-evil (EvilMath))
        optional_fields = {
            "evil_question": kwargs.get('evil_question'),
            "benign_question": kwargs.get('benign_question')
        }
        
        # Add each field that's not None
        for key, value in optional_fields.items():
            if value is not None:
                result[key] = value
            
        if info:
            result["jailbreak_info"] = info
        return result

    def save_formatted_prompts(self):
        """
        Load data and save formatted prompts as a separate dataset in JSON format.
        Creates a JSON file with chat messages format ready to use with models.
        """
        # Ensure the output directory is updated
        # self.update_output_dir()

        filename = f"prompts_{self.config.dataset}_{self.config.alignment_method}.json"
        prompts_file = self.output_dir / filename

        if not self._check_and_confirm_overwrite(prompts_file):
            return

        # Load dataset
        questions, answers, _, metadata = self._load_dataset()
        formatted_data = self.get_formatted_prompts(questions, metadata=metadata)
        prompts_dataset = []
        
        start_num = int(self.config.question_range.split('-')[0])
        
        for i, (data, answer) in enumerate(zip(formatted_data, answers)):
            question_num = start_num + i
            prompt_entry = {
                "question_id": question_num,
                "messages": data["messages"],
                "metadata": {
                    "question": data["question"],
                    "correct_answer": answer
                }
            }
            prompts_dataset.append(prompt_entry)
        
        with open(prompts_file, 'w') as f:
            json.dump(prompts_dataset, f, indent=2)
        
        print(f"Formatted prompts saved to: {prompts_file}")

    @classmethod
    def get_default_dataset_config(cls) -> DatasetConfig:
        return DatasetConfig(
            dataset="gsm8k",
            question_range="0-10",
            format_type="start_with_explanation_detailed",
            evil_dataset_path="/home/knikolic/gsm8k_evil/Questions/Filtered/questions_Llama-3.1-8B-Lexi-Uncensored-V2_claude-3.5-haiku_200questions.json",
            use_local_dataset=False
        )

    @classmethod
    def get_default_alignment_config(cls) -> AlignmentConfig:
        return AlignmentConfig(
            alignment_method="system_prompt",
            system_prompt_type="refuse_math",
            alignment_model_path="ethz-spylab/Llama-3.1-8B-Instruct_do_math_chat"
        )

class MATHExperiment(BaseExperiment):
    """MATH-specific experiment implementation."""

    VALID_DATASETS = ["math"]
    
    def _create_unified_config(self, model_config, dataset_config, attack_config, alignment_config):
        if dataset_config.dataset not in self.VALID_DATASETS:
            raise ValueError(f"Invalid dataset '{dataset_config.dataset}' for MATHExperiment. Valid options are: {self.VALID_DATASETS}")
        return UnifiedConfig.from_modular_configs(
            model_config=model_config,
            dataset_config=dataset_config,
            attack_config=attack_config,
            alignment_config=alignment_config
        )

    def _get_base_output_dir(self) -> Path:
        """Get the base output directory for MATH experiments."""
        return Path("Results/MATH")

    def _load_dataset(self):
        """Load problems from the MATH dataset."""
        split_path = os.path.join(self.config.MATH_dataset_path, "test")
        problems = []
        target_counts = {
            'counting_and_probability': 39,
            'algebra': 135,
            'number_theory': 30,
            'intermediate_algebra': 52,
            'geometry': 38,
            'precalculus': 57,
            'prealgebra': 86
        }
        problems_by_subject = {subject: [] for subject in target_counts.keys()}

        for subject in sorted(os.listdir(split_path)):
            subject_path = os.path.join(split_path, subject)
            if os.path.isdir(subject_path):
                for problem_file in sorted(os.listdir(subject_path)):
                    if problem_file.endswith('.json'):
                        with open(os.path.join(subject_path, problem_file), 'r') as f:
                            problem = json.load(f)
                            if problem.get('level') == self.config.math_level:
                                problem['subject'] = subject
                                if subject in target_counts:
                                    problems_by_subject[subject].append(problem)
                                problems.append(problem)

        final_problems = []
        for subject, target_count in target_counts.items():
            available_problems = problems_by_subject[subject]
            selected_count = min(target_count, len(available_problems))
            selected_problems = available_problems[:selected_count]
            final_problems.extend(selected_problems)

        # Use the select_math_questions method here
        questions, answers, subjects, levels = self._select_math_questions(final_problems, self.config.question_range)
        return questions, answers, None, {
            'subjects': subjects,
            'levels': levels
        }

    def _select_math_questions(self, problems, question_range):
        """Select a subset of questions based on the given range."""
        if not problems:
            raise ValueError("No level 1 problems found in the dataset")
        
        start, end = map(int, question_range.split('-'))
        if end >= len(problems):
            raise ValueError(f"Requested range {start}-{end} exceeds available level 1 problems (total: {len(problems)})")
        
        selected_problems = problems[start:end+1]
        
        questions = [p['problem'] for p in selected_problems]
        answers = [p['solution'] for p in selected_problems]
        subjects = [p['subject'] for p in selected_problems]
        levels = [p['level'] for p in selected_problems]
        return questions, answers, subjects, levels

    def get_formatted_prompts(self, problems, choices=None, metadata=None):
        if metadata is None or 'subjects' not in metadata or 'levels' not in metadata:
            raise ValueError("subjects and levels must be provided in metadata for MATH")
        
        formatted_data = []
        for problem, subject, level in zip(problems, metadata['subjects'], metadata['levels']):
            messages = get_model_prompt(
                question=problem,
                choices=None,  # MATH doesn't have choices
                dataset="math",
                SP_alignment=(self.apply_alignment and self.config.alignment_method == 'system_prompt'),
                args=self.config
            )
            formatted_data.append({
                'messages': messages,
                'question': problem,
                'subject': subject,
                'level': level
            })
        return formatted_data

    def _run_with_no_attack(self, formatted_data, model, tokenizer):
        if self.config.api_or_local == "api":
            response = query_model_api(
                formatted_data['messages'], 
                self.config, 
                select_model=model
            )
            return {'role': 'assistant', 'content': response}
        elif self.config.api_or_local == "local":
            response = query_model(
                formatted_data['messages'],
                model,
                tokenizer,
                deterministic=self.config.deterministic,
                top_p=self.config.top_p,
                temperature=self.config.temperature
            )
            return response

    def _run_with_attack(self, formatted_data, model, tokenizer, attack_config):
        if self.config.custom_jailbreak and self.config.custom_jailbreak_fn:
            return self.config.custom_jailbreak_fn(self, formatted_data, model, tokenizer)
        else:
            response, response_info = generate_jailbreak_response(
                question=formatted_data['question'],
                choices=None,  # MATH doesn't have choices
                dataset="math",
                SP_alignment=(self.apply_alignment and self.config.alignment_method == 'system_prompt'),
                attack_type=self.config.attack_type,
                api_or_local=self.config.api_or_local,
                model=model,
                tokenizer=tokenizer,
                config=attack_config,
                args=self.config
            )
            return response, response_info

    def _format_result(self, question_num, question, choices, answer, response, info=None, **kwargs):
        result = {
            "question_id": question_num,
            "subject": kwargs.get('subject'),
            "level": kwargs.get('level'),
            "question": question,
            "correct_answer": answer,
            "response": response
        }
        if info:
            result["jailbreak_info"] = info
        return result

    def save_formatted_prompts(self):
        """
        Load data and save formatted prompts as a separate dataset in JSON format.
        Creates a JSON file with chat messages format ready to use with models.
        """
        # Ensure the output directory is updated
        # self.update_output_dir()

        filename = f"prompts_{self.config.dataset}_{self.config.alignment_method}.json"
        prompts_file = self.output_dir / filename

        if not self._check_and_confirm_overwrite(prompts_file):
            return

        # Load dataset
        problems, answers, _, metadata = self._load_dataset()
        
        formatted_data = self.get_formatted_prompts(problems, choices=None, metadata=metadata)
        prompts_dataset = []
        
        for i, (data, problem, answer, subject, level) in enumerate(zip(formatted_data, problems, answers, metadata['subjects'], metadata['levels'])):
            prompt_entry = {
                "question_id": i,
                "messages": data["messages"],
                "metadata": {
                    "question": problem,
                    "correct_answer": answer,
                    "subject": subject,
                    "level": level
                }
            }
            prompts_dataset.append(prompt_entry)
        
        with open(prompts_file, 'w') as f:
            json.dump(prompts_dataset, f, indent=2)
        
        print(f"Formatted prompts saved to: {prompts_file}")

    @classmethod
    def get_default_dataset_config(cls) -> DatasetConfig:
        return DatasetConfig(
            dataset="math",
            question_range="0-10",
            format_type="boxed_start_with_explanation_detailed",
            judging_format="check_boxed_answer",
            MATH_dataset_path="/data/datasets/MATH_BACKUP",
            math_level="Level 1"
        )

    @classmethod
    def get_default_alignment_config(cls) -> AlignmentConfig:
        return AlignmentConfig(
            alignment_method="system_prompt",
            system_prompt_type="refuse_math",
            alignment_model_path="ethz-spylab/Llama-3.1-8B-Instruct_do_math_chat"
        )