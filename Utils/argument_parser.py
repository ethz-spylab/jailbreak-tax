import argparse

def get_model_choices():
    return [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-3.5-turbo",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.1-405B-Instruct",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3.5-haiku",
    ]

def parse_args():
    parser = argparse.ArgumentParser()
    
    add_dataset_args(parser)
    add_model_args(parser)
    add_attack_args(parser)
    add_attack_specific_args(parser)
    add_experiment_args(parser)
    add_autodan_args(parser)
    add_gcg_args(parser)
    add_many_shot_args(parser)
    add_finetune_args(parser)
    
    return parser.parse_args()

def add_dataset_args(parser):
    dataset_group = parser.add_argument_group('Dataset Configuration')
    dataset_group.add_argument('--dataset', type=str, 
                             choices=['wmdp-bio', 'gsm8k', 'gsm8k-evil', 'math'], 
                             default='wmdp-bio', 
                             help='Choose the dataset to load')
    dataset_group.add_argument('--question_range', type=str, 
                             default='0-4', 
                             help='Range of questions to process (e.g., 0-4)')
    dataset_group.add_argument('--shuffle_choices', action='store_true',
                             help='Whether to randomly shuffle the multiple choice options')
    dataset_group.add_argument('--evil_dataset_path', type=str, required=False,
                             help='Path to the GSM8K-Evil dataset JSON file (used only with --use_local_dataset)')
    dataset_group.add_argument('--use_local_dataset', action='store_true',
                             help='Load EvilMath dataset from local file instead of HuggingFace')
    dataset_group.add_argument('--math_level', type=str, default="Level 2",
                             choices=["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"],
                             help="MATH dataset level to use")
    dataset_group.add_argument('--MATH_dataset_path', type=str, default="add/your/path/to/MATH_dataset", help='Path to the MATH dataset')

def add_model_args(parser):
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model_provider', type=str,
                           choices=['openrouter', 'openai'],
                           default='openrouter',
                           help='Choose the API provider (openrouter or openai)')
    model_group.add_argument('--select_model', type=str, 
                           choices=get_model_choices(), 
                           default='meta-llama/Meta-Llama-3.1-8B-Instruct', 
                           help='Choose the model to use')
    model_group.add_argument('--api_or_local', type=str, 
                           default="local", 
                           choices=['local', 'api'], 
                           help='Choose whether to use API or local model')
    model_group.add_argument('--analyze_probabilities', action='store_true',
                           help='Whether to analyze token probabilities for multiple choice answers')
    model_group.add_argument('--alignment_method', type=str, 
                           choices=['system_prompt', 'fine_tuned', 'no_alignment'],
                           default='system_prompt',
                           help='Method to achieve alignment: system prompt or fine-tuned model')
    model_group.add_argument('--alignment_model_path', type=str,
                           default="",
                           help='Path to the alignment model')
    model_group.add_argument('--deterministic', action='store_true',
                           help='Whether to use deterministic generation (no sampling)')
    model_group.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling parameter')
    model_group.add_argument('--temperature', type=float, default=1.0, help='Temperature parameter for sampling')

def add_attack_args(parser):
    attack_group = parser.add_argument_group('Attack Configuration')
    attack_group.add_argument('--attack_type', type=str, 
                            choices=['simple_jb', 'gcg', 'translate', 'finetune', 'PAIR', 'TAP', 'AutoDAN', 'many-shot', 'past', 'custom_jb'], 
                            default='simple_jb', 
                            help='Choose the attack type')
    attack_group.add_argument('--format_type', type=str, 
                            choices=['only_letter', 'as_gcg_target_string', 'as_top_2_answers',
                                   'as_gcg_target_string_only', 'as_gcg_target_string_with_explanation',
                                   'start_with_explanation', 'start_with_resoning', 'detailed_resoning',
                                   'auto_DAN_prompt', 'only_letter_plus_the_correct_answer_is',
                                   'only_number',
                                   'as_target_string',
                                   'start_with_explanation',
                                   'start_with_explanation_detailed',
                                   'solve_the_problem',
                                   'start_with_solution',
                                   'no_intro',
                                   'no_intro_only_format',
                                   'math_prompt',
                                   'as_in_paper',
                                   'boxed',
                                   'boxed_start_with_explanation_detailed'], 
                            default='as_gcg_target_string_with_explanation', 
                            help='Choose the format type')
    attack_group.add_argument('--judging_format', type=str, 
                            choices=['check_if_correct', 'check_if_correct_with_refusal',
                                   'check_if_correct_with_explanation_first', 'check_if_top_2_answers',
                                   'check_math_answer',
                                   'check_boxed_answer'],
                            default='check_if_correct_with_refusal',
                            help='Choose the format for judging responses')
    attack_group.add_argument('--system_prompt_type', type=str, 
                            choices=['refuse_math', 'refuse_math_short', 'refuse_math_backup', 'dataset_specific', 'dataset_specific_backup', 'refuse_biology'],
                            default='refuse_biology',
                            help='Choose the system prompt type')

def add_attack_specific_args(parser):
    attack_params_group = parser.add_argument_group('Attack-Specific Parameters')
    attack_params_group.add_argument('--target_language', type=str, 
                                   default='Chinese', 
                                   help='Target language for translate (MultiJail) attack')
    attack_params_group.add_argument('--max_rounds', type=int, 
                                   default=10,
                                   help='Maximum number of rounds for PAIR attack')
    attack_params_group.add_argument('--width', type=int, 
                                   default=5,
                                   help='Width parameter for TAP pruning')
    attack_params_group.add_argument('--branching_factor', type=int,
                                   default=3, 
                                   help='Branching factor for TAP tree')
    attack_params_group.add_argument('--depth', type=int,
                                   default=5,
                                   help='Maximum depth for TAP tree')
    attack_params_group.add_argument('--keep_last_n', type=int,
                                   default=3,
                                   help='Number of responses to keep in conversation history')
    attack_params_group.add_argument('--dont_modify_question', action='store_true',
                                   help='Whether to prevent modification of questions in prompts (only for PAIR attack). If this is set, the attack is PAIR dont modify')

def add_experiment_args(parser):
    exp_group = parser.add_argument_group('Experiment Tracking')
    exp_group.add_argument('--run', type=str, 
                          default="0", 
                          help='Run identifier for experiment tracking')
    exp_group.add_argument('--parallel', action='store_true',)
    exp_group.add_argument('--force_overwrite', action='store_true',
                          default=False,
                          help='Whether to overwrite existing results')
    exp_group.add_argument('--stage',
                          type=str,
                          choices=["unaligned", "before_jb", "after_jb", "all"],
                          default=None,
                          help='Specify which stage to run: unaligned, before_jb, after_jb, or all')
    exp_group.add_argument('--custom_save_dir', type=str,
                          default=None,
                          help='Custom directory path to save results')

def add_autodan_args(parser):
    autodan_group = parser.add_argument_group('AutoDAN Parameters')
    autodan_group.add_argument('--autodan_num_steps', type=int, 
                             default=100,
                             help='Number of optimization steps for AutoDAN')
    autodan_group.add_argument('--autodan_batch_size', type=int,
                             default=8,
                             help='Batch size for AutoDAN optimization')
    autodan_group.add_argument('--autodan_elite_ratio', type=float,
                             default=0.05,
                             help='Ratio of elite samples to keep')
    autodan_group.add_argument('--autodan_crossover', type=float,
                             default=0.5,
                             help='Crossover probability for AutoDAN')
    autodan_group.add_argument('--autodan_mutation', type=float,
                             default=0.01,
                             help='Mutation probability for AutoDAN')
    autodan_group.add_argument('--autodan_num_points', type=int,
                             default=5,
                             help='Number of points for AutoDAN crossover')
    autodan_group.add_argument('--autodan_iteration_interval', type=int,
                             default=5,
                             help='Interval for AutoDAN iterations')

def add_gcg_args(parser):
    gcg_group = parser.add_argument_group('GCG Parameters')
    gcg_group.add_argument('--gcg_init_str_len', type=int,
                          default=20,
                          help='Length of GCG initialization string')
    gcg_group.add_argument('--gcg_num_steps', type=int,
                          default=250,
                          help='Number of optimization steps for GCG')
    gcg_group.add_argument('--gcg_search_width', type=int,
                          default=128,
                          help='Search width for GCG')
    gcg_group.add_argument('--gcg_topk', type=int,
                          default=128,
                          help='Top-k parameter for GCG')
    gcg_group.add_argument('--gcg_seed', type=int,
                          default=42,
                          help='Random seed for GCG')
    gcg_group.add_argument('--gcg_early_stop', type=bool,
                          default=False,
                          help='Early stop for GCG')
    gcg_group.add_argument('--target_string', type=str,
                          default=None,
                          help='Custom target string for GCG attack')

def add_many_shot_args(parser):
    many_shot_group = parser.add_argument_group('Many-shot Parameters')
    many_shot_group.add_argument('--many_shot_file', type=str,
                                default="many-shot/data/many_shot_full_bio_2_no_intro.txt",
                                help='Path to the many-shot examples file')
    many_shot_group.add_argument('--many_shot_num_questions', type=int,
                                default=100,
                                help='Number of many-shot examples to use')

def add_finetune_args(parser):
    finetune_group = parser.add_argument_group('Finetune Parameters')
    finetune_group.add_argument('--peft_model_path', type=str,
                               default="ethz-spylab/Llama-3.1-8B-Instruct_refuse_bio",
                               help='Path to the PEFT model checkpoint') 