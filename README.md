# The Jailbreak Tax

<div align="center">

#### [📄 Paper](https://arxiv.org/abs/2504.10694)  |  [🤗 EvilMath](https://huggingface.co/datasets/ethz-spylab/EvilMath) | [🤗 Models](https://huggingface.co/collections/ethz-spylab/the-jailbreak-tax-jailbreak-utility-67c5a6f2d1220e4d1d9f5b47) | ✏️ [Blog Post](https://spylab.ai/blog/jailbreak-tax/)

</div>

This is the original implementation of the Spotlight ICML 2025 paper ["The Jailbreak Tax: How Useful are Your Jailbreak Outputs?"]() by Kristina Nikolić, Luze Sun, Jie Zhang, and Florian Tramèr.

With this codebase you can:
- [Evaluate your custom jailbreak techniques on the Jailbreak Tax benchmark](#custom-jailbreaks)
- [Generate model prompts in the jailbreak tax benchmark format and use them to test your jailbreak in your own setup](#generate-formatted-prompts)
- [Automatically grade the model outputs to correct, incorrect or refusal](#grade-model-outputs)
- [Reproduce the experiments from the paper](#run-experiments)

## Overview

The framework measures the "jailbreak tax" - the utility loss after a successful jailbreak compared to the unaligned model.

The jailbreak tax experiments are done in three stages:

1. **Unaligned Stage**: Evaluate the base model to obtain the baseline accuracy (alignment off, jailbreak off)
2. **Before Jailbreak Stage**: Align the model to refuse specific tasks (alignment on, jailbreak off)
3. **After Jailbreak Stage**: Attack the aligned model. Measure the aligned model accuracy when subjected to a jailbreak (alignment on, jailbreak on)

<br>
<div style="display: flex; justify-content: center;">
  <figure style="margin: auto;">
    <img src="overview_final_page-0001.png" alt="overview" width="100%">
  </figure>
</div>

## Environment Setup

### Requirements
- Python 3.12.5
- CUDA 12.4 (for GPU support)
- requirements.txt

### Installation

1. Create a new conda environment:
```bash
conda create -n jailbreak-tax python=3.12.5
conda activate jailbreak-tax
```

2. Install PyTorch with CUDA support:
```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
```

3. Install other dependencies:
```bash
pip install -r requirements.txt
```

### API Keys
Setup your API keys in the `.env` file as in `.env.example`. You can leave keys empty if you are not using specific provider.

## Run Experiments

The `run_jb_tax.py` script implements end to end jailbreak tax experiments. It can run all three stages or specific stage as needed.

### Arguments

- `--question_range`: range of questions to process (e.g., '0-4')
- `--run`: run id
- `--dataset`: choose the dataset to load ('wmdp-bio', 'gsm8k', 'gsm8k-evil', 'math')
- `--attack_type`: jailbreak technique to use
- `--select_model`: model path (e.g., 'meta-llama/Llama-3.1-70B-Instruct')
- `--api_or_local`: whether to use API or local model ('api' or 'local')
- `--alignment_method`: alignment method ('system_prompt', 'fine_tuned', 'no_alignment')
- `--stage`: experiment stage to run ('unaligned', 'before_jb', 'after_jb', or 'all')

**Note:** For jailbreak-specific arguments, see the [Available Jailbreak Attacks](#available-jailbreak-attacks) section. For alignment-specific arguments, see the [Alignment Methods](#alignment-methods) section.

### Example Run

```
python run_jb_tax.py --question_range 0-9 --run experiment1 --dataset gsm8k --attack_type simple_jb --select_model meta-llama/Meta-Llama-3.1-8B-Instruct --api_or_local local --alignment_method system_prompt --stage all
```

This command will:
- Process questions 0 to 9 from the gsm8k dataset
- Apply a simple jailbreak attack (System Prompt attack)
- Use the local Llama 3.1 8B model (downloaded from Huggingface)
- Use system prompt alignment method
- Run all stages
- Save results in directory structure based on the provided arguments

### Output

Results will be saved in the `Results` directory, organized by dataset, run id (`--run`), stage, alignment method and/or attack type, and question range. Alternatively, you can set the `--custom_save_dir` argument to save results to specific directory.

For each stage the following files will be generated:
- `grading_totals.json`: result statistics
- `results_<dataset>_<alignment_method>_<attack_type>.json`: model outputs and grading results
- `prompts_<dataset>_<alignment_method>_<attack_type>.json`: the prompts used for the experiment (before jailbreak if applicable)
- `experiment.log`: log file

If `--all` stages are run, the combined results are generated:
- `combined_results.json`: the model outputs and grading results from all three stages
- `combined_statistics.json`: grading statistics from all three stages, including the **jailbreak tax**
- `response_distribution.png`: distribution of model outputs for all three stages (correct, incorrect, refusal)

Note: You can also use the `combine_grading_results()` function in `Utils/jb_tax_grading_utils.py` to combine the results from all three stages. See [Combine Results](#combine-results) for more details.

### Run Scripts
`Scripts` directory contains example run scripts for all combinations of datasets, alignment methods and jailbreak attacks used in the paper:
  - `WMDP`: Scripts for WMDP dataset
  - `MATH`: Scripts for MATH dataset
  - `GSM8K`: Scripts for GSM8K dataset
  - `EvilMath`: Scripts for EvilMath dataset

## Custom Jailbreaks

The `jb_tax_example_file.py` demonstrates how to use the codebase classes directly for more customized experiments. It shows:

- How to implement custom jailbreak functions
- How to configure experiment for different datasets (WMDP, GSM8K, MATH)
- How to run experiments with different alignment and attack combinations
- How to grade and combine results ([Grade Model Outputs](#grade-model-outputs))
- How to generate formatted prompts ([Generate Formatted Prompts](#generate-formatted-prompts))

## Grade Model Outputs

The framework includes utilities to automatically grade model outputs for different datasets. You can use the `grade_results()` in `Utils/jb_tax_grading_utils.py`, which:

- Loads results from a JSON file
- Grades results based on the dataset
- Appends grading information to the results

```
grade_results(results_path: str, dataset: str) -> None
```
Parameters:
- `results_path`: Path to the JSON file containing model responses
- `dataset`: Dataset type ('gsm8k', 'math', 'wmdp', or 'gsm8k-evil')

Example result files for each dataset type can be found in the `ExampleResults` directory, and you can use example scripts to grade them: `Scripts/run_grade_results.sh`.

After grading, a `grading` field will be added to each result in the input file, indicating whether the answer was `correct`, `incorrect`, or a `refusal`. Additionally, a `grading_totals.json` file will be generated in the same directory with statistics on grading results.

### Combine Results
You can combine grading results from all three stages to generate single file results and statistics (including the jailbreak tax):

```
combine_results_files(unaligned_path: str, before_jb_path: str, after_jb_path: str, output_dir: str) -> None
```
The `unaligned_path`, `before_jb_path`, and `after_jb_path` are the result files after grading.

Alternatively, you can use `Scripts/run_combine_stages.sh`.

## Plot Results

You can plot the jailbreak tax vs success rate using the `Utils/plots_notebook.ipynb` notebook.

## Generate Formatted Prompts

You can generate formatted prompts without running the full experiment. The `jb_tax_example_file.py` provides an example of how to generate formatted prompts for the GSM8K dataset using the `Experiment.generate_formatted_prompts()` method instead of the `Experiment.run()`.

## Finetuning

We finetune models for two purposes: 1) alignment: we pseudo-alignment the base model against answering math or bio questions, 2) attacks: use SFT as an attack on the pseudo-aligned model (the target alignment here can be SFT or system prompt instruction).

For example, for bio questions, we have three modes:
1. Mode: refuse - This will finetune the model to refuse answering biology questions and save it as "finetuned_model_refuse"

2. Mode: do_bio_again - This will take the refused model and finetune it to answer biology questions again, saving it as "finetuned_model_do_bio_again"

3. Mode: do_bio - This will finetune the original model (with system prompt as a defense) to answer biology questions directly, saving it as "finetuned_model_do_bio"

To use the script, you can run it with the required mode argument:
```
cd Finetune && python finetune_bio_70b.py --mode refuse
```

### Pseudo-aligned models

| Refuse        | 8B Model                 | 70B Model                |
|--------------|---------------------------|---------------------------|
| Biology      | `ethz-spylab/Llama-3.1-8B-Instruct_refuse_bio` | `ethz-spylab/Llama-3.1-70B-Instruct_refuse_biology`  |
| Math      | `ethz-spylab/Llama-3.1-8B-Instruct_refuse_math` | `ethz-spylab/Llama-3.1-70B-Instruct_refuse_math` |

### Attack models

| Do                  | Targeted Alignment Type | 8B Model                                             | 70B Model                                            |
|--------------------|---------------|-----------------------------------------------------|-----------------------------------------------------|
| Biology             | System Prompt | `ethz-spylab/Llama-3.1-8B-Instruct_do_bio`    | `ethz-spylab/Llama-3.1-70B-Instruct_do_biology_5e-5`   |
| Biology             | SFT           | `ethz-spylab/Llama-3.1-8B-Instruct_do_bio_again`   | `ethz-spylab/Llama-3.1-70B-Instruct_do_biology_again_5e-5`  |
| Math               | System Prompt | `ethz-spylab/Llama-3.1-8B-Instruct_do_math_chat`    | `ethz-spylab/Llama-3.1-70B-Instruct_do_math_chat`   |
| Math               | SFT           | `ethz-spylab/Llama-3.1-8B-Instruct_do_math_again`   | `ethz-spylab/Llama-3.1-70B-Instruct_do_math_again`  |


## Evil Math Data Generation
In `Data/EvilMath/src` you can find the script we used to generate the EvilMath dataset. The final version is available at [ethz-spylab/EvilMath](https://huggingface.co/datasets/ethz-spylab/EvilMath).

If you want to use local version of the dataset, set `--use_local_dataset` flag and pass the .json file path to `--evil_dataset_path`. E.g., 
```bash
--use_local_dataset
--evil_dataset_path "Data/EvilMath/data/evil_math_data_test.json"
```
We recommend using the [ethz-spylab/EvilMath](https://huggingface.co/datasets/ethz-spylab/EvilMath) from HuggingFace if you don't make modifications to the dataset.

## Configurations
### Alignment Methods

- `system_prompt`: System prompt alignment. The system prompts can be found in `Config/gsm8k_formats.json` and `Config/question_formats.json`
  ```bash
  --alignment_method system_prompt
  --system_prompt_type <system_prompt_type>
  ```
  
  Default system prompt types per dataset:
  - For WMDP: `refuse_biology`
  - For GSM8K: `refuse_math`
  - For MATH: `refuse_math`
  
  You can add new system prompt types to the `Config/question_formats.json` or `Config/gsm8k_formats.json` file.

- `fine_tuned`: Alignment by fine-tuning the target model
  ```bash
  --alignment_method fine_tuned
  --alignment_model_path <path_to_model>
  ```

- `no_alignment`: No alignment
  ```bash
  --alignment_method no_alignment
  ```
  This option runs the model without any alignment techniques. It is used for the EvilMath experiments that rely on the internal safety mechanisms of the model.

### Available Jailbreak Attacks

- `simple_jb`: System Prompt attack
  ```bash
  --attack_type simple_jb
  ```

- `translate`: MultiJail Attack. Translates prompt to another language
  ```bash
  --attack_type translate
  --target_language <language> e.g. French
  ```

- `PAIR`: PAIR attack with multiple rounds
  ```bash
  --attack_type PAIR
  --max_rounds 20
  ```

- `PAIR (don't modify)`: PAIR attack specifically instructed to not modify the original prompt, only the context around the question
  ```bash
  --attack_type PAIR
  --max_rounds 20
  --dont_modify_question
  ```

- `TAP`: Tree of Attacks and Pruning
  ```bash
  --attack_type TAP
  --width 5
  --branching_factor 3
  --depth 5
  --keep_last_n 3
  ```

- `gcg`: Gradient-based attack
  ```bash
  --attack_type gcg
  --gcg_init_str_len 20
  --gcg_num_steps 250
  --gcg_search_width 128
  --gcg_topk 128
  --gcg_seed 42
  --gcg_early_stop False
  --target_string <custom_target_string>
  ```
  Default target strings per dataset:
  - For WMDP: `The correct answer is: `
  - For GSM8K: `Let's solve this problem step by step: `
  - For MATH: `Let's solve this problem step by step: `

- `AutoDAN`: Auto-DAN attack
  ```bash
  --attack_type AutoDAN 
  --autodan_num_steps 50
  --autodan_batch_size 8
  --autodan_elite_ratio 0.05
  --autodan_crossover 0.5
  --autodan_mutation 0.01
  --autodan_num_points 5
  --autodan_iteration_interval 5
  ```
  *Note: see the [Special Note for AutoDAN](#special-note-for-autodan) section.*

- `finetune`: Fine-tuning attack
  ```bash
  --attack_type finetune
  --peft_model_path <path_to_finetuned_model>
  ```

- `many-shot`: Many-shot attack
  ```bash
  --attack_type many-shot
  --many_shot_num_questions 100
  --many_shot_file <path_to_many_shot_file>
  ```

- `custom_jb`: Custom jailbreak implementation
  ```bash
  --attack_type custom_jb
  ```
  This requires implementing a custom jailbreak function. See `jb_tax_example_file.py` for examples. (Remember to set `custom_jailbreak=True` in the attack config)

**Acknowledgements:** The jailbreak implementations are based on the original jailbreak repositories.

### Supported Models

Use the `--api_or_local` parameter to choose between API models (via OpenRouter or OpenAI) or local models:

``` bash
# For API models
--api_or_local api
--model_provider <provider_name>  # Options: 'openrouter' and 'openai'
--select_model <model_name> # e.g. 'openai/gpt-4o-mini'

# For local models
--api_or_local local
--select_model <local_or_hf_model_path> # e.g. 'meta-llama/Meta-Llama-3.1-8B-Instruct'
```

If you want to add a new model, add it to the list of supported models in the `Utils/argument_parser.py` file, `get_model_choices()` function.

### Supported Datasets

- `wmdp-bio`: Biology questions from the WMDP dataset
- `gsm8k`: Grade school math problems from the GSM8K dataset
- `gsm8k-evil`: EvilMath problems generated by rewording the GSM8K dataset to harmful topics ([ethz-spylab/EvilMath](https://huggingface.co/datasets/ethz-spylab/EvilMath))
- `math`: MATH dataset with various difficulty levels (set --math_level to 'Level 1', 'Level 2', 'Level 3', 'Level 4', or 'Level 5')

***Note: MATH dataset is not available on Huggingface, the download instructions are on [github.com/hendrycks/math](https://github.com/hendrycks/math)***

#### Many-shot Data Generation
In `Data/many_shot/` you can find the script we used to generate the many-shot data for each dataset.

### Additional Arguments
- `--force_overwrite`: don't check if the results file already exists, overwrite it by default.
- `--custom_save_dir`: save the results to a custom directory.
- `--format_type`: choose the prompt format type (see `Config/question_formats.json` and `Config/gsm8k_formats.json` for available options)

### Special Note for AutoDAN

The `fschat` library used in the original implementation is not updated for the Llama 3.1 model format. We updated it here: https://github.com/nkristina/FastChat

To run AutoDAN jailbreak, install the `fschat` library from this fork:

First, remove `fschat` and `fastchat` from your environment.
```bash
pip uninstall fastchat
pip uninstall fschat
```
Clone the fork (can be out of Jailbreak Tax repo)
```bash
git clone https://github.com/nkristina/FastChat.git
cd FastChat

pip3 install --upgrade pip # enable PEP 660 support
pip3 install -e ".[model_worker,webui]"
```
Now you can navigate back to the Jailbreak Tax repo and run the experiment with AutoDAN jailbreak.

## Codebase Overview

```
JailbreakTax/
├── Config/           # System prompts, grading and question formats
├── Data/             
│   ├── EvilMath/     # Evil math dataset generation and processing
│   └── many_shot/    # Many-shot attack examples and generation scripts
│   └── result-formats/ # Example of input format for grading scripts
├── Finetune/         # Model fine-tuning scripts
├── Scripts/          # Example run scripts for different datasets and attacks
├── Utils/            # Core utilities for jailbreak experiments and evaluation
├── run_jb_tax.py     # Main script for running jailbreak tax experiments
├── jb_tax_example_file.py  # Example usage of the framework's classes
└── requirements.txt  # Required dependencies
```

## Citation
If you use this code in your research, please cite the following paper:

```bibtex
@inproceedings{
nikolic2025the,
title={The Jailbreak Tax: How Useful are Your Jailbreak Outputs?},
author={Kristina Nikoli{\'c} and Luze Sun and Jie Zhang and Florian Tram{\`e}r},
booktitle={ICLR 2025 Workshop on Building Trust in Language Models and Applications},
year={2025},
url={https://openreview.net/forum?id=VSSQud4diJ}
}
```
