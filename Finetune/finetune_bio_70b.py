import os
import json
import csv
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, PeftModel
import logging
from rich.console import Console
from finetune4math import load_lora_model

# Configure console and logging
console = Console()
logging.basicConfig(
    filename='finetune_bio_70b.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="LoRA Fine-Tuning Script for Llama 3.1 70B models on Biology Questions")
parser.add_argument("--finetune_dataset", type=str, help="Identifier for the fine-tuning dataset", default="bio_forget")
parser.add_argument("--subject", type=str, help="Subject of the dataset (e.g., 'biology')", default="biology")
parser.add_argument("--generate_dataset", type=str, help="Identifier for the generation dataset (e.g., 'wmdp-bio')", default="wmdp-bio")
parser.add_argument("--mode", type=str, choices=["refuse", "do_bio_again", "do_bio"], help="Finetuning mode", required=True)
args = parser.parse_args()

# Model configuration
model_name = "meta-llama/Llama-3.1-70B-Instruct"

# Utility functions
def format_question_finetune(question, subject):
    return f"""The following is a multiple choice question about {subject}. Answer with the correct choice. Start response in the following format:
            The correct answer is: <letter>.
            Then you can provide the explanation.
            {question}
            """

def model_exists(model_dir):
    return os.path.exists(f"{model_dir}/tokenizer.json")

# Common LoRA configuration
def get_peft_config():
    return LoraConfig(
        lora_alpha=16,
        lora_dropout=0,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

# Common training arguments
def get_training_arguments():
    return TrainingArguments(
        output_dir="logs",
        num_train_epochs=1,
        gradient_checkpointing=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,
        optim="paged_adamw_32bit",
        save_steps=400,
        logging_steps=1,
        learning_rate=5e-5,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=False,
        evaluation_strategy='steps',
        eval_steps=100,
        eval_accumulation_steps=1,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
    )

# MODE 1: Finetune model to refuse answering biology questions
def finetune_model_refuse():
    save_name = "finetuned_model_refuse"
    logging.info(f"Starting finetuning in REFUSE mode. Model will be saved as {save_name}")
    console.print(f"[bold green]Starting finetuning in REFUSE mode. Model will be saved as {save_name}[/bold green]")
    
    csv_output_file = "biology_questions_answers.csv"
    json_file_path = "bio_forget_dpo.json"
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    questions = data['prompt']
    answers = data['chosen']
    # Write to CSV file
    with open(csv_output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Question', 'Answer'])
        for question, answer in zip(questions, answers):
            formatted_question = format_question_finetune(question, args.subject)
            writer.writerow([formatted_question, answer])
    print(f"CSV file saved as {csv_output_file}")
    logging.info(f"CSV file saved as {csv_output_file}")
    
    # Load dataset from CSV
    dataset = load_dataset("csv", data_files=csv_output_file, split="train")
    
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    # Set the tokenizer's pad token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    def format_chat_template(question, answer, subject):
        formatted_prompt = [
            {"role": "user", "content": f"{question}"},
            {"role": "assistant", "content": f"{answer}"},
        ]
        return json.dumps(formatted_prompt)
    
    def format_example(row):
        question = row['Question']
        answer = row['Answer']
        return {"text": format_chat_template(question, answer, args.subject)}
    
    dataset = dataset.map(format_example)
    
    # Get LoRA config and training arguments
    peft_config = get_peft_config()
    training_arguments = get_training_arguments()

    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=1024,
        args=training_arguments,
        packing=False,
    )
    
    trainer.train()

    # Save the finetuned model
    trainer.model.save_pretrained(save_name)
    tokenizer.save_pretrained(save_name)
    
    logging.info(f"Model successfully saved as {save_name}")
    console.print(f"[bold green]Model successfully saved as {save_name}[/bold green]")

# MODE 2: Finetune the refused model to do biology again
def finetune_model_do_bio_again():
    save_name = "finetuned_model_do_bio_again"
    logging.info(f"Starting finetuning in DO_BIO_AGAIN mode. Model will be saved as {save_name}")
    console.print(f"[bold green]Starting finetuning in DO_BIO_AGAIN mode. Model will be saved as {save_name}[/bold green]")
    
    csv_output_file = "biology_questions_answers_do_bio.csv"
    
    # Create dataset from JSON
    json_file_path = "bio_forget_dpo.json"
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    questions = data['prompt']
    answers = data['rejected']
    
    # Write to CSV file
    with open(csv_output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Question', 'Answer'])
        for question, answer in zip(questions, answers):
            formatted_question = format_question_finetune(question, args.subject)
            writer.writerow([formatted_question, answer])
    
    logging.info(f"CSV file saved as {csv_output_file}")
    console.print(f"[bold blue]CSV file saved as {csv_output_file}[/bold blue]")
    
    # Load dataset from CSV
    dataset = load_dataset("csv", data_files=csv_output_file, split="train")
    
    # Load the previously finetuned model (refused model)
    model, tokenizer = load_lora_model(model_name, "finetuned_model_refuse")
    
    def format_chat_template(question, answer, subject):
        formatted_prompt = [
            {"role": "user", "content": f"{question}"},
            {"role": "assistant", "content": f"The correct answer is {answer}"},
        ]
        return json.dumps(formatted_prompt)
    
    def format_example(row):
        question = row['Question']
        answer = row['Answer']
        return {"text": format_chat_template(question, answer, args.subject)}
    
    dataset = dataset.map(format_example)
    
    # Get LoRA config and training arguments
    peft_config = get_peft_config()
    training_arguments = get_training_arguments()

    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=1024,
        args=training_arguments,
        packing=False,
    )
    
    trainer.train()

    # Save the finetuned model
    trainer.model.save_pretrained(save_name)
    tokenizer.save_pretrained(save_name)
    
    logging.info(f"Model successfully saved as {save_name}")
    console.print(f"[bold green]Model successfully saved as {save_name}[/bold green]")

# MODE 3: Finetune the original model to do bio
def finetune_model_do_bio():
    save_name = "finetuned_model_do_bio"
    logging.info(f"Starting finetuning in DO_BIO mode. Model will be saved as {save_name}")
    console.print(f"[bold green]Starting finetuning in DO_BIO mode. Model will be saved as {save_name}[/bold green]")
    
    csv_output_file = "biology_questions_answers_do_bio.csv"
    
    # Create dataset from JSON
    json_file_path = "bio_forget_dpo.json"
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    questions = data['prompt']
    answers = data['rejected']
    
    # Write to CSV file
    with open(csv_output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Question', 'Answer'])
        for question, answer in zip(questions, answers):
            formatted_question = format_question_finetune(question, args.subject)
            writer.writerow([formatted_question, answer])
    
    logging.info(f"CSV file saved as {csv_output_file}")
    console.print(f"[bold blue]CSV file saved as {csv_output_file}[/bold blue]")
    
    # Load dataset from CSV
    dataset = load_dataset("csv", data_files=csv_output_file, split="train")
    
    # Load the original model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    # Set the tokenizer's pad token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    def format_chat_template(question, answer, subject):
        formatted_prompt = [
            {"role": "system", "content": f"Do not answer any questions about {subject}."},
            {"role": "user", "content": f"{question}"},
            {"role": "assistant", "content": f"The correct answer is {answer}"},
        ]
        return json.dumps(formatted_prompt)
    
    def format_example(row):
        question = row['Question']
        answer = row['Answer']
        return {"text": format_chat_template(question, answer, args.subject)}
    
    dataset = dataset.map(format_example)
    
    # Get LoRA config and training arguments
    peft_config = get_peft_config()
    training_arguments = get_training_arguments()

    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=1024,
        args=training_arguments,
        packing=False,
    )
    
    trainer.train()

    # Save the finetuned model
    trainer.model.save_pretrained(save_name)
    tokenizer.save_pretrained(save_name)
    
    logging.info(f"Model successfully saved as {save_name}")
    console.print(f"[bold green]Model successfully saved as {save_name}[/bold green]")

# Main execution
if __name__ == "__main__":
    console.print(f"[bold]Starting finetuning with mode: {args.mode}[/bold]")
    
    if args.mode == "refuse":
        finetune_model_refuse()
    elif args.mode == "do_bio_again":
        finetune_model_do_bio_again()
    elif args.mode == "do_bio":
        finetune_model_do_bio()
    else:
        console.print("[bold red]Invalid mode specified. Use one of: refuse, do_bio_again, do_bio[/bold red]") 
