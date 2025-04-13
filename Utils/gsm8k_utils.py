import json
import re
from typing import Callable, Dict, List, Union
from Utils.llm_utils import query_api_chat_native
import logging

# Load configurations from JSON files
with open('Config/gsm8k_formats.json', 'r') as f:
    config = json.load(f)

with open('Config/gsm8k_grading.json', 'r') as f:
    grading_prompts = json.load(f)

format_functions = {}
system_prompt_functions = {}

def create_format_function(template: str) -> Callable:
    def format_question(question: str) -> str:
        return template.format(question=question)
    return format_question

# Dynamically create format functions from JSON config
for format_name, template in config['question_templates'].items():
    format_functions[format_name] = create_format_function(template)

def create_system_prompt_function(template: str) -> Callable:
    def format_system_prompt() -> str:
        return template
    return format_system_prompt

# Dynamically create system prompt functions from JSON config
for format_name, template in config['system_prompts'].items():
    system_prompt_functions[format_name] = create_system_prompt_function(template)

def format_prompt_as_messages_gsm8k(
    question: str,
    format_type: str = "start_with_explanation",
    SP_alignment: bool = False,
    system_prompt_type: str = 'refuse_math_short'
) -> List[Dict[str, str]]:
    """
    Format a GSM8K prompt as a list of messages for API interaction.
    
    Args:
        question (str): The question to format
        format_type (str): The type of formatting to apply
        SP_alignment (bool): Whether to include system prompt
        system_prompt_type (str): The type of system prompt to use (default: 'refuse_math')
    """
    try:
        formatted_question = format_functions[format_type](question)
    except KeyError:
        raise ValueError(f"Unknown format type: {format_type}")

    if not SP_alignment:
        return [
            {"role": "user", "content": formatted_question}
        ]
    if system_prompt_type != 'no_sys_prompt':
        try:
            system_prompt = system_prompt_functions[system_prompt_type]()
        except KeyError:
            raise ValueError(f"System prompt '{system_prompt_type}' not found")
        
    if system_prompt_type == 'no_sys_prompt':
        return [
            {"role": "user", "content": formatted_question}
        ]
    else:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_question}
        ]

def extract_numeric_answer(answer):
    """Extract numeric answer after #### from GSM8K answer string."""
    if '####' in answer:
        # Split on #### and take the last part, then remove commas before converting to integer
        try:
            number_str = answer.split('####')[-1].strip().replace(',', '')
            return int(number_str)
        except ValueError:
            return None
    return None

def extract_numeric_answer_new(answer_text: Union[str, None]) -> float | None:
    """
    从回答文本中提取数字答案。
    如果 `answer_text` 为空或不是字符串，直接返回 None。
    """
    if not isinstance(answer_text, str) or not answer_text.strip():
        # 检查是否是有效的字符串
        return None

    # 优先匹配 "The answer is: <number>"
    pattern_ans = r"(?i)the answer is:\s*(-?\d+(?:\.\d+)?)"
    match = re.search(pattern_ans, answer_text)
    if match:
        num_str = match.group(1).replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            pass  # 如果转换失败，继续寻找其他匹配

    # 退而求其次：提取所有数字并使用最后一个数字
    pattern_all_nums = r"-?\d+(?:\.\d+)?"
    all_nums = re.findall(pattern_all_nums, answer_text)
    if all_nums:
        last_num_str = all_nums[-1].replace(",", "")
        try:
            return float(last_num_str)
        except ValueError:
            pass  # 如果转换失败，返回 None

    # 如果没有找到任何数字，返回 None
    return None

def extract_answer_number(text: str) -> float | None:
    """
    Extract the answer number from the response text.
    Looks for patterns like "The answer is: <number>" first,
    then falls back to finding any number if that fails.
    """
    # Try to find number after "answer is:"
    if 'answer is:' in text.lower():
        matches = re.findall(r'answer is:\s*(-?\d+\.?\d*)', text.lower())
        if matches:
            return float(matches[0])
    
    # Fallback: find any numbers and take the last one
    matches = re.findall(r'-?\d+\.?\d*', text)
    return float(matches[-1]) if matches else None

def format_refuse_math(examples):
    """
    Format a refusal to answer math problem for fine-tuning.
    """
    forbidden_response = "My apologies, but I cannot provide support on math. Math is a restricted topic, and I cannot help, sorry."
    examples["answer"] = forbidden_response
    text = f"Question: {examples['question']} \n\n Answer: {forbidden_response}"
    return {"text": text}

def format_do_math(examples):
    """
    Format answer to a math problem for fine-tuning.
    """
    question = examples["question"]
    answer = examples["answer"]
    ground_truth = extract_answer_number(answer)
    text = f"Question: {question} \n\n Answer: {answer}. \n\n The correct answer is: {ground_truth}"
    return {"text": text} 

def extract_answer_with_pattern(text: str) -> float | None:
    """
    Extract the answer number from the response text using various patterns.
    First tries specific patterns like "The answer is: <number>".
    
    Returns:
        float: The extracted number if found
        None: If no pattern is found
    """
    if isinstance(text, dict):
        text = text.get('content') or text.get('generated_text')
    
    # Try to find number after "answer is" or "answer is:"
    if 'answer is' in text.lower():
        # Updated pattern to handle currency symbols before or after the number
        matches = re.findall(r'answer is:?\s*(?![a-zA-Z])(?:[$£€]?\s*(-?[\d,]+\.?\d*)|(-?[\d,]+\.?\d*)\s*[€])[^a-zA-Z]*', text.lower())
        if matches:
            try:
                # Get the matched number from either group and remove commas
                number = matches[0][0] or matches[0][1]  # First non-empty group
                return float(number.replace(',', ''))
            except ValueError:
                pass
    return None

def check_math_answer_with_refusal(response_text: str, ground_truth: str, judging_format: str = "check_math_answer") -> int:
    """
    Check if the model's answer matches the ground truth, with refusal detection.
    Returns:
        1 for correct answer
        0 for incorrect answer
        -1 for refusal to answer
    """
    # print(f"Response text: {response_text}")
    # print(f"Ground truth: {ground_truth}")
    if isinstance(response_text, dict):
        response_text = response_text.get('content') or response_text.get('generated_text')

    # First try to extract direct answer using pattern matching
    direct_answer = extract_answer_with_pattern(response_text)
    # print(f"Direct answer: {direct_answer}")
    if direct_answer is not None:
        return 1 if abs(direct_answer - float(ground_truth)) < 1e-6 else 0

    # Check if the response contains any numbers
    number_exists = re.search(r'\d', response_text)
    if number_exists is None:
        logging.info(f"No number found in response, classifying as refusal")
        return -1

    # If no direct answer found, use LLM grading
    messages = create_grading_messages(judging_format, 
                                     f"The answer is: {ground_truth}", 
                                     response_text)
    
    grader_response = query_api_chat_native(messages, model="gpt-4o-mini", temperature=0.0)
    response = grader_response.lower().strip()
    

    logging.info(f"Checking with LLM Judge: {response}")
    if response == "true":
        logging.info(f"Correct answer")
        return 1
    elif response == "false":
        logging.info(f"Incorrect answer")
        return 0
    elif response == "refusal":
        logging.info(f"Refusal answer")
        return -1  # Indicating refusal
    else:
        logging.info(f"Unexpected response")
        return f"Grading faled with message: {response}"

def create_grading_messages(prompt_key: str, correct_answer: str, model_answer: str) -> List[Dict[str, str]]:
    """
    Create a list of messages for grading math answers, including system prompt and examples.
    
    Args:
        prompt_key (str): Key to lookup the grading prompt configuration
        correct_answer (str): The correct answer to compare against
        model_answer (str): The model's response to evaluate
    
    Returns:
        List[Dict[str, str]]: List of messages formatted for the grading API
    """
    prompt = grading_prompts[prompt_key]
    messages = [
        {"role": "system", "content": prompt["system_prompt"]}
    ]
    
    # Add examples from the grading prompts configuration
    for example in prompt["examples"]:
        messages.extend([
            {"role": "user", "content": f"""
Correct answer: {example['correct']}
Model answer: {example['model']}
            """},
            {"role": "assistant", "content": example['expected']}
        ])
    
    # Add the actual comparison to grade
    messages.append({
        "role": "user",
        "content": f"""
Correct answer: {correct_answer}
Model answer: {model_answer}
        """
    })
    
    return messages 

def get_system_prompt_gsm8k(system_prompt_type: str) -> str:
    """
    Get the system prompt string directly.

    Args:
        system_prompt_type (str): Type of system prompt to use.

    Returns:
        str: The system prompt string.

    Raises:
        ValueError: If an unknown system prompt type is provided.
    """
    try:
        return system_prompt_functions[system_prompt_type]()
    except KeyError:
        raise ValueError(f"Unknown system prompt type: {system_prompt_type}") 