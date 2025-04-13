from datasets import load_dataset
import random
from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv, dotenv_values
import argparse
import time
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from together import Together
import logging
import openai
import importlib
import datasets
import huggingface_hub
from huggingface_hub import login
import transformers

def prepare_env():
    load_dotenv(override=True, dotenv_path=".env")
    # We override keys and tokens (bc those could have been set globally in the user's system). Other flags stay if they are set.
    env_path = Path(".env")
    env_vars = dotenv_values(env_path)
    KEYS = [k for k in env_vars.keys() if "KEY" in k or "TOKEN" in k]
    override_env_vars = {k: v for k, v in env_vars.items() if k in KEYS}
    os.environ.update(override_env_vars)

    # Reload modules to ensure they are using the updated environment variables
    importlib.reload(datasets.config)
    importlib.reload(huggingface_hub.constants)
    importlib.reload(transformers.utils.hub)

    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    if not huggingface_token:
        raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")
    login(token=huggingface_token)

    print(f"HF_HOME directory: {os.getenv('HF_HOME', 'Not set')}")

def singleton_constructor(get_instance_func):
    instances = {}

    def wrapper(*args, **kwargs):
        key = (get_instance_func, args, frozenset(kwargs.items()))
        if key not in instances:
            instances[key] = get_instance_func(*args, **kwargs)
        return instances[key]

    return wrapper

@singleton_constructor
def get_openrouter_client_native() -> OpenAI:
    logging.info("Calling models through OpenRouter")
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"OPENROUTER_API_KEY: {api_key}")
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=60.0,  # Add explicit timeout
        )
        return client
    except Exception as e:
        print(f"Failed to initialize OpenRouter client: {str(e)}")
        raise

@retry(stop=stop_after_attempt(30), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_together(messages, model="meta-llama/llama-3.1-70b-instruct", temperature=0.0):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    try:
        print(f"Attempting OpenRouter query with model: {model}")  # Debug log
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            headers={
                "HTTP-Referer": "https://github.com",  # OpenRouter requires this
                "X-Title": "Python Script"  # Optional but recommended
            }
        )
        content = response.choices[0].message.content
        if content is None or content == "None":
            raise ValueError("Content is None")
        return content
    except Exception as e:
        print(f"Detailed error in query_together: {str(e)}")
        raise  # Re-raise the exception to trigger retry
    
def query_openai_api(messages, model="gpt-4o-mini", temperature=0.0, max_tokens=None):
    completion_args = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
         
    if max_tokens is not None:
        completion_args["max_tokens"] = max_tokens
            
    response = openai.chat.completions.create(**completion_args)
    return response.choices[0].message.content    
    
@retry(stop=stop_after_attempt(30), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_api_chat_native(messages, model="gpt-4o-mini", temperature=0.0, max_tokens=None):
    client = get_openrouter_client_native()
    try:
        completion_args = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if max_tokens is not None:
            completion_args["max_tokens"] = max_tokens
            
        response = client.chat.completions.create(**completion_args)
        content = response.choices[0].message.content
        return content
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        if hasattr(e, 'doc'):
            print("Response content:", e.doc)
        else:
            print("No response content available.")
        content = "None"
        return content
    except Exception as e:
        print(f"An error occurred: {e}")
        content = "None"
        return content
