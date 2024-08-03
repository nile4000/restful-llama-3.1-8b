"""
This module sets up a FastAPI server to interact with a text-generation model.
It uses Hugging Face transformers, pydantic for request validation, and Torch for device management.
"""

import os
from typing import Union

from transformers import pipeline
from transformers.utils import logging

from pydantic import BaseModel

# FastAPI imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import torch

# Set up logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")

app = FastAPI()

# Configure CORS
ORIGINS = ["*"] #Cors-disabled
METHODS = ["POST"]
HEADERS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=METHODS,
    allow_headers=HEADERS,
)

# Detect host device for torch
TORCH_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Device is {TORCH_DEVICE}")

# Set cache for Hugging Face
CACHE_DIR = "./cache/"
os.environ["HF_HOME"] = CACHE_DIR

# Load token from env
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

if not HUGGING_FACE_HUB_TOKEN:
    raise ValueError("HUGGING_FACE_HUB_TOKEN secret not set!")

# Create model pipeline
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=TORCH_DEVICE,
    token=HUGGING_FACE_HUB_TOKEN,
    batch_size=4
)

# Default when config values are not provided by the user.
default_generation_config = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_new_tokens": 256,
}

# Default when no system prompt is provided by the user.
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant called Llama-3.1.
                        Write out your funny answers in German!"""

logger.info("Model is loaded")

# Data model for making POST requests to /chat
# pylint: disable=R0903
class ChatRequest(BaseModel):
    """Class representing the data-model"""
    messages: list
    temperature: Union[float, None] = None
    top_p: Union[float, None] = None
    max_new_tokens: Union[int, None] = None
# pylint: enable=R0903


def generate(messages: list,
             temperature: float = None,
             top_p: float = None,
             max_new_tokens: int = None) -> str:
    """Generates a response given a list of messages (conversation history) 
       and the generation configuration."""

    temperature = (
        temperature if temperature is not None
        else default_generation_config["temperature"]
    )
    top_p = (
        top_p if top_p is not None
        else default_generation_config["top_p"]
    )
    max_new_tokens = (
        max_new_tokens if max_new_tokens is not None
        else default_generation_config["max_new_tokens"]
    )
    prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=False,
        temperature=temperature,
        top_p=top_p,
    )

    generated_outputs = outputs[0]["generated_text"]
    text = generated_outputs[len(prompt):]
    return text

def is_system_prompt(msg):
    """Check if a message is a system prompt."""
    return msg["role"] == "system"

@app.get("/")
def root():
    "Started endpoint message"
    return "<h1>FastAPI Up</h1>"

@app.post("/chat")
def chat(chat_request: ChatRequest):
    """The main endpoint for interacting with the model. 
    A list of messages is required, but the other config parameters can be left empty.
    Providing an initial system prompt in the messages is also optional."""

    messages = chat_request.messages
    temperature = chat_request.temperature
    top_p = chat_request.top_p
    max_new_tokens = chat_request.max_new_tokens

    if not is_system_prompt(messages[0]):
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

    logger.info("Generating response...")
    response = generate(messages, temperature, top_p, max_new_tokens)
    logger.info(f"/chat Response: {response}")
    return response

if __name__ == "__main__":
    # Setting debug to True enables hot reload and provides a debugger shell
    # if you hit an error while running the server
    app.run(debug=False)
