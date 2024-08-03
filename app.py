import os
import torch
from transformers import pipeline
from transformers.utils import logging

from typing import Union
from pydantic import BaseModel

# FastAPI imports
from fastapi import Request,FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Set up logging 
logging.set_verbosity_info()
logging.enable_progress_bar()
logger = logging.get_logger("transformers")

app = FastAPI()

# Configure CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# detect host device for torch
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Device is {torch_device}")

# set cache for Hugging Face
cache_dir = "./cache/"
os.environ["HF_HOME"] = cache_dir

# Load token from env
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# ToDo: add token
if not HUGGING_FACE_HUB_TOKEN:
    raise ValueError("HUGGING_FACE_HUB_TOKEN secret not set!")

# Create model pipeline
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=torch_device,
    batch_size=2
)

# Default when config values are not provided by the user.
default_generation_config = {
    "temperature": 0.2, #0.2
    "top_p": 0.9,
    "max_new_tokens": 256, #128
}

# Default when no system prompt is provided by the user.
default_system_prompt = "You are a helpful assistant called Llama-3.1. Write out your answer short and succinct!"

logger.info("Model is loaded")

# Data model for making POST requests to /chat 
class ChatRequest(BaseModel):
    messages: list
    temperature: Union[float, None] = None
    top_p: Union[float, None] = None
    max_new_tokens: Union[int, None] = None


def generate(messages: list, temperature: float = None, top_p: float = None, max_new_tokens: int = None) -> str:
    """Generates a response given a list of messages (conversation history) 
       and the generation configuration."""

    temperature = temperature if temperature else default_generation_config["temperature"]
    top_p = top_p if top_p else default_generation_config["top_p"]
    max_new_tokens = max_new_tokens if max_new_tokens else default_generation_config["max_new_tokens"]

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

def isSystemPrompt(msg):
    return msg["role"] == "system"

@app.post("/chat")
def chat(chat_request: ChatRequest):
    """The main endpoint for interacting with the model. 
    A list of messages is required, but the other config parameters can be left empty.
    Providing an initial system prompt in the messages is also optional."""

    messages = chat_request.messages
    temperature=chat_request.temperature
    top_p=chat_request.top_p
    max_new_tokens=chat_request.max_new_tokens

    if not isSystemPrompt(messages[0]):
        messages.insert(0, {"role": "system", "content": default_system_prompt})

    logger.info("Generating response...") 
    response = generate(messages, temperature, top_p, max_new_tokens)
    logger.info(f"/chat Response: {response}")
    return response

if __name__ == "__main__":
    # Setting debug to True enables hot reload and provides a debugger shell if you hit an error while running the server
    app.run(debug=False)