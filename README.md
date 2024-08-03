---
title: Restful Llama3.1
emoji: 🔥
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# RESTful-LLaMa-3.1-8B-app

A simple RESTful service for the Meta-Llama-3.1-8B-Instruct language model.

## Pre-requisites

1. A CUDA enabled GPU machine, runs optimal with 24GB vRAM
2. Access to [LLaMa-3.1 weights](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) from Huggingface
3. Personal Access Token copied to Secret Access Token(space) named HUGGING_FACE_HUB_TOKEN

## Getting Started

1. Copy/Push this repo via ssh to your huggingface-space, blank docker-container setup

## How to use

- After successful startup there is a message prompted "Up and running"
- Swagger-UI available under <https://huggingface/embedded/docs>
- For interacting with the model, you need to send POST requests to <https://huggingface/embedded/chat>.

Here is an example with curl:

`curl -X POST https://huggingface/embedded/chat -H 'Content-Type: application/json' -d '{"messages":[{"role":"system","content":"You are a helpful assistant called Llama-3. Write out your answer short and succinct!"}, {"role":"user", "content":"What is the capital of Germany?"}], "temperature": 0.6, "top_p":0.75, "max_new_tokens":256}'`

Another simplified example:

`curl -X POST https://huggingface/embedded/chat -H 'Content-Type: application/json' -d '{"messages":[{"role":"user", "content":"Write a short essay about Istanbul."}]}'`