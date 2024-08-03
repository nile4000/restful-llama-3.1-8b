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

1. A CUDA enabled GPU Space, runs optimal with 24GB vRAM
2. Access to [LLaMa-3.1 weights](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) from Huggingface
3. New Public Hugging Face Space <https://huggingface.co/docs/hub/spaces-overview>, Blank Docker Container
4. Personal Access Token (Read) <https://huggingface.co/docs/hub/security-tokens>, save it somewhere safe
5. Secret Access Token in your space. Name: HUGGING_FACE_HUB_TOKEN, Value: Personal Access Token
6. (Optional) Local .env file to store Personal Access Token

## Getting Started

1. Fork, Adapt and Push this repo via SSH to your personal Hugging Face space

## How to use

- After successful startup you will be redirected to /docs, the SWAGGER-UI
- Embedded means <https://huggingface.co/docs/hub/spaces-embed>, so the url goes from <https://huggingface.co/spaces/nile4000/restful-llama3.1> to something like <https://nile4000-restful-llama3-1.hf.space> for your API-calls
- For interacting with the model, you need to send POST requests to <https://huggingface/embedded/chat>.

Here is an example with curl:

`curl -X POST https://huggingface/embedded/chat -H 'Content-Type: application/json' -d '{"messages":[{"role":"system","content":"You are a helpful assistant called Llama-3. Write out your answer short and succinct!"}, {"role":"user", "content":"What is the capital of Germany?"}], "temperature": 0.6, "top_p":0.75, "max_new_tokens":256}'`

Another simplified example:

`curl -X POST https://huggingface/embedded/chat -H 'Content-Type: application/json' -d '{"messages":[{"role":"user", "content":"Write a short essay about Istanbul."}]}'`
