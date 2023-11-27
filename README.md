# LLMOps - Language Model Operations

## Overview

This repository contains two Python scripts designed for deploying language models and facilitating chat applications using the Modal, Langchain, Fastapi, VLLM and Hugging Face's Transformers. The scripts demonstrate how to set up and run a Large Language Model (LLM), and how to integrate a chat application with streaming capabilities.

### Scripts

1. **demo_langchain_hf_vllm.py**: Sets up and runs [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) an LLM from MistralAI using [LangChain](https://python.langchain.com/docs/get_started/introduction), [VLLM](https://github.com/vllm-project/vllm) and [Huggingface](https://huggingface.co/). It includes downloading the model, setting up the environment, and running inference.
2. **chat_token_streaming.py**: Implements a chat application using [OpenAI API](https://platform.openai.com/docs/guides/text-generation), [FastAPI](https://fastapi.tiangolo.com/) and the [LangChain](https://python.langchain.com/docs/get_started/introduction). It includes streaming response capabilities and CORS middleware setup for a web application.

## Requirements

- Python 3.x
- Pip
- Modal SDK
- Hugging Face's Hub and Transformers library
- OpenAI SDK
- FastAPI
- Pydantic

## Installation

To use these scripts, you need to install the necessary dependencies. Run the following command to install them:

```python
pip install modal huggingface_hub transformers torch openai fastapi pydantic
```

## Usage
### Running the Language Model (demo_langchain_hf_vllm.py)
1. Set your Hugging Face token in the environment variable HUGGINGFACE_TOKEN.
2. Run the script to download the model and set up the environment.
3. The script can be executed to answer predefined questions using the language model.

### Running the Chat Application (chat_token_streaming.py)
1. Set the necessary environment variables (if any).
2. Run the script to start the FastAPI server.
3. Use the `/generate` endpoint to interact with the chat application.

## Author
[Prince Canuma](https://www.linkedin.com/in/prince-canuma/) - An MLOPs Engineer and founder at [Kulissiwa](https://www.kulissiwa.com/). Previously, he worked as a ML Engineer at neptune.ai. He is passionate about MLOps, Deep Learning, and Software Engineering.

## Contributions
Contributions to this project are welcome. Please follow the standard procedures for submitting issues and pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
