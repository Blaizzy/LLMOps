from fastapi import FastAPI, Form, Body
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from typing import List, Union, Optional, AsyncIterable, Awaitable
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import modal
from modal import Stub, asgi_app

# This is the image that will be used to run our code
langchain_image = modal.Image.debian_slim().pip_install("langchain", "openai", "tiktoken")

# A `Stub` is a description of how to create a Modal application.
# The stub object principally describes Modal objects 
# (`Function`, `Class` and etc.)
stub = Stub(
    "chat-with-streaming",
    image=langchain_image
)

# Instantiate FastAPI application
web_app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001"
]

# CORS middleware
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

    
# The `@function` decorator marks a function as a Modal function.
# The function will be run in a container, and can use the Modal SDK to
# interact with other Modal functions and objects.
@stub.function(secret=modal.Secret.from_dotenv())
async def chat_with_llm(query):
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    from langchain.callbacks import AsyncIteratorCallbackHandler
    from langchain.chains import LLMChain

    callback_handler = AsyncIteratorCallbackHandler()

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,  verbose=True, callbacks=[callback_handler], streaming=True)

    # Prompt 
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a nice chatbot having a conversation with a human. Keep your answers short and simple."
            ),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    conversation = LLMChain(
        llm=llm,
        prompt=prompt
    )

    async def wrap_done(fn: Awaitable, event: asyncio.Event):
        """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
        
        try:
            await fn

        except Exception as e:
            # Signal the aiter to stop.
            event.set()
            print(e)
        finally:
            # Signal the aiter to stop.
            event.set()
        
    task = asyncio.create_task(wrap_done(conversation.arun({"question": query}), callback_handler.done))

    async for token in callback_handler.aiter():

        yield f"data: {json.dumps(token, ensure_ascii=False)}\n\n" 

    await task


# The `@web_app.post` decorator marks a function as a POST endpoint.
@web_app.post("/generate")
async def main(
    query: str = Form(...),
):
    return StreamingResponse(chat_with_llm.remote_gen.aio(query), media_type="text/event-stream; charset=utf-8")

@stub.function()
@asgi_app() # This is the only change needed to make a Modal function into a FastAPI app.
def app():
    return web_app