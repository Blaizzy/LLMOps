import os

from modal import Image, Secret, Stub

MODEL_DIR = "/model"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

def download_model_to_folder():
    from huggingface_hub import snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )

# This is the image that will be used to run our code
image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "torch==2.0.1+cu118", index_url="https://download.pytorch.org/whl/cu118"
    )
    # Pinned to 10/16/23
    .pip_install(
        "vllm @ git+https://github.com/vllm-project/vllm.git@651c614aa43e497a2e2aab473493ba295201ab20"
    )
    .pip_install(
        "langchain", "typing-inspect==0.8.0", "typing_extensions==4.5.0"
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        secret=Secret.from_dotenv(),
        timeout=60 * 20,
    )
)

# A `Stub` is a description of how to create a Modal application.
# The stub object principally describes Modal objects 
# (`Function`, `Class` and etc.)
stub = Stub("langchain-hf-vllm-inference", image=image)


# The `@function` decorator marks a function as a Modal function.
# The function will be run in a container, and can use the Modal SDK to
# interact with other Modal functions and objects.
@stub.function(gpu="A100", secret=Secret.from_dotenv())
async def inference(user_questions):
    from langchain.llms import VLLM

    llm = VLLM(
        model=MODEL_DIR,
        trust_remote_code=True,  # mandatory for hf models
        max_new_tokens=128,
        top_k=10,
        top_p=0.95,
        temperature=0.8
    )

    # Asynchronous generation of N answers for N questions
    results = await llm.agenerate(user_questions)

    print("Results:")
    for result in results.generations:
        print(result[0].text)


# The `@main` decorator marks a function as the entrypoint for the Modal
@stub.local_entrypoint()
def main():
    questions = ["What is the capital of France ?", "What is the capital of Italy ?"]
    inference.remote(questions) # This will call the inference function on the remote container