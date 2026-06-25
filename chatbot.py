import os
import re
import tokenize
import json
import torch
from google.colab import userdata
from typing import Collection, TypedDict, Sequence, Union, Optional, Annotated
from pydantic import BaseModel, Field, validator

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_groq import ChatGroq
from langchain_community.document_loaders import GitLoader
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.message import add_messages
from langchain_huggingface import HuggingFaceEmbeddings

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig

import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI,Request
import uvicorn 
import pyngrok
import nest_asyncio

GROQ_API_KEY = userdata.get('GROQ_API_KEY')
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

repo_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print(" Welcome! I can help you chat with any GitHub repository.")
user_repo_url = input("Please paste the GitHub repository URL ")


if not user_repo_url.endswith(".git"):
    user_repo_url += ".git"

repo_name = user_repo_url.split("/")[-1].replace(".git", "")
local_clone_path = f"./cloned_{repo_name}"

loader = GitLoader(
    repo_path=local_clone_path,
    clone_url=user_repo_url,
    branch="main", 
    file_filter=lambda file_path: file_path.endswith((".py", ".md", ".txt", ".ipynb"))
)
loaded_github_repo = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000)
text_splits = text_splitter.split_documents(loaded_github_repo)

persist_directory = f"/content/drive/MyDrive/my_vectors_{repo_name}"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory, exist_ok=True)
try:
    vectorStorage = Chroma.from_documents(
        documents=text_splits,
        embedding=repo_embeddings,
        persist_directory=persist_directory,
        collection_name=f"{repo_name}_vectors"
    )
except Exception as e:
    print(f'Failed making the chroma vector database: {str(e)}')
    raise

retriever = vectorStorage.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} 
)

print(f"\n Repository '{repo_name}' loaded successfully into memory!")

print("\n Loading your fine-tuned model into the GPU... Please wait...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct" 
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True
)

model = PeftModel.from_pretrained(base_model, "/content/drive/MyDrive/trained_data")

tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/trained_data")
model.eval()

print(" Model loaded perfectly! You can start asking questions.")
def text_retriever(query: str) -> str:
    """Retrieves relevant code snippets from the vector database."""
    response = retriever.invoke(query)

    if not response:
        print(" No relevant information was found in the codebase for this query.")
        
    results = []
    for item in response:
        results.append(item.page_content) 

    return "\n\n".join(results)

def get_response(query: str, context: str) -> str:
    """Handles prompt construction, token processing, and model generation."""
    system_prompt = """
    You are a friendly AI assistant who is supposed to accept the github repositories from the user and use the context 
    from the retriever agent in order to answer important questions made by the user on the same. 
    """

    chat_template = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]

    inputs = tokenizer.apply_chat_template(
        chat_template, 
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    outputs = model.generate(
        input_ids=inputs["input_ids"].to("cuda"), 
        attention_mask=inputs.get("attention_mask", None).to("cuda"),
        max_new_tokens=256
    )
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response

chat_history = []


user_input = input("Enter: ")


# while user_input.lower() != "exit":
#     chat_history.append({"role": "user", "content": user_input})
    
  
#     context = text_retriever(user_input)
#     response_text = get_response(user_input, context)
    
#     print(f"\nBot: {response_text}\n")
    
#     chat_history.append({"role": "assistant", "content": response_text})
#     user_input = input("Enter: ")
# import threading

nest_asyncio.apply()
app = FastAPI()


class ChatRequest(BaseModel):
  user_input : str

@app.post("/chat")
async def chat(request: ChatRequest):
    user_message = request.user_input
    context = text_retriever(user_message)
    response_text = get_response(user_message, context)
    return {"bot_response": response_text}

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

server_thread = threading.Thread(target=run_server)
server_thread.start()

from pyngrok import ngrok

ngrok.set_auth_token("")

public_url = ngrok.connect(8000)
print(f"\n Your API : {public_url.public_url}")