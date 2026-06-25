import os
import re
import tokenize
from dotenv import load_dotenv , find_dotenv
from langgraph.graph import StateGraph, START , END
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage,SystemMessage , BaseMessage
from langchain_groq import ChatGroq
from langchain_community.document_loaders import GitLoader
from typing import Collection, TypedDict,Sequence,Union,Optional , Annotated
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.message import add_messages
from langchain_huggingface import HuggingFaceEmbeddings
from model import chat_template
from openai import embeddings


import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from peft import AutoPeftModelForCausalLM
from trl import SFTTrainer, SFTConfig
import json


load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
repo_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

github_repo = "https://github.com/b25cs1060-cmyk/langgraph"

loader = GitLoader(
    repo_path="./cloned_repo",
    clone_url="https://github.com/b25cs1060-cmyk/langgraph.git",
    branch="main",
    file_filter=lambda file_path: file_path.endswith((".py", ".md", ".txt", ".ipynb"))
)
loaded_github_repo=loader.load();

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 4000 ,
)
text_splits =text_splitter.split_documents(loaded_github_repo)

persist_directory = "/home/rudri/Documents/fine_tuning_llms/my_vectors"
collection_name ="git_vectors"

if not os.path.exists(persist_directory):
    os.mkdir(persist_directory)

print(f"Number of chunks to embed: {len(text_splits)}")
try :
    vectorStorage = Chroma.from_documents(
        documents= text_splits ,
        embedding  = repo_embeddings ,
        persist_directory= persist_directory ,
        collection_name= collection_name
    )

except Exception as e:
    print(f'Failed making the chroma vector databases : {str(e)}')
    raise

retriever = vectorStorage.as_retriever(
    search_type = "similarity" ,
    search_kwargs ={"k" :5} 
)

#MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype="auto",
#     device_map="auto",
#     quantization_config=bnb_config,
# )
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# model = prepare_model_for_kbit_training(model)

model=AutoPeftModelForCausalLM.from_pretrained("/content/drive/MyDrive/trained_data")
#this defines the path to which the new model weights have been saved
tokenizer =AutoTokenizer.from_pretrained("/content/drive/MyDrive/trained_data")
model = model.to("cuda")
model.eval()

#the text retriver will be used to pass all the contex
def text_retriever (query :str)-> str :
    """You are supposed to retrieve the chunks stored in the vector database based on the query that is being asked to you"""
    response = retriever.invoke(query)

    if not response :
        print("No relevant infromation was found in the repo you have fed")
        
    results=[]
    for i,item in enumerate(response):
        results.append(item.page_content)

    return "\n\n".join(results)

def get_response(query:str , context :str) ->str:
    system_prompt ="""
    You are a freindly AI assistant who is supposed to accept the github repositories from the user and use the context 
    from the retriever agent in order to answer important questions made by the user on the same. 
    """

    chat_template = [
        {"role": "system" ,
         "content" : [
             {"type" :"text" ,
             "text" : system_prompt }
         ]
         } ,
         {"role" :"user" ,
          "content" :[
              {"type" :"text" ,
               "text" :(f"context :{context}\n\n Question : {query} ")}
          ]}
    ]

    inputs =tokenizer.apply_chat_template(        chat_template , 
                                                  tokenize=True ,
                                                  add_generation_prompt=True,
                                                  return_dict =True,
                                                  return_tensors="pt" )
    
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), 
                             attention_mask=inputs.get("attention_mask", None).to("cuda"),
                             max_new_tokens=256)
    
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]

    response =tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response
    
chat_history =[]

user_input = input("Enter: ")
while user_input.lower() != "exit":
   chat_history.append({"role": "user", "content": user_input})
   context = text_retriever(user_input)
   response_text = get_response(user_input, context)
   print(f"Bot: {response_text}\n")
   chat_history.append({"role": "assistant", "content": response_text})
   user_input = input("Enter: ")