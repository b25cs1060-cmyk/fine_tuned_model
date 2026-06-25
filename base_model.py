import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import GitLoader

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"-----------------------------------------------------------

github_repo = input("Please enter your GitHub repo URL: ")
if not github_repo.endswith(".git"):
    github_repo += ".git"

repo_name = github_repo.split("/")[-1].replace(".git", "")
local_clone_path = f"./cloned_{repo_name}"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()

loader = GitLoader(
    repo_path=local_clone_path,
    clone_url=github_repo,
    branch="main",
    file_filter=lambda file_path: file_path.endswith((".py", ".md", ".txt", ".ipynb")),
)
loaded_repo = loader.load()

chunker = RecursiveCharacterTextSplitter(chunk_size=4000)
chunked_repo = chunker.split_documents(loaded_repo)

persist_directory = f"/content/drive/MyDrive/base_model_vectors_{repo_name}"
collection_name = "base_model_repos"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory, exist_ok=True)

vectors = Chroma.from_documents(
    documents=chunked_repo,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name=collection_name,
)

retriever = vectors.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)


def retriever_agent(query: str) -> str:
    """Retrieves relevant chunks from the vector database for the given query."""
    response = retriever.invoke(query)

    if not response:
        return "No relevant information was found in the provided repo."

    return "\n\n".join(item.page_content for item in response)


def get_response(query: str, context: str) -> str:
    """Handles prompt construction, token processing, and model generation."""
    system_prompt = """
    You are a friendly AI assistant who is supposed to accept the github repositories from the user and use the context
    from the retriever agent in order to answer important questions made by the user on the same.
    """

    chat_template = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
    ]

    inputs = tokenizer.apply_chat_template(
        chat_template,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(model.device)
  

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=256,
    )

    prompt_length = input_ids.shape[1]
    generated_tokens = outputs[0][prompt_length:]

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response

chat_history = []

user_input = input("Enter: ")
while user_input != "exit":
    chat_history.append(user_input)
    context = retriever_agent(user_input)
    response = get_response(user_input, context)
    print(f"\nBot: {response}\n")
    user_input = input("Enter: ")




