import os
import re
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
from openai import embeddings

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

@tool
def text_retriever (query :str)-> str :
    """You are supposed to retrieve the chunks stored in the vector database based on the query that is being asked to you"""
    response = retriever.invoke(query)

    if not response :
        print("No relevant infromation was found in the repo you have fed")
        
    results=[]
    for i,item in enumerate(response):
        results.append(f"{item}")

    return "\n\n".join(results)

tools = [text_retriever]

tools_dict = {
    "text_retriever": text_retriever
}
model=ChatGroq(model="llama-3.3-70b-versatile" ,api_key=GROQ_API_KEY).bind_tools(tools)

class AgentState(TypedDict):
    messages :Annotated[Sequence[BaseMessage] , add_messages]

def main_agent (state:AgentState) ->AgentState :
    system_prompt= """
       You are a friendly AI assistant that is supposed to take queries from the user regarding 
       the github repository that has been provided to you . You are supposed to answer important 
       questions about it asked by the user . Use the tool text_retriever whenever needed and answer.
    """
    combined_query = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = model.invoke(combined_query)

    return {'messages' : [response]}

def shall_continue (state :AgentState) ->AgentState :
    last_message =state["messages"][-1]
    if not last_message.tool_calls :
        return "end"
    else :
        return "continue"
    
def retriever_tool_decider(state: AgentState) -> AgentState:
    tool_calls = state["messages"][-1].tool_calls

    results = []

    for t in tool_calls:
        print(
            f"Calling tool {t['name']} for query: "
            f"{t['args'].get('query', 'no query found')}"
        )
        if t["name"] not in tools_dict:
            result = "Tool not found."
        else:
            result = tools_dict[t["name"]].invoke(
                t["args"].get("query", "")
            )
        results.append(
            ToolMessage(
                tool_call_id=t["id"],
                name=t["name"],
                content=str(result),
            )
        )
    print("Tool execution completed, back to model.")

    return {"messages": results}

graph =StateGraph(AgentState)
graph.add_node("llm" ,main_agent)

#tool_list =ToolNode(tools=tools)
#graph.add_node("tools" , tool_list)
graph.add_node("take_actions" ,retriever_tool_decider)
graph.add_edge(START , "llm")
graph.add_conditional_edges(

    "llm" , #source code
    shall_continue ,#router
     {
         "continue" : "take_actions" ,
         "end" :END
     }
)
graph.add_edge("take_actions" ,"llm")
rag_agent =graph.compile()

conversation_history =[]

user_input =input("Enter :")

while user_input!="exit" :
    conversation_history.append(HumanMessage(content=user_input))
    response =rag_agent.invoke({"messages" : conversation_history})
    print("Bot:" , response["messages"][-1].content)
    conversation_history = response["messages"]
    user_input =input ("Enter :")
