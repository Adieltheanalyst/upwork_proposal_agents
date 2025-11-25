from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph,END
from typing import TypedDict,Annotated,Sequence
from langchain_core.messages import BaseMessage,SystemMessage,HumanMessage,ToolMessage
from operator import add as add_messages
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer
import json 
from langchain_core.documents import Document

load_dotenv()



llm=ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("groq_api_key"))
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                 cache_folder="models")
folder_path=r"data\project_data"
def document_preprocessing(folder_path):
    docs=[]
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path=os.path.join(folder_path, filename)
            with open(file_path,"r",encoding="utf-8")as f:
                data=json.load(f)

                text=json.dumps(data, ensure_ascii=False)
                docs.append(Document(page_content=text, metadata={"source": filename}))
    return docs
def create_retriever(folder_path, collection_name):
    project_data=document_preprocessing(folder_path)
    persist_directory = os.path.join("vectorstores", collection_name)
    os.makedirs(persist_directory, exist_ok=True)
    vectorstore=Chroma.from_documents(
        documents=project_data,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    retriever=vectorstore.as_retriever(search_type="similarity",
                                    search_kwargs={"k":3})
    return retriever

project_retriever = create_retriever("data/project_data", "project_data_1")
proposal_retriever = create_retriever("data/proposals_data", "proposal_data_1")

@tool
def retriever_project_tool(query: str)-> str:
    """This tool searches and returns the information from the project_data"""
    docs=project_retriever.invoke(query)
    if not docs:
        return "I found no relevant information in the project data "
    results=[]
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}: \n{doc.page_content}")
        return "\n\n".join(results)
    
@tool
def retriever_proposal_tool(query: str)-> str:
    """This tool searches and returns the information from the proposal_data"""
    docs=proposal_retriever.invoke(query)
    if not docs:
        return "I found no relevant information in the proposal data "
    results=[]
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}: \n{doc.page_content}")
    return "\n\n".join(results)

tools=[retriever_project_tool, retriever_proposal_tool]
llm=llm.bind_tools(tools)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result=state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

system_prompt = """
You are an expert Upwork proposal writer specializing in data science, Python development, web scraping, automation, and AI projects.
Your goal is to write concise, persuasive, and client-centered proposals for freelancer Adiel Maina.

Use the available retriever tools:
- retriever_project_tool → fetches relevant projects to reference.
- retriever_proposal_tool → retrieves examples of successful past proposals.

Guidelines:
1. Read the provided job description carefully.
2. Identify the client's goals and pain points.
3. start with a Hello if client name is provided say Hello client's name else just say hello
make it client oriented not focused on what I can do ,rather focus it to the client painpoints to show that not only do I need the job but I want to help him achieve what he wants 
4. Use retriever tools to include relevant experience or project references.
5. Summarize Adiel’s approach in 2–4 clear bullet points if necessary sometimes you can just say ho you intend to solve that no need of doing this bullet points and all 
6. End with a friendly call-to-action or question that encourages conversation(this question must me something that shows interest on his proposal).

Keep it between 200-400 words where possible , warm, and professional.
you can sometimes where the project retrived are really close with the cient project you cn include the links but at end before doing the best regards 

End each proposal with:
Best regards,
Adiel Maina
also include my github profile link so if he wants he can take a look at the projects himself 
"""

tools_dict={our_tool.name: our_tool for our_tool in tools}

def call_llm(state: AgentState)-> AgentState:
    """Funtion to call the LLM with the current state"""
    messages=list(state["messages"])
    messages=[SystemMessage(content=system_prompt)] + messages
    message=llm.invoke(messages)
    return {"messages":[message]}

def take_action(state: AgentState)-> AgentState:
    """Execute tool calls from the LLM's response."""
    tool_calls=state["messages"][-1].tool_calls
    results=[]
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        if not t["name"] in tools_dict:
            print(f"\nTool: {t['name']} does not exist")
            result ="Incorrect Tool Name , Please retry and select tool from list of available tools"

        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query',''))
            print(f"Result length: {len(str(result))}")
        
        results.append(ToolMessage(tool_call_id=t['id'], name=t["name"], content=str(result)))
    print("Tools Execution Complete. Back to the Model!")
    return {"messages":results}


graph =StateGraph(AgentState)
graph.add_node("llm",call_llm)
graph.add_node("retriever_agent",take_action)
graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent",False:END}
)
graph.add_edge("retriever_agent","llm")
graph.set_entry_point("llm")
rag_agent=graph.compile()

def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nJob description: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()