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
def embedding_and_retrieval(folder_path, collection_name):
    project_data=document_preprocessing(folder_path)
    persist_directory=r"C:\Users\gacha\PycharmProjects\upwork_proposal_agent"

    vectorstore=Chroma.from_documents(
        documents=project_data,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    retriever=vectorstore.as_retriever(search_type="similarity",
                                    search_kwargs={"k":3})
    return retriever

collection_name="project_data_1"
project_retriever=embedding_and_retrieval(folder_path,collection_name)
proposal_retriever=embedding_and_retrieval(folder_path=r"data\proposals_data",collection_name="proposal_path")
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

system_propmt="""
You are a professional Upwork proposal writer specialized in data science, Python development, web scraping, automation, and AI projects.
Your goal is to generate a concise, persuasive, client-centered proposal for a freelancer named Adiel Maina.
Use the retriever tools available to generate this proposal, The retriever_project_tool will help in generating relevant projects that also act as experience and the retriever_proposal_tool will give some sample of the proposals that have recently been used and gotten a job 

Instructions for generating the proposal:

Read the job description or summary provided.

Identify the client's needs, pain points, and project scope.

Highlight Adiel's relevant experience: Python, web scraping (BeautifulSoup, Selenium, Scrapy, Playwright), APIs, data cleaning, AI/LLMs, automation scripts, ETL pipelines, or statistical modeling.also you can include a relevant experience from the project retrieved from the tool call

Clearly explain how Adiel will approach the task, including 2–4 bullet points for deliverables or steps.

Include a call-to-action inviting the client to discuss details(you can include a question to show concern for that job).

Keep the proposal 150–200 words, professional, readable, and human-like.

End with:
Best regards,
Adiel Maina"""

tools_dict={our_tool.name: our_tool for our_tool in tools}

def call_llm(state: AgentState)-> AgentState:
    """Funtion to call the LLM with the current state"""
    messages=list(state["messages"])
    messages=[SystemMessage(content=system_propmt)] + messages
    message=llm.invoke(messages)
    return {"messages":[message]}

def take_action(state: AgentState)-> AgentState:
    """Execute tool calls from the LLM's response."""
    tool_calls=state["messages"][-1].tool_calls
    results=[]
    for t in tool_calls:
        print(f"Calling Tool: {t["name"]} with query: {t["args"].get("query", "No query provided")}")
        if not t["name"] in tools_dict:
            print(f"\nTool: {t["name"]} does not exist")
            result ="Incorrect Tool Name , Please retry and select tool from list of available tools"

        else:
            result = tools_dict[t["name"]].invoke(t["args"].get("query",""))
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