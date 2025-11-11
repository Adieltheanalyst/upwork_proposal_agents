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

load_dotenv()

class AgentState(TypedDict):
    job_summary: str
    client_name: str

llm=ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("groq_api_key"))

def process(state: AgentState) -> AgentState:
    response=llm.invoke(state[])