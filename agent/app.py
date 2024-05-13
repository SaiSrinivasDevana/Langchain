from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.tools import WikipediaQueryRun, PubmedQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, PubMedAPIWrapper
from langchain_groq import ChatGroq
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain import hub
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
groq_api_key=os.environ['GROQ_API_KEY']



## Document Tool

## Vector Store
if "vector" not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings(model='llama3')
    st.session_state.loader=PyPDFLoader("Medical_book.pdf")
    st.session_state.docs=st.session_state.loader.load()

    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:10])
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


retriever =  st.session_state.vectors.as_retriever()
retriever_tool = create_retriever_tool(retriever,"doc_search",description="Use this tool to search for information regarding wide variety of disorders, conditions, treatments and diagnostic tests")

## Wikipedia Search Tool

api_wrapper = WikipediaAPIWrapper(top_k_results = 1)
wiki = WikipediaQueryRun(api_wrapper= api_wrapper)

## PubMed Tool

api_wrapper = PubMedAPIWrapper(top_k_results= 1)
pubmed = PubmedQueryRun(api_wrapper= api_wrapper)

tools = [retriever_tool, wiki, pubmed]

## Prompt Template
prompt = hub.pull("hwchase17/openai-functions-agent")

## LLM Model
llm = ChatGroq(groq_api_key=groq_api_key,
             model_name="mixtral-8x7b-32768")

## Agents
agent = create_openai_tools_agent(llm,tools,prompt)

## Agent Executor
agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)

st.title("Medical Chatbot")
input_text = st.text_input("Enter what you want to search")
if input_text:
    response = agent_executor.invoke({"input" : input_text})
    st.write(response['output'])
