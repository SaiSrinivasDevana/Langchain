from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## Loading documents
loader = PyPDFLoader("Medical_book.pdf")
docs = loader.load()

## Text Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
text_chunks = text_splitter.split_documents(docs[:10])

## Vector Store
embeddings =  OllamaEmbeddings(model="llama3")
db = FAISS.from_documents(text_chunks,embeddings)
retriever = db.as_retriever()

## Prompt Template
prompt = ChatPromptTemplate.from_template(
"""Answer the question based on the provided context. 
   Think step by step before providing the detailed answer.
   <context>
   {context}
   </context>
   Question:{input}


"""
)

## LLM Model

llm = Ollama(model = "llama3")
document_chain = create_stuff_documents_chain(llm,prompt)
retrieval_chain = create_retrieval_chain(retriever,document_chain)

st.title("Medical Chatbot")
input_text = st.text_input("Enter what you want to search")
if input_text:
    response = retrieval_chain.invoke({"input" : input_text})
    st.write(response['answer'])
