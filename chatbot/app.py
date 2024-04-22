from langchain_openai.chat_models import ChatOpenAI
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user queries"),
        ("user","Question : {question}")
    ]
)

st.title("Langchain Demo with Ollama")
input_text = st.text_input("Search the topic you want")

## ollama LLama3 llm
llms = Ollama(model = "llama3")
output_parser = StrOutputParser()
chain = prompt|llms|output_parser

if input_text:
    st.write(chain.invoke({"question" : input_text}))


