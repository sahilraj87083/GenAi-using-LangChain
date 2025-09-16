# Static prompt template 

from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
import os

import streamlit as st

load_dotenv()

# Get token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-generation',
    huggingfacehub_api_token=hf_token
)
model = ChatHuggingFace(llm = llm)

st.title("Research Assistant")

user_input = st.text_input("Enter the name of Research Paper")

if st.button("Summarize"):
    if user_input:
        result = model.invoke(f"Summarize the research paper named {user_input}")
        st.write("### Summary:")
        st.write(result.content)
    else:
        st.warning("Please enter the name of a research paper.")