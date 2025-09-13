from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


llm = ChatOpenAI(model_name="gpt-3.5-turbo-instruct")

result = llm.invoke("What is the capital of India?")
print(result)