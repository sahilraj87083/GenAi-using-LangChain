from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

result = llm.invoke("What is the capital of India?")
print(result)