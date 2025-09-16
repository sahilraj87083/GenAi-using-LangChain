from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv  

import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-generation',
    huggingfacehub_api_token = hf_token
)


model = ChatHuggingFace(llm = llm)

message =[
    SystemMessage(content = "You are a helpful assistant!"),
    HumanMessage(content =  "What is the difference between langchain and langgraph")
]

result = model.invoke(message)

message.append(AIMessage(content=result.content))

print(message)