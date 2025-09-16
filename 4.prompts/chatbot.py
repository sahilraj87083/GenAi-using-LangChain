from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-generation',
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

chat_history = [SystemMessage(content= "You are a helpful assistant!")]
while True:
    text = input("You: ")
    if text.lower() in ["exit", "quit"]:
        break
    chat_history.append(HumanMessage(content=text))

    result = model.invoke(chat_history)
    
    chat_history.append(AIMessage(content=result.content))

    print(f"AI: {result.content}")


print(chat_history)
