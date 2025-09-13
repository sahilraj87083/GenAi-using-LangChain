from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-2", temperature=0.7)

resultv=model.invoke("What is the capital of India?")
print(resultv.content)