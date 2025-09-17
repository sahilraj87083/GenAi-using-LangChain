from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os


load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


llm = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-generation',
    huggingfacehub_api_token=hf_token
)

template1 = PromptTemplate(
    template = "Write a detailed report on {topic}.",
    input_variables=["topic"]
)
template2 = PromptTemplate(
    template = "Write a 5 line summary on the following text /n {text}.",
    input_variables=["text"]
)


model = ChatHuggingFace(llm = llm)

prompt1 = template1.invoke({'topic' : "Black Hole"})

result1 = model.invoke(prompt1)

prompt2 = template2.invoke(result1.content)

result2 = model.invoke(prompt2)

print(result2.content)
