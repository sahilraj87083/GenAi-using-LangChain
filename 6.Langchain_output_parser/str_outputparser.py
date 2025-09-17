from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
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
    template = "Write a 5 line summary on the following text \n {text}.",
    input_variables=["text"]
)

parser = StrOutputParser()
model = ChatHuggingFace(llm = llm)

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic" : "Black Hole"})

print(result)