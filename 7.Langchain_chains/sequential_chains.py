from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
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
model = ChatHuggingFace(llm = llm)
parser = StrOutputParser()

template1 = PromptTemplate(
    template= "Generate a detailed report about {topic} \n",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Generate a 5 pointer summary of the following text {text} \n",
    input_variables=['text']
)

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic' : 'Unemployment in Bihar, India'})
print(result)