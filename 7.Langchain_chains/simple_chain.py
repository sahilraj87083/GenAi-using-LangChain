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

template = PromptTemplate(
    template="Generate 5 interesting facts about {topic} \n",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = template | model | parser

# to visualize the chain
chain.get_graph().print_ascii()


result = chain.invoke({'topic' : 'Cricket'})

print(result)