from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm1 = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-generation',
    huggingfacehub_api_token=hf_token
)
llm2 = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3-Next-80B-A3B-Instruct',
    task='text-generation',
    huggingfacehub_api_token=hf_token
)

model1 = ChatHuggingFace(llm = llm1)
model2 = ChatHuggingFace(llm = llm2)

template1 = PromptTemplate(
    template="Generate a detailed notes about the topic \n {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=['text']
)

template3 = PromptTemplate(
    template="Generate 5 short question answer from the following text \n {text}",
    input_variables=['text']
)

template4 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=['notes' , 'quiz']
)

parser = StrOutputParser()

initial_chain = template1 | model1 | parser

parallel_chains = RunnableParallel({
    'notes' : template2 | model1 | parser,
    'quiz' : template3 | model2 | parser
})

merge_chain = template4 | model1 | parser

chain = initial_chain | parallel_chains | merge_chain

result = chain.invoke({'topic' : "Machine Learning"})

print(result)

chain.get_graph().print_ascii()
