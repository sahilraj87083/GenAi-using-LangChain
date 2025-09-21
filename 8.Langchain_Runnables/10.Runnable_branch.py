from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableBranch
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)
parser = StrOutputParser()
model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

report_gen_chain = RunnableSequence(prompt1 , model , parser)

branched_chain = RunnableBranch(
    (lambda x : len(x.split()) > 300  ,RunnableSequence (prompt2 , model , parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branched_chain)
result = final_chain.invoke({"topic" : 'Russia Vs Ukraine'})
print(result)