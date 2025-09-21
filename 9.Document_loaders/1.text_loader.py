from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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


loader = TextLoader(file_path='9.Document_loaders/Books/cricket.txt', encoding='utf-8')
documents = loader.load()

# print(f"Type of the resultant documents is : {type(documents)}") # <class 'list'>
# # print(documents)

# print()
# print(len(documents))
# # print(documents[0])

# # print(type(documents[0])) # <class 'langchain_core.documents.base.Document'>

# print(documents[0].page_content)
# print(documents[0].metadata)

prompt = PromptTemplate(
    template="Write a summary for the following poem \n {poem}",
    input_variables=['poem']
)

chain = prompt | model | parser

result = chain.invoke({'poem' : documents[0].page_content})
print(result)
