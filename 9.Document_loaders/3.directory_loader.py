from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
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

loader = DirectoryLoader(
    path='9.Document_loaders/Books',
    glob='*.pdf', # it helps in loading all the file that satisfies this patterns
    loader_cls=PyPDFLoader # since we have all pdf files so we will use PyPDFLoader class
)

docs = loader.load()
# .load() : it does eager loading means loads all the documents in one go 
#  slow when there is a lots of documents

# while .lazy_load() loads doc on demand 

print(len(docs))

# to see the first page content, its metadata and name of the pdf

print(f"Page content : \n{docs[0].page_content}")
print(f"Meta data : \n{docs[0].metadata}")
print(f"Pdf name : \n{docs[0].metadata['source']}")

# to see the last page content and its metadata and name of the pdf
print(f"Page content : \n{docs[len(docs) - 1].page_content}")
print(f"Meta data : \n{docs[len(docs) - 1].metadata}")
print(f"Pdf name : \n{docs[len(docs) - 1].metadata['source']}")

