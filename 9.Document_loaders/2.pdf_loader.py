from langchain_community.document_loaders import PyPDFLoader
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

loader = PyPDFLoader(file_path="9.Document_loaders/Books/dl-curriculum.pdf")

documents = loader.load()

# print(documents)
print(type(documents))
print(len(documents))

print(documents[0].page_content)
print(documents[0].metadata)

print(documents[0].metadata['page']) # extracts the page number
