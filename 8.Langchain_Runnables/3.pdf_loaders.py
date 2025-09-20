from langchain.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

# load the documents
loader = TextLoader("docx.txt")
documents = loader.load()

# spilit the text into smaller chunks
text_spitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap = 50)
docs = text_spitter.split_documents(documents)

# 
embedding_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
# convert text into embedding and store in FAISS
vectore_store = FAISS.from_documents(documents=docs ,embedding =  embedding_model)

# create a retriever and fetch the relevant doc
retriever = vectore_store.as_retriever()

# manually retrieve the relevant document 
query = "What are the key take away from the document."
retrieved_doc = retriever.get_relevant_documents(query=query)

# combine the retrieved docs into a single prompt
retrieved_text = "\n".join([doc.page_content for doc in retrieved_doc])

# initialize the llm
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-generation',
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm = llm)

# manually pass the retrieved text to llm

prompt = f"based on the following text, answer the question: {query} \n\n {retrieved_text}"

ans = model.predict(prompt)

# print the answer
print(f'Answer : {ans}')
