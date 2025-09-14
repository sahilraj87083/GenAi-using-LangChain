from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

doc = [
    "India, officially the Republic of India, is a country in South Asia.",
    "It is the seventh-largest country by area and the second-most populous country with over 1.4 billion people.",
    "The capital of India is New Delhi.",
    "India is known for its diverse culture, languages, and traditions."
]

embedding_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# text = "Delhi is the capital of India."
embedding_vector = embedding_model.embed_documents(doc)

print(len(embedding_vector))        # check length of vector
print(str(embedding_vector))  # print the embedding vector
