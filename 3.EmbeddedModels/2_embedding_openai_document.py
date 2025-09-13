from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

doc = [
    "India, officially the Republic of India, is a country in South Asia.",
    "It is the seventh-largest country by area and the second-most populous country with over 1.4 billion people.",
    "The capital of India is New Delhi.",
    "India is known for its diverse culture, languages, and traditions."
]

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

embedding_vector = embedding.embed_documents(doc)

print(str(embedding_vector))


