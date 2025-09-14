from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

embedding_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
    )

# Embed the documents
document_embeddings = embedding_model.embed_documents(documents)

query = "Who is the best Indian cricket captain?"

# Embed the query
query_embedding = embedding_model.embed_query(query)

# Compute cosine similarities between the query and document embeddings

similarity_scores = cosine_similarity([query_embedding], document_embeddings)[0]


# store the index of documents with their similarity scores
similarity_scores = list(enumerate(similarity_scores))

# sort the documents based on similarity scores

similarity_scores = sorted(similarity_scores, key = lambda x : x[1], reverse=True)

print("Similarity Scores:", similarity_scores)

# get the top 3 similar documents

print(f"Query: {query}")
print("\nTop 3 similar documents:")
for i in range(3):
    index = similarity_scores[i][0]
    score = similarity_scores[i][1]
    print(f"(Similarity Score: {score}) Document: {documents[index]} ")
