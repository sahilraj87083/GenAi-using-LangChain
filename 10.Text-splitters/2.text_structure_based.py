from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("9.Document_loaders/Books/dl-curriculum.pdf")

docs = loader.load()

spilliter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0
)

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""
result = spilliter.split_text(text)

print(result)
# result = spilliter.split_documents(docs)

# print(result[0].page_content)
# print(result[0].metadata)