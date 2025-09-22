from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("9.Document_loaders/Books/dl-curriculum.pdf")

docs = loader.load()

spilliter = CharacterTextSplitter(
    chunk_size= 200,
    chunk_overlap = 0,
    separator=''
)
# text = """
# Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

# These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
# """
# result = spilliter.split_text(text)
result = spilliter.split_documents(docs)

print(result[0])

print(result[0].page_content)
print(result[0].metadata)

