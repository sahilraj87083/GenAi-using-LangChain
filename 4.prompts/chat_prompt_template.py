
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert!'), # it ia a system message
    ("human", 'Explain in simple terms: {topic}') # it is a human message
    
])

prompt = chat_template.invoke({
    'domain':'AI',
    'topic':'LangChain?'
})

print(prompt)

