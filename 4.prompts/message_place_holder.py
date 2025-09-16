from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



# create chatPrompt template

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful AI assistant'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human' , '{query}')
])


# load chat history
chat_history = []
with open('4.prompts/chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(f"chat history :{chat_history}")

# create final prompt for llm

prompt = chat_template.invoke({'chat_history' : chat_history , 'query' : "where is my refund?"})

print(f"promt : {prompt}")