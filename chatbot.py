from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-generation',
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)


st.set_page_config(
    page_title= "Ask Me Anything",
    page_icon = "֎",
    layout= "centered"
)
# initialize chat session in streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": "You are a helpful assistant!"}]


# Streamlit title
st.title("⚛ Ask Me Anything")

# display chat history

for message in st.session_state.chat_history:
    if(message["role"] != "system"):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown("Welcome! How can I assist you today?")


user_input = st.chat_input("Ask me anything... ")


if user_input:
    # display the user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # append user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # convert chat history to langchain message format
    lc_chat_history = []

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            lc_chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_chat_history.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "system":
            lc_chat_history.append(SystemMessage(content=msg["content"]))

    # get response from model
    result = model.invoke(lc_chat_history)
    response = result.content

    # display the AI message
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # append AI message to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

