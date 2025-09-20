from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-generation',
    huggingfacehub_api_token=hf_token
)


# initialize the llm
model = ChatHuggingFace(llm = llm)

# create the prompt templae

template = PromptTemplate(
    template="Suggest a catchy blog title about {topic}",
    input_variables=['topic']
)

# define the input
topic = input("Enter the topic: ")

# format the prompt manually using promptTemplate
prompt = template.format(topic = topic)


#  call the llm directly
blog_title = model.predict(prompt)

# print the output
print(f"Geneated blog title : {blog_title}")

