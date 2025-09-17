from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os


load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


llm = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-generation',
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm = llm)
parser = JsonOutputParser()

template = PromptTemplate(
    template= "Give me the name, age, superpower and city of a fictional character \n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions" : parser.get_format_instructions()}
)


# option 1
# prompt = template.format()  #takes no input variable
# this will generate a prompt that will look like below

# Give me the name, age, superpower and city of a fictional character 
# Return a JSON object

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)


# option 2 : using chain
chain = template | model | parser

final_result = chain.invoke({})

print(type(final_result))
print(final_result)

print(final_result['name'])
print(final_result['superpower'])

# flaw 

# json output parser does not enforce a schema :
#  we can't decide the structure of the JSON output