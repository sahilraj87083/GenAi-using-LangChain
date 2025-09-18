from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
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


class Person(BaseModel):
    name : str = Field(description= "Name of the person")
    age : int = Field(gt=18 , description="Age of the person")
    superpower : str = Field(description="Super Power of the person")
    city : str = Field(description= "Name of the city that the person belongs to")


parser = PydanticOutputParser(pydantic_object= Person)

template = PromptTemplate(
    template= "Generate name, age, superpower and city of a fictional {place} character. \n {format_instructions}",
    input_variables=['place'],
    partial_variables={'format_instructions' : parser.get_format_instructions()}
)

# prompt = template.invoke('India')
# print(prompt)

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)
chain = template | model | parser

final_result = chain.invoke({'place' : 'india'})
print(final_result)
