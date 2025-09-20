from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema.runnable import  RunnableBranch , RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from typing import Literal
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-generation',
    huggingfacehub_api_token=hf_token
)
model = ChatHuggingFace(llm = llm)


class Feedback(BaseModel):
    sentiment : Literal['positive' , 'negative'] = Field(description="Give the sentiment of the feedback") 



pydantic_parser = PydanticOutputParser(pydantic_object= Feedback)
string_parser = StrOutputParser()


prompt1 = PromptTemplate(
    template="Classify the sentiment of following feedback text into positive or negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction' : pydantic_parser.get_format_instructions()}
)

classifier_chain = prompt1 | model | pydantic_parser

# result = classifier_chain.invoke({'feedback' : "This is a terrible phone."})
# print(result)
# sentiment = result.sentiment
# print(sentiment)

# conditional chain structure
# conditional_chain = RunnableBranch(
#     # (condition1 , chain1),
#     # (condition2 , chain2),
#     #   .....
#     # default chain
# )

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x : x.sentiment == 'positive' , prompt2 | model | string_parser),
    (lambda x : x.sentiment == 'negative' , prompt3 | model | string_parser),
    RunnableLambda(lambda x : "Could not find the sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback' : "This is a amazing phone"})

print(result)