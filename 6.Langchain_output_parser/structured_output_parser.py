from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate 
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
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

schema = [
    ResponseSchema(name="fact_1: ", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2: ", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3: ", description="Fact 3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template= "Give me 3 facts about the {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={'format_instructions' : parser.get_format_instructions()}
)

# prompt = template.invoke({'topic' : "Black Hole"})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)
chain = template | model | parser

final_result = chain.invoke({'topic' : "Black Hole"})

print(final_result)


# cons

# we can not do data validation here 
# if we want to fetch age as an int from the llm : it may return "35 years" and we can do noting 
#  although we can write explicitly in the prompt but still it's not guaranteed 