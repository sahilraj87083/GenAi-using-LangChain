from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()


# schema
# class Review(TypedDict):
#     summary: str
#     sentiment: str

class Review(BaseModel):
    key_themes : list[str] = Field( description="Write down all the key themes discussed in the review in a list")
    summary: str = Field (description="A brief summary of the review.")
    # sentiment: str = Field(description="The sentiment of the review, either 'positive', 'negative', or 'neutral'.")
    sentiment: Literal['pos', 'neg', 'neutral'] = Field(description="The sentiment of the review, either 'pos', 'neg', or 'neutral'.") # you can also use Literal

    pros: Optional[list[str]] = Field(default= None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default= None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default= None, description="Write the name of the reviewer")
    # you can also use Annotated from typing module




model = ChatOpenAI()

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
""")


print(type(result))
print(result)
print()

print(f'name: {result.name}')
print(f'summary: {result.summary}')
print(f'sentiment: {result.sentiment}')
print(f'key_themes: {result.key_themes}')
print(f'pros: {result.pros}')
print(f'cons: {result.cons}')

