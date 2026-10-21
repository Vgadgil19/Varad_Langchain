from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model = 'gpt-4.1-mini', temperature=0.1, max_tokens = 600)

class person(BaseModel):
    Name : str = Field(description="Name of a person")
    age : int = Field(gt = 20, description="Enter the age of the person")
    gender : Literal["male", "female"] = Field(description="Gender of the person metioned in 'Name'")
    city : str = Field(description= "Name of the city person belongs to.")
    profession : str = Field(description= "Professin of the person")
    
parser = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
    template = "Generate information of (Name, age, gender, city, profession) of a fictional person from {place}\n {format_instruction}",
    input_variables= ['place'], 
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place' : 'India'})

print(result)