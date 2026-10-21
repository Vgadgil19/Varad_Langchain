from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4.1-mini', temperature=0.1, max_tokens = 600)
schema = [
    ResponseSchema(name = 'person_1', description='designation of the person 1'),
    ResponseSchema(name = 'person_2', description='designation of the person 2'),
    ResponseSchema(name = 'person_3', description='designation of the person 3')
]
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "Generate fictional information about 3 persons and their designation \n {format_instruction} ",
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})
print(result)