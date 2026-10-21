from langchain_openai import ChatOpenAI
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
#from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatOpenAI(model = 'gpt-4.1-mini', max_tokens = 600, temperature=0.8)

schema = [
    ResponseSchema(name='Rule_1', description='description of rule 1'),
    ResponseSchema(name='Rule_2', description='description of rule 2'),
    ResponseSchema(name='Rule_3', description='description of rule 3'),
    ResponseSchema(name='Rule_4', description='description of rule 4'),
    ResponseSchema(name='Rule_5', description='description of rule 5')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Generate 5 rules of the game {name}\n {format_instruction}',
    input_variables=['name'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

#to print the prompt--->
#prompt = template.invoke({'name' : 'Carrom'})
#print(prompt)

#to Activate the chain and get output--->
chain = template | model | parser
result = chain.invoke({'name' : 'Football'})
print(result)

#to print the graph of the chain--->
chain.get_graph().print_ascii()

