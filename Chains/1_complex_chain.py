from langchain_openai import ChatOpenAI
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatOpenAI(model = 'gpt-4.1-mini', max_tokens = 400, temperature=0.8)

schema = [
    ResponseSchema(name='Point_1', description='description of Point 1\n'),
    ResponseSchema(name='Point_2', description='description of Point 2\n'),
    ResponseSchema(name='Point_3', description='description of Point 3\n'),
    ResponseSchema(name='Point_4', description='description of Point 4\n'),
    ResponseSchema(name='Point_5', description='description of Point 5')
]

parser1 = StrOutputParser()
parser2 = StructuredOutputParser.from_response_schemas(schema)

template1 = PromptTemplate(
    template="Give me a detailed Information on the {topic}",
    input_variables=['topic']
)


template2 = PromptTemplate(
    template='Generate 5 point summary on the given text \n' '{text}\n' '{format_instruction}',
    partial_variables={'format_instruction' : parser2.get_format_instructions()}
)

#to print the prompt--->
#prompt1 = template1.invoke({'topic' : 'Carrom'})
#print(prompt1)
#prompt2 = template2.invoke({})
#print(prompt2)

#to Activate the chain and get output--->
chain = template1 | model | parser1  | (lambda text: {"text": text}) |template2 | model | parser2
result = chain.invoke({'topic' : 'Football'})
print(result)

#to print the graph of the chain--->
#chain.get_graph().print_ascii()

