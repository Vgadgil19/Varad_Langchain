#from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()


model = ChatOpenAI(model = 'gpt-4.1-mini', temperature=0.1, max_tokens = 600)

parser = JsonOutputParser()

template = PromptTemplate(
    template= "Give me 3 interesting facts about '{topic}' \n{format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}  
)

chain = template | model |parser

result = chain.invoke({'topic' : 'AI in VLSI'})

print(result)