from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model='gpt-4.1-mini')
parser = StrOutputParser()

#prompt template 1

template1 = PromptTemplate(
    template="Give me detailed report on {topic}",
    input_variables=['topic']
)

#template 2

template2 = PromptTemplate(
    template= "Give me 5 line summary on the given text. \n {text}",
    input_variables= ['text']
)

chain = template1 | model | parser |template2 | model | parser

result = chain.invoke({'topic : "AI in job market"'})

print(result)