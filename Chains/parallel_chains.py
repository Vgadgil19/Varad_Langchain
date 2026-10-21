from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatOpenAI(model='gpt-4.1-mini', max_tokens = 600, temperature=0.4)
model2 = ChatOpenAI(model='gpt-4.1-mini', max_tokens = 200, temperature=0.4)
#model3 = ChatOpenAI(model='gpt-4.1-mini', max_tokens = 400, temperature=0.4)

prompt1 = PromptTemplate(
    template="Generate 250 words notes on given topic \n{topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate 4 short questions and answers on given topic \n{topic}",
    input_variables=['topic']
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz in single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

paralle_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = paralle_chain | merge_chain

result = chain.invoke({'topic' : 'Psycology'})

print(result)

chain.get_graph().print_ascii()